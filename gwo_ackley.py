# Import visualization library
import numpy as np
import matplotlib.pyplot as plt

# Import Ackley benchmark function
def ackley_function(x):
    """
    Ackley function: f(x) = -20*exp(-0.2*sqrt(0.5*(x[0]^2 + x[1]^2))) - exp(0.5*(cos(2*pi*x[0]) + cos(2*pi*x[1]))) + e + 20
    This is a common benchmark function with many local minima and one global minimum at (0,0)
    """
    a = 20  # Constant parameter
    b = 0.2  # Constant parameter
    c = 2 * np.pi  # Constant parameter
    
    # First term: controls the exponential decay
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x[0]**2 + x[1]**2)))
    # Second term: controls the cosine wave
    term2 = -np.exp(0.5 * (np.cos(c * x[0]) + np.cos(c * x[1])))
    return term1 + term2 + a + np.exp(1)  # Final function value

# STEP 1: Subclass GWO to log best fitness each iteration
class LoggingGWO:
    def __init__(self, population_size=30, max_iter=100, dim=2):
        """
        Initialize the GWO algorithm with logging capabilities
        """
        self.num_wolves = population_size
        self.max_iter = max_iter
        self.dim = dim
        self.bounds = [-5, 5]  # Search space bounds for Ackley function
        
        # Initialize random positions and fitness history
        self.positions = np.random.uniform(self.bounds[0], self.bounds[1], 
                                         (self.num_wolves, self.dim))
        self.fitness = np.array([ackley_function(pos) for pos in self.positions])
        self.fitness_history = []
        
        # Initialize leader wolves
        sorted_indices = np.argsort(self.fitness)
        self.alpha_pos = self.positions[sorted_indices[0]]
        self.alpha_score = self.fitness[sorted_indices[0]]
        self.beta_pos = self.positions[sorted_indices[1]]
        self.beta_score = self.fitness[sorted_indices[1]]
        self.delta_pos = self.positions[sorted_indices[2]]
        self.delta_score = self.fitness[sorted_indices[2]]

# STEP 2: Initialize the benchmark problem
def initialize_swarm(seed=42, dimension=2):
    # Initialize GWO optimizer with population size and random seed
    np.random.seed(seed)
    gwo = LoggingGWO(population_size=30, max_iter=100, dim=dimension)
    return gwo

# STEP 3: Communication logic 
def update_leader_positions(gwo):
    """Update alpha, beta, delta positions based on fitness"""
    sorted_indices = np.argsort(gwo.fitness)
    gwo.alpha_pos = gwo.positions[sorted_indices[0]]
    gwo.alpha_score = gwo.fitness[sorted_indices[0]]
    gwo.beta_pos = gwo.positions[sorted_indices[1]]
    gwo.beta_score = gwo.fitness[sorted_indices[1]]
    gwo.delta_pos = gwo.positions[sorted_indices[2]]
    gwo.delta_score = gwo.fitness[sorted_indices[2]]

# STEP 4: Run optimization and track progress
def manual_evolve(gwo):
    """Execute GWO optimization process"""
    for iteration in range(gwo.max_iter):
        a = 2 * (1 - iteration / gwo.max_iter)
        
        for i in range(gwo.num_wolves):
            # Update wolf positions based on alpha, beta, delta
            r1, r2 = np.random.rand(2)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            
            r1, r2 = np.random.rand(2)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            
            r1, r2 = np.random.rand(2)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            
            D_alpha = abs(C1 * gwo.alpha_pos - gwo.positions[i])
            D_beta = abs(C2 * gwo.beta_pos - gwo.positions[i])
            D_delta = abs(C3 * gwo.delta_pos - gwo.positions[i])
            
            X1 = gwo.alpha_pos - A1 * D_alpha
            X2 = gwo.beta_pos - A2 * D_beta
            X3 = gwo.delta_pos - A3 * D_delta
            
            new_position = np.clip((X1 + X2 + X3) / 3, gwo.bounds[0], gwo.bounds[1])
            
            new_fitness = ackley_function(new_position)
            if new_fitness < gwo.fitness[i]:
                gwo.positions[i] = new_position
                gwo.fitness[i] = new_fitness
        
        update_leader_positions(gwo)
        gwo.fitness_history.append(gwo.alpha_score)
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Best fitness: {gwo.alpha_score}")
            
    return gwo.alpha_pos, gwo.alpha_score, gwo.fitness_history

# STEP 5: Termination check placeholder
def meet_termination_conditions():
    pass

# STEP 6: Visualize and report optimization results
def display_solution(best_solution, best_fitness, fitness_history):
    print("\nOptimization Results:")
    print(f"Best position found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history, marker="o", linestyle="-")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("GWO Convergence on Ackley Function")
    plt.grid(True)
    plt.savefig('gwo_convergence.png')
    plt.close()

# Main optimization routine
gwo = initialize_swarm()
best_solution, best_fitness, fitness_history = manual_evolve(gwo)
meet_termination_conditions()
display_solution(best_solution, best_fitness, fitness_history)

    # Explaination of the code:
    # 1. The code is a implementation of the Grey Wolf Optimizer (GWO) algorithm for the Ackley function.
    # 2. The Ackley function is a common benchmark function with many local minima and one global minimum at (0,0).
    # 3. The GWO algorithm is a population-based optimization algorithm that is inspired by the social behavior of grey wolves.
    # 4. The code is a simple implementation of the GWO algorithm for the Ackley function.

    # Explaination of the GWO algorithm:
    # 1. The GWO algorithm is a population-based optimization algorithm that is inspired by the social behavior of grey wolves.
    # 2. The algorithm starts by initializing a population of wolves in the search space.
    # 3. Each wolf represents a potential solution to the optimization problem.
    # 4. The wolves are then iteratively updated based on the fitness of the best wolves in the population.
    # 5. The wolves are updated based on the positions of the best wolves in the population.

    #comparison between GWO and ABC and PSO and ACO:
    # 1. GWO is a population-based optimization algorithm that is inspired by the social behavior of grey wolves.
    # 2. ABC is a population-based optimization algorithm that is inspired by the behavior of honey bees.
    # 3. PSO is a population-based optimization algorithm that is inspired by the behavior of birds.
    # 4. ACO is a population-based optimization algorithm that is inspired by the behavior of ants.

    # Conclusion about the GWO algorithm:
    # 1. The GWO algorithm is a population-based optimization algorithm that is inspired by the social behavior of grey wolves.
    # 2. The algorithm starts by initializing a population of wolves in the search space.
    # 3. Each wolf represents a potential solution to the optimization problem.
    # 4. The wolves are then iteratively updated based on the fitness of the best wolves in the population.
    # 5. The wolves are updated based on the positions of the best wolves in the population.


# Convergence Plot (gwo_convergence.png):

    # X-axis: Iteration number (0 to 100)
    # Y-axis: Best fitness value found
    # The plot should show a decreasing trend
    # Initial rapid improvement followed by slower convergence
    # The curve should flatten out near the end, indicating convergence


# Interpretation of the Algorithm:
    # Initialization:
        # 30 wolves are randomly placed in the search space [-5, 5] × [-5, 5]
        # Each wolf's position represents a potential solution
        # The fitness of each wolf is calculated using the Ackley function
    # Optimization Process:
        # The algorithm simulates the hunting behavior of grey wolves
        # Alpha (best), beta (second best), and delta (third best) wolves guide the search
        # Other wolves update their positions based on the leaders' positions
        # The coefficient 'a' decreases linearly, shifting from exploration to exploitation
    # Convergence:
        # The plot shows how the algorithm converges to the optimal solution
        # Initial rapid improvement indicates exploration of the search space
        # Later slower improvement shows fine-tuning of the solution
        # The final position should be close to the global minimum at (0,0)
    # Performance:
        # The algorithm's performance can be judged by:
        # How close the final position is to (0,0)
        # How close the final fitness is to 0
        # How quickly the algorithm converges (steepness of the convergence curve)


