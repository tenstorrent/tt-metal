import matplotlib.pyplot as plt


def plot_colored_points(data):
    """
    Plots points from a 2D array of tuples.
    Each tuple contains (x, y, condition) and is colored green if condition is True, red otherwise.

    Parameters:
    data (list of list of tuples): 2D array where each tuple is (x, y, condition).
    """
    # Flatten the 2D list to a single list of points
    points = [point for row in data for point in row]

    # Separate into lists based on condition
    x_true = [x for (x, y, cond) in points if cond]
    y_true = [y for (x, y, cond) in points if cond]

    x_false = [x for (x, y, cond) in points if not cond]
    y_false = [y for (x, y, cond) in points if not cond]

    # Plot the points
    plt.figure(figsize=(8, 6))
    plt.scatter(x_true, y_true, color="green", label="True")
    plt.scatter(x_false, y_false, color="red", label="False")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Colored Points Based on Condition")
    plt.legend()
    plt.grid(True)
    plt.show()
