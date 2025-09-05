from termcolor import colored
import numpy as np


def print_boolean_comparison(tensor_a, tensor_b):
    """
    Creates an ASCII art representation comparing two boolean tensors.

    Args:
        tensor_a: First boolean tensor of shape (1, h, w)
        tensor_b: Second boolean tensor of shape (1, h, w)

    Returns:
        A string containing ASCII art representation where:
        - 'X' where both tensors are True
        - 'A' where only tensor_a is True
        - 'B' where only tensor_b is True
        - ' ' where both tensors are False
    """
    # Ensure tensors are boolean and in CPU numpy format
    tensor_a = tensor_a.squeeze(0).bool().cpu().numpy()
    tensor_b = tensor_b.squeeze(0).bool().cpu().numpy()

    # Create result array as characters
    result = np.full(tensor_a.shape, ".", dtype=str)

    # Use termcolor for coloring

    # Define constants for better readability and maintainability
    BOTH_TRUE = "0"
    ONLY_A_TRUE = "1"
    ONLY_B_TRUE = "2"
    BOTH_FALSE = "."

    # Set values based on conditions
    result[np.logical_and(tensor_a, tensor_b)] = BOTH_TRUE
    result[np.logical_and(tensor_a, ~tensor_b)] = ONLY_A_TRUE
    result[np.logical_and(~tensor_a, tensor_b)] = ONLY_B_TRUE
    # Both false is already set by default with "."

    # Convert to ASCII art string with colors
    lines = []
    for row in result:
        colored_row = []
        for char in row:
            color = None
            if char == BOTH_TRUE:
                color = "green"
            elif char == ONLY_A_TRUE:
                color = "yellow"
            elif char == ONLY_B_TRUE:
                color = "red"
            else:  # BOTH_FALSE
                color = "dark_grey"

            colored_row.append(colored(char, color))
        lines.append("".join(colored_row))

    ascii_art = "\n".join(lines)

    print(ascii_art)

    return


def visualize_tensor_distributions(tensor1, tensor2, title1="Tensor 1", title2="Tensor 2"):
    """
    Visualizes the distribution of values in two tensors.

    Args:
        tensor1: First tensor to visualize
        tensor2: Second tensor to visualize
        title1: Title for the first tensor's histogram
        title2: Title for the second tensor's histogram

    Returns:
        matplotlib.axes.Axes: Axes object containing the plots
    """
    import ttnn
    import matplotlib.pyplot as plt

    if isinstance(tensor1, ttnn.Tensor):
        tensor1 = ttnn.to_torch(tensor1)
    if isinstance(tensor2, ttnn.Tensor):
        tensor2 = ttnn.to_torch(tensor2)

    # Flatten tensors to 1D
    t1_flat = tensor1.float().flatten().detach().cpu().numpy()
    t2_flat = tensor2.float().flatten().detach().cpu().numpy()

    # Calculate statistics
    t1_mean, t1_std = t1_flat.mean(), t1_flat.std()
    t2_mean, t2_std = t2_flat.mean(), t2_flat.std()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot histogram for tensor1
    ax1.hist(t1_flat, bins=50, alpha=0.7)
    ax1.axvline(t1_mean, color="r", linestyle="--", label=f"Mean: {t1_mean:.4f}")
    ax1.axvline(t1_mean + t1_std, color="g", linestyle=":", label=f"Std: {t1_std:.4f}")
    ax1.axvline(t1_mean - t1_std, color="g", linestyle=":")
    ax1.set_title(f"{title1}\nMean: {t1_mean:.4f}, Std: {t1_std:.4f}")
    ax1.legend()

    # Plot histogram for tensor2
    ax2.hist(t2_flat, bins=50, alpha=0.7)
    ax2.axvline(t2_mean, color="r", linestyle="--", label=f"Mean: {t2_mean:.4f}")
    ax2.axvline(t2_mean + t2_std, color="g", linestyle=":", label=f"Std: {t2_std:.4f}")
    ax2.axvline(t2_mean - t2_std, color="g", linestyle=":")
    ax2.set_title(f"{title2}\nMean: {t2_mean:.4f}, Std: {t2_std:.4f}")
    ax2.legend()

    plt.tight_layout()
    return fig
