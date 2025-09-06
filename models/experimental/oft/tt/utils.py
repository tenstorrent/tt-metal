from termcolor import colored
import numpy as np


def plot_boolean_comparison(tensor_a, tensor_b):
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

    # Set values based on conditions
    result[np.logical_and(tensor_a, tensor_b)] = "X"  # both True
    result[np.logical_and(tensor_a, ~tensor_b)] = "A"  # only tensor_a True
    result[np.logical_and(~tensor_a, tensor_b)] = "B"  # only tensor_b True

    # Convert to ASCII art string with colors
    lines = []
    for row in result:
        colored_row = []
        for char in row:
            if char == "X":
                colored_row.append(colored("O", "green"))
            elif char == "A":
                colored_row.append(colored("X", "yellow"))
            elif char == "B":
                colored_row.append(colored("X", "red"))
            else:
                colored_row.append(colored(char, "dark_grey"))
        lines.append("".join(colored_row))

    ascii_art = "\n".join(lines)
    print("ASCII Art Comparison:")
    print("X: Both tensors are True")
    print("A: Only first tensor is True")
    print("B: Only second tensor is True")
    print(" : Both tensors are False")
    print("-" * 50)
    print(ascii_art)

    return result
