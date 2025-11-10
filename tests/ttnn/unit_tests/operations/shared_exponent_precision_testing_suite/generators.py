# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch


# helper functions
def fa_rand(shape) -> torch.Tensor:
    """
    Generate tensor with mixed Gaussian distribution for testing numerical precision.

    Creates a tensor that combines a standard normal distribution with sparse high-variance
    outliers. This pattern is useful for testing how operations handle data with both
    typical values and occasional extreme outliers.

    Args:
        shape: Tuple specifying the desired tensor dimensions

    Returns:
        torch.Tensor: Tensor with shape `shape` containing:
                     - Base values from N(0,1) normal distribution
                     - Sparse outliers (0.1% probability) from N(0,10) distribution

    Note:
        This is the original fa_rand implementation with fixed parameters.
        For configurable parameters, use fa_rand_custom instead.
    """
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def fa_rand_custom(shape, base_std=1.0, outlier_std=10.0, outlier_prob=0.001) -> torch.Tensor:
    """
    Generate tensor with customizable mixed Gaussian distribution for precision testing.

    Creates a tensor combining a base normal distribution with configurable sparse outliers.
    This allows fine-tuning the distribution characteristics to test specific precision
    scenarios and shared exponent behavior.

    Args:
        shape: Tuple specifying the desired tensor dimensions
        base_std: Standard deviation for the base normal distribution (default: 1.0)
        outlier_std: Standard deviation for the outlier normal distribution (default: 10.0)
        outlier_prob: Probability of each element being an outlier (default: 0.001, i.e., 0.1%)

    Returns:
        torch.Tensor: Tensor with mixed distribution where:
                     - Most values come from N(0, base_std²)
                     - Sparse outliers come from N(0, outlier_std²) with probability outlier_prob

    Example:
        # Create aggressive outlier pattern for testing
        tensor = fa_rand_custom((32, 32), base_std=1.0, outlier_std=100.0, outlier_prob=0.01)
    """
    normal_base = torch.randn(shape) * base_std
    normal_outlier = torch.randn(shape) * outlier_std
    bernoulli = torch.bernoulli(torch.full(shape, outlier_prob))
    return normal_base + normal_outlier * bernoulli


def add_outliers(tensor: torch.Tensor, outlier_prob=0.001, outlier_scale=100) -> torch.Tensor:
    """
    Add random outliers to an existing tensor for precision testing.

    Modifies an input tensor by adding sparse random outliers with specified probability
    and magnitude. This is useful for testing how operations handle mixed-magnitude data
    and shared exponent precision behavior.

    Args:
        tensor: Input tensor to which outliers will be added
        outlier_prob: Probability of each element becoming an outlier (default: 0.001)
        outlier_scale: Scale factor for the outlier values (default: 100)

    Returns:
        torch.Tensor: Modified tensor with added outliers in bfloat16 format.
                     Original values are preserved where no outliers are added.

    Note:
        - Outliers are additive (original_value + outlier)
        - Uses bfloat16 precision to match typical TT-Metal operation requirements
        - Random outlier positions are determined by Bernoulli sampling
    """
    mask = torch.bernoulli(torch.full(tensor.shape, outlier_prob)).bfloat16()
    outliers = torch.randn(tensor.shape) * outlier_scale
    return tensor + mask * outliers.bfloat16()


# Main module functions
def generate_distributions(shape) -> dict:
    """
    Generate comprehensive set of tensor distributions for precision testing.

    Creates various statistical distributions designed to test different aspects
    of numerical precision, shared exponent behavior, and operation robustness.
    Each distribution targets specific precision characteristics.

    Args:
        shape: Tuple specifying tensor dimensions (height, width)

    Returns:
        dict: Dictionary mapping distribution names to torch.Tensor objects.
              Includes the following distribution categories:

              Basic Normal Distributions:
              - 'normal_0_1': Standard normal N(0,1)
              - 'normal_skewed_mean': Normal with shifted mean N(5,1)

              High Variance Distributions:
              - 'normal_high_var_10': High variance N(0,10²)
              - 'normal_high_var_100': Very high variance N(0,100²)

              Outlier Distributions:
              - 'normal_with_outliers': N(0,1) with sparse large outliers

              Combined Distributions:
              - 'skewed_high_var_10': N(10,10²) - shifted mean + high variance
              - 'skewed_high_var_100': N(10,100²) - shifted mean + very high variance

              Negative Versions:
              - '*_negative': Negated versions of all above distributions

              Mixed Gaussian (fa_rand variations):
              - 'fa_rand_default': Standard mixed Gaussian with 0.1% outliers
              - 'fa_rand_aggressive': Aggressive mixed Gaussian with 1% large outliers

    Note:
        All distributions are designed to stress-test shared exponent precision
        and identify potential accuracy issues in TT-Metal operations.
    """
    distributions = {}

    # Constant (all ones)
    distributions["constant"] = torch.ones(shape)

    # Normal 0-1
    distributions["normal_0_1"] = torch.randn(shape)

    # Normal with skewed mean
    distributions["normal_skewed_mean"] = torch.randn(shape) + 5.0

    # High variance (10 and 100)
    distributions["normal_high_var_10"] = torch.randn(shape) * 10
    distributions["normal_high_var_100"] = torch.randn(shape) * 100

    # Normal 0-1 with large outliers
    distributions["normal_with_outliers"] = add_outliers(torch.randn(shape))

    # Skewed mean + high variance combinations
    distributions["skewed_high_var_10"] = torch.randn(shape) * 10 + 10
    distributions["skewed_high_var_100"] = torch.randn(shape) * 100 + 10

    # Negative versions
    for key in list(distributions.keys()):
        distributions[f"{key}_negative"] = -distributions[key]

    # Mixture of Gaussians (fa_rand variations)
    distributions["fa_rand_default"] = fa_rand(shape)
    distributions["fa_rand_aggressive"] = fa_rand_custom(shape, 1.0, 100.0, 0.01)

    return distributions


def generate_test_patterns(shape) -> dict:
    """Generate various distribution patterns for testing

    Args:
        shape: tuple of (height, width) for the desired tensor shape

    Returns:
        Dictionary of pattern generation functions
    """

    # Pattern 1: Large magnitude differences across columns
    def column_magnitude_gradient(shape):
        tensor = torch.ones(shape)
        for col in range(shape[1]):
            tensor[:, col] *= 10 ** (col / shape[1] * 6 - 3)  # 10^-3 to 10^3
        return tensor

    # Pattern 2: Reverse column magnitude gradient
    def reverse_column_magnitude_gradient(shape):
        tensor = torch.ones(shape)
        for col in range(shape[1]):
            tensor[:, col] *= 10 ** -((col / shape[1]) * 6 - 3)  # 10^3 to 10^-3
        return tensor

    # Pattern 3: Row magnitude gradient
    def row_magnitude_gradient(shape):
        tensor = torch.ones(shape)
        for row in range(shape[0]):
            tensor[row, :] *= 10 ** ((row / shape[0]) * 6 - 3)  # 10^-3 to 10^3
        return tensor

    # Pattern 4: Reverse row magnitude gradient
    def reverse_row_magnitude_gradient(shape):
        tensor = torch.ones(shape)
        for row in range(shape[0]):
            tensor[row, :] *= 10 ** -((row / shape[0]) * 6 - 3)  # 10^3 to 10^-3
        return tensor

    # Pattern 5: Row-wise uniform, column-wise varying
    def row_uniform_column_varying(shape):
        base = torch.randn(shape)
        for row in range(shape[0]):
            base[row, :] *= 10 ** (row % 4 - 2)  # Cycle through magnitudes
        return base

    # Pattern 6: Checkerboard pattern of magnitudes
    def checkerboard_magnitudes(shape):
        tensor = torch.randn(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (i + j) % 2 == 0:
                    tensor[i, j] *= 100
        return tensor

    # Pattern 7: Extreme outliers in specific rows (adaptive to shape)
    def row_outliers(shape, outlier_fraction=0.1):
        tensor = torch.randn(shape)
        num_outlier_rows = max(1, int(shape[0] * outlier_fraction))
        outlier_rows = torch.randperm(shape[0])[:num_outlier_rows].tolist()
        for row in outlier_rows:
            tensor[row, :] *= 1000
        return tensor

    # Pattern 8: Tile-aware patterns (for multi-tile testing)
    def tile_boundaries(shape, tile_size=32):
        """Create magnitude differences at tile boundaries"""
        tensor = torch.randn(shape)
        for i in range(0, shape[0], tile_size):
            for j in range(0, shape[1], tile_size):
                # Each tile gets a different magnitude
                tile_magnitude = 10 ** ((i // tile_size + j // tile_size) % 4 - 2)
                tensor[i : i + tile_size, j : j + tile_size] *= tile_magnitude
        return tensor

    # Pattern 9: Diagonal gradient
    def diagonal_gradient(shape):
        tensor = torch.randn(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                # Distance from top-left corner
                dist = (i + j) / (shape[0] + shape[1])
                tensor[i, j] *= 10 ** (dist * 6 - 3)
        return tensor

    # Pattern 10: Block patterns with extreme differences
    def block_pattern(shape, block_size=8):
        tensor = torch.randn(shape)
        magnitudes = [1e-3, 1e-1, 1, 1e2, 1e4]
        for i in range(0, shape[0], block_size):
            for j in range(0, shape[1], block_size):
                mag_idx = ((i // block_size) * (shape[1] // block_size) + (j // block_size)) % len(magnitudes)
                tensor[i : i + block_size, j : j + block_size] *= magnitudes[mag_idx]
        return tensor

    # Pattern 11: Use fa_rand as a pattern
    def fa_rand_pattern(shape):
        """Use fa_rand directly as a pattern"""
        return fa_rand_custom(shape, base_std=1.0, outlier_std=100.0, outlier_prob=0.005)

    return {
        "column_gradient": lambda: column_magnitude_gradient(shape),
        "reverse_column_magnitude_gradient": lambda: reverse_column_magnitude_gradient(shape),
        "row_gradient": lambda: row_magnitude_gradient(shape),
        "reverse_row_magnitude_gradient": lambda: reverse_row_magnitude_gradient(shape),
        "row_uniform": lambda: row_uniform_column_varying(shape),
        "checkerboard": lambda: checkerboard_magnitudes(shape),
        "row_outliers": lambda: row_outliers(shape),
        "tile_boundaries": lambda: tile_boundaries(shape),
        "diagonal_gradient": lambda: diagonal_gradient(shape),
        "block_pattern": lambda: block_pattern(shape),
        "fa_rand_pattern": lambda: fa_rand_pattern(shape),
    }
