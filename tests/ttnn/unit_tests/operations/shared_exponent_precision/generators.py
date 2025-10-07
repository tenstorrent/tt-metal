# SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch


# helper functions
def fa_rand(shape) -> torch.Tensor:
    """Original fa_rand function"""
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def fa_rand_custom(shape, base_std=1.0, outlier_std=10.0, outlier_prob=0.001) -> torch.Tensor:
    """Modified fa_rand with controllable parameters"""
    normal_base = torch.randn(shape) * base_std
    normal_outlier = torch.randn(shape) * outlier_std
    bernoulli = torch.bernoulli(torch.full(shape, outlier_prob))
    return normal_base + normal_outlier * bernoulli


def add_outliers(tensor: torch.Tensor, outlier_prob=0.001, outlier_scale=100) -> torch.Tensor:
    """Add outliers to existing tensor"""
    mask = torch.bernoulli(torch.full(tensor.shape, outlier_prob)).bfloat16()
    outliers = torch.randn(tensor.shape) * outlier_scale
    return tensor + mask * outliers.bfloat16()


# Main module functions
def generate_distributions(shape) -> dict:
    """Generate different distribution types"""
    distributions = {}

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

    # Pattern 2: Row-wise uniform, column-wise varying
    def row_uniform_column_varying(shape):
        base = torch.randn(shape)
        for row in range(shape[0]):
            base[row, :] *= 10 ** (row % 4 - 2)  # Cycle through magnitudes
        return base

    # Pattern 3: Checkerboard pattern of magnitudes
    def checkerboard_magnitudes(shape):
        tensor = torch.randn(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (i + j) % 2 == 0:
                    tensor[i, j] *= 100
        return tensor

    # Pattern 4: Extreme outliers in specific rows (adaptive to shape)
    def row_outliers(shape, outlier_fraction=0.1):
        tensor = torch.randn(shape)
        num_outlier_rows = max(1, int(shape[0] * outlier_fraction))
        outlier_rows = torch.randperm(shape[0])[:num_outlier_rows].tolist()
        for row in outlier_rows:
            tensor[row, :] *= 1000
        return tensor

    # Pattern 5: Tile-aware patterns (for multi-tile testing)
    def tile_boundaries(shape, tile_size=32):
        """Create magnitude differences at tile boundaries"""
        tensor = torch.randn(shape)
        for i in range(0, shape[0], tile_size):
            for j in range(0, shape[1], tile_size):
                # Each tile gets a different magnitude
                tile_magnitude = 10 ** ((i // tile_size + j // tile_size) % 4 - 2)
                tensor[i : i + tile_size, j : j + tile_size] *= tile_magnitude
        return tensor

    # Pattern 6: Diagonal gradient
    def diagonal_gradient(shape):
        tensor = torch.randn(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                # Distance from top-left corner
                dist = (i + j) / (shape[0] + shape[1])
                tensor[i, j] *= 10 ** (dist * 6 - 3)
        return tensor

    # Pattern 7: Block patterns with extreme differences
    def block_pattern(shape, block_size=8):
        tensor = torch.randn(shape)
        magnitudes = [1e-3, 1e-1, 1, 1e2, 1e4]
        for i in range(0, shape[0], block_size):
            for j in range(0, shape[1], block_size):
                mag_idx = ((i // block_size) * (shape[1] // block_size) + (j // block_size)) % len(magnitudes)
                tensor[i : i + block_size, j : j + block_size] *= magnitudes[mag_idx]
        return tensor

    # Pattern 8: Use fa_rand as a pattern
    def fa_rand_pattern(shape):
        """Use fa_rand directly as a pattern"""
        return fa_rand_custom(shape, base_std=1.0, outlier_std=100.0, outlier_prob=0.005)

    return {
        "column_gradient": lambda: column_magnitude_gradient(shape),
        "row_uniform": lambda: row_uniform_column_varying(shape),
        "checkerboard": lambda: checkerboard_magnitudes(shape),
        "row_outliers": lambda: row_outliers(shape),
        "tile_boundaries": lambda: tile_boundaries(shape),
        "diagonal_gradient": lambda: diagonal_gradient(shape),
        "block_pattern": lambda: block_pattern(shape),
        "fa_rand_pattern": lambda: fa_rand_pattern(shape),
    }
