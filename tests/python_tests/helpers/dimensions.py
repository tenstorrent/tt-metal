# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for dimension-related calculations in matrix operations.
"""

from typing import List, Tuple


def validate_tile_dimensions(dimension: int, row_col_dim: int):
    """Validate that dimension is divisible by row/col."""
    if dimension % row_col_dim != 0:
        raise AssertionError(f"{dimension} must be divisible by {row_col_dim}")


def calculate_matmul_dimensions(
    input_A_dimensions: Tuple[int, int], input_B_dimensions: Tuple[int, int]
) -> dict:
    """
    Calculate matrix multiplication tile dimensions.

    For matrix multiplication A[M,K] × B[K,N] = C[M,N], calculate:
    - rt_dim: Row tiles (M dimension)
    - ct_dim: Column tiles (N dimension)
    - kt_dim: Inner dimension tiles (K dimension)
    - output_dimensions: Result matrix dimensions
    - output_tile_cnt: Number of tiles in result

    Args:
        input_A_dimensions: (rows, cols) for matrix A
        input_B_dimensions: (rows, cols) for matrix B

    Returns:
        dict: Dictionary containing all calculated dimensions

    Raises:
        AssertionError: If matrix dimensions are incompatible for multiplication
    """
    M, K1 = input_A_dimensions[0], input_A_dimensions[1]
    K2, N = input_B_dimensions[0], input_B_dimensions[1]

    # Verify K dimensions match for valid matmul
    assert (
        K1 == K2
    ), f"Matrix dimensions incompatible for multiplication: A[{M},{K1}] × B[{K2},{N}]"

    # Calculate output dimensions: A[M,K] × B[K,N] = C[M,N]
    output_dimensions = (M, N)

    # Calculate tile dimensions (each tile is 32×32)
    num_rows = 32  # matrix A
    num_cols = 32  # matrix B

    validate_tile_dimensions(M, num_cols)
    validate_tile_dimensions(N, num_rows)
    validate_tile_dimensions(K1, num_cols)

    rt_dim = M // num_cols  # Row tiles in result
    ct_dim = N // num_rows  # Column tiles in result
    kt_dim = (
        K1 // num_cols
    )  # Inner dimension tiles rt_dim (matrix A) = kt_dim = ct_dim (matrix B) = 1

    # Calculate tile counts
    output_tile_cnt = rt_dim * ct_dim

    return {
        "rt_dim": rt_dim,
        "ct_dim": ct_dim,
        "kt_dim": kt_dim,
        "output_dimensions": output_dimensions,
        "output_tile_cnt": output_tile_cnt,
    }


def generate_matmul_dimension_combinations(max_tiles: int) -> List[tuple]:
    """
    Generate all valid matrix multiplication dimension combinations.

    Creates all possible combinations of (inputA_dimensions, inputB_dimensions) where:
    - The result matrix also has at most max_tiles tiles
    - Matrix multiplication is valid: inputA[1] == inputB[0] (K dimensions match)
    - Returns combinations that can be used for comprehensive matmul testing

    Args:
        max_tiles: Maximum number of tiles allowed per result matrix

    Returns:
        List of tuples: Each tuple contains (inputA_dimensions, inputB_dimensions)
        where inputA_dimensions and inputB_dimensions are [rows, cols] lists

    Note: When 16-bit datums in dest can fit max 8 tiles and 4 tiles for 32-bit datums
    Example:
        For max_tiles=4:
        Returns combinations like:
        ([32, 32], [32, 32])    # 1×1 tiles each, result: 1×1 = 1 tile
        ([32, 64], [64, 32])    # 1×2 and 2×1 tiles, result: 1×1 = 1 tile
        ([64, 128], [128, 128])    # result: 2×4 = 8 tiles, works for 16-bit datums
        ([32, 32], [32, 128])  # 1×1 and 1×4 tiles, result: 1×4 = 4 tiles, works for 16-bit and 32-bit datums

        But NOT ([256, 32], [32, 256]) because result would be 8×8 = 64 tiles > 4 for 32-bit datums and >8 for 16-bit datums
    """


def generate_matmul_dimension_combinations(max_tiles: int) -> List[tuple]:
    valid_combinations = []
    tile_rows = 32
    tile_cols = 32

    for m_tiles in range(1, max_tiles + 1):
        for k_tiles in range(1, max_tiles + 1):
            # Check if matrix A is valid: m_tiles * k_tiles <= max_tiles
            if m_tiles * k_tiles > max_tiles:
                break  # Early termination - larger k_tiles will also be invalid

            # Calculate maximum valid n_tiles based on constraints
            max_n_from_B = (
                max_tiles // k_tiles
            )  # From B constraint: k_tiles * n_tiles <= max_tiles
            max_n_from_C = (
                max_tiles // m_tiles
            )  # From C constraint: m_tiles * n_tiles <= max_tiles
            max_n_tiles = min(max_n_from_B, max_n_from_C)

            # Generate all valid n_tiles values
            for n_tiles in range(1, max_n_tiles + 1):
                # Convert tile counts to actual dimensions
                m_dim = m_tiles * tile_cols
                k_dim = k_tiles * tile_cols
                n_dim = n_tiles * tile_rows

                inputA_dims = [m_dim, k_dim]
                inputB_dims = [k_dim, n_dim]
                valid_combinations.append((inputA_dims, inputB_dims))

    return valid_combinations
