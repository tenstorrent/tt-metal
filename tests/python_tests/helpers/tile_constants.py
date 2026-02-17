# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tile dimension constants for LLK testing.

This module defines the hardware tile layout constants used across the test
infrastructure. A tile is composed of faces arranged in a grid:

    - Total tile rows = face_r_dim * num_faces_r_dim
    - Total tile cols = face_c_dim * num_faces_c_dim

Example layouts:
    - 16x16 tile: 1 face of 16x16  (num_faces_r=1, num_faces_c=1)
    - 32x32 tile: 4 faces of 16x16 (num_faces_r=2, num_faces_c=2)
    - 32x16 tile: 2 faces of 16x16 (num_faces_r=2, num_faces_c=1)
    - 8x32 tile:  2 faces of 8x16  (num_faces_r=1, num_faces_c=2)
"""

# =============================================================================
# Maximum/Default Face Dimensions
# =============================================================================

# Maximum row dimension of a single face (hardware constraint)
MAX_FACE_R_DIM = 16

# Column dimension of a single face (always 16 for hardware)
FACE_C_DIM = 16

# =============================================================================
# BFP Format Constants
# =============================================================================

# Minimum exponents required for BFP formats (hardware constraint)
# The exponent section must be padded to at least 16 bytes for both unpacker and packer
MIN_BFP_EXPONENTS = 16

# =============================================================================
# Maximum/Default Tile Dimensions
# =============================================================================

# Default tile dimensions (standard 32x32 tile)
DEFAULT_TILE_R_DIM = 32
DEFAULT_TILE_C_DIM = 32

# Maximum number of faces in row dimension
MAX_NUM_FACES_R_DIM = 2

# Maximum number of faces in column dimension
MAX_NUM_FACES_C_DIM = 2

# Maximum total number of faces per tile (2x2 grid = 4 faces)
MAX_NUM_FACES = MAX_NUM_FACES_R_DIM * MAX_NUM_FACES_C_DIM

# =============================================================================
# Derived Constants
# =============================================================================

# Maximum elements per face (16 rows × 16 cols)
MAX_FACE_ELEMENTS = MAX_FACE_R_DIM * FACE_C_DIM

# Maximum elements per tile (32 rows × 32 cols = 4 faces × 256 elements)
MAX_TILE_ELEMENTS = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM

# =============================================================================
# Supported Tile Sizes
# =============================================================================

# All supported tile dimensions as (rows, cols) tuples
SUPPORTED_TILE_SIZES = [
    (16, 16),
    (1, 32),
    (2, 32),
    (4, 32),
    (8, 32),
    (16, 32),
    (32, 32),
    (32, 16),
]


def validate_tile_dimensions(tile_dimensions):
    """
    Validate that tile dimensions are supported.

    Args:
        tile_dimensions: List or tuple of [rows, cols]

    Raises:
        ValueError: If tile dimensions are not in SUPPORTED_TILE_SIZES
    """
    tile_tuple = tuple(tile_dimensions)
    if tile_tuple not in SUPPORTED_TILE_SIZES:
        raise ValueError(
            f"Unsupported tile dimensions: {tile_dimensions}. "
            f"Supported sizes are: {SUPPORTED_TILE_SIZES}"
        )


def get_tile_params(tile_dimensions):
    """
    Calculate TensorShape parameters from tile dimensions.

    Supported tile dimensions:
    - [16, 16] -> face_r_dim=16, num_faces=1, num_faces_r_dim=1, num_faces_c_dim=1
    - [1, 32]  -> face_r_dim=1,  num_faces=2, num_faces_r_dim=1, num_faces_c_dim=2
    - [2, 32]  -> face_r_dim=2,  num_faces=2, num_faces_r_dim=1, num_faces_c_dim=2
    - [4, 32]  -> face_r_dim=4,  num_faces=2, num_faces_r_dim=1, num_faces_c_dim=2
    - [8, 32]  -> face_r_dim=8,  num_faces=2, num_faces_r_dim=1, num_faces_c_dim=2
    - [16, 32] -> face_r_dim=16, num_faces=2, num_faces_r_dim=1, num_faces_c_dim=2
    - [32, 32] -> face_r_dim=16, num_faces=4, num_faces_r_dim=2, num_faces_c_dim=2
    - [32, 16] -> face_r_dim=16, num_faces=2, num_faces_r_dim=2, num_faces_c_dim=1

    Args:
        tile_dimensions: List or tuple of [rows, cols]

    Returns:
        tuple: (face_r_dim, num_faces_r_dim, num_faces_c_dim)

    Raises:
        ValueError: If tile dimensions are not supported
    """
    validate_tile_dimensions(tile_dimensions)

    tile_rows, tile_cols = tile_dimensions

    # face_r_dim is the number of rows per face, capped at 16
    face_r_dim = min(tile_rows, 16)
    # face_c_dim is always 16 for hardware
    face_c_dim = 16

    # num_faces_r_dim: number of faces vertically (row dimension)
    num_faces_r_dim = (tile_rows + 15) // 16

    # num_faces_c_dim: number of faces horizontally (column dimension)
    num_faces_c_dim = tile_cols // 16

    return face_r_dim, num_faces_r_dim, num_faces_c_dim


def calculate_tile_size_bytes(data_format, tile_dimensions, format_tile_sizes):
    """
    Calculate the actual tile size in bytes based on tile dimensions and data format.

    For standard 32x32 tiles, uses the predefined format_tile_sizes which includes
    format-specific overhead (e.g., exponent bytes for Bfp8_b).

    For BFP formats, hardware requires minimum 16 exponents total for both
    unpacker (input) and packer (output).

    Args:
        data_format: DataFormat enum value
        tile_dimensions: List or tuple of [rows, cols]
        format_tile_sizes: Dict mapping DataFormat to full tile size in bytes

    Returns:
        int: Tile size in bytes
    """
    from .format_config import DataFormat

    tile_rows, tile_cols = tile_dimensions
    tile_elements = tile_rows * tile_cols

    # For standard 32x32 tiles, use the predefined sizes (includes format overhead)
    if tile_elements == MAX_TILE_ELEMENTS:
        return format_tile_sizes[data_format]

    # For BFP8_b, hardware requires minimum exponents for both unpacker and packer
    if data_format in (DataFormat.Bfp8, DataFormat.Bfp8_b):
        actual_exponents = tile_elements // 16
        total_exponents = max(actual_exponents, MIN_BFP_EXPONENTS)
        # mantissas = 1 byte per element
        return total_exponents + tile_elements

    # Use data_format.num_bytes_per_tile() for other formats
    # - MxFp8: 1 scale per 32 elements (at beginning of tile)
    # - Other formats: just element size * count
    return data_format.num_bytes_per_tile(tile_elements)
