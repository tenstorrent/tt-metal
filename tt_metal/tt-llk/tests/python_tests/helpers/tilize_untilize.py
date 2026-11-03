# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

from .format_config import DataFormat
from .llk_params import format_dict
from .tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    FACE_C_DIM,
    MAX_FACE_R_DIM,
    MAX_NUM_FACES,
    MAX_NUM_FACES_C_DIM,
    MAX_NUM_FACES_R_DIM,
    get_tile_params,
)


def tilize_block(
    input_tensor,
    dimensions,
    stimuli_format=DataFormat.Float16_b,
    num_faces=MAX_NUM_FACES,
    tile_dimensions=None,
    face_r_dim=MAX_FACE_R_DIM,
):
    """Tilize a block of data into face-based tile layout.

    Args:
        input_tensor: Input tensor to tilize
        dimensions: [rows, cols] of the input tensor
        stimuli_format: Data format for output
        num_faces: Number of faces per tile (1, 2, or 4)
        tile_dimensions: Optional [tile_rows, tile_cols] for tile size
        face_r_dim: Number of rows per face (1, 2, 4, 8, or 16)

    Returns:
        Tilized tensor with tiles laid out sequentially
    """
    if input_tensor.numel() != dimensions[0] * dimensions[1]:
        raise ValueError(
            f"Cannot reshape tensor of size {input_tensor.numel()} to shape {dimensions}."
        )

    rows, cols = dimensions
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Dimensions {dimensions} must be positive.")

    if tile_dimensions is None:
        if num_faces != MAX_NUM_FACES or not 0 < face_r_dim <= MAX_FACE_R_DIM:
            raise ValueError(
                f"Unsupported combination: num_faces={num_faces}, face_r_dim={face_r_dim}"
            )
        output_elements_per_tile = num_faces * face_r_dim * FACE_C_DIM
        tile_dimensions = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]
    else:
        output_elements_per_tile = tile_dimensions[0] * tile_dimensions[1]

    tile_rows, tile_cols = tile_dimensions
    if tile_rows == DEFAULT_TILE_R_DIM and tile_cols == DEFAULT_TILE_C_DIM:
        face_r_dim = MAX_FACE_R_DIM
        num_faces_r_dim = MAX_NUM_FACES_R_DIM
        num_faces_c_dim = MAX_NUM_FACES_C_DIM
    else:
        face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)

    if rows % tile_rows != 0 or cols % tile_cols != 0:
        raise ValueError(
            f"Input dimensions {dimensions} must be divisible by tile dimensions "
            f"[{tile_rows}, {tile_cols}]."
        )

    # Calculate number of tiles
    row_tiles = rows // tile_rows
    col_tiles = cols // tile_cols
    total_tiles = row_tiles * col_tiles

    elements_per_tile = tile_rows * tile_cols
    tilized_output = (
        input_tensor.reshape(
            row_tiles,
            num_faces_r_dim,
            face_r_dim,
            col_tiles,
            num_faces_c_dim,
            FACE_C_DIM,
        )
        .permute(0, 3, 1, 4, 2, 5)
        .to(
            dtype=format_dict[stimuli_format],
            memory_format=torch.contiguous_format,
            copy=True,
        )
        .view(total_tiles, elements_per_tile)
    )
    if output_elements_per_tile != elements_per_tile:
        return tilized_output.view(-1)[: total_tiles * output_elements_per_tile].view(
            total_tiles, output_elements_per_tile
        )
    return tilized_output


def untilize_block(
    input_tensor,
    stimuli_format=DataFormat.Float16_b,
    dimensions=None,
    tile_dimensions=None,
    num_faces=MAX_NUM_FACES,
    face_r_dim=MAX_FACE_R_DIM,
):
    """Optimized function to untilize blocks of data.

    Args:
        input_tensor: Input tensor to be untilized
        stimuli_format: Data format
        dimensions: Target dimensions for the output [rows, cols]
        tile_dimensions: Optional [tile_rows, tile_cols] for tile size
        num_faces: Number of faces per tile (1, 2, or 4)
        face_r_dim: Number of rows per face (1, 2, 4, 8, or 16)

    Returns:
        Untilized tensor with specified dimensions and data format
    """
    # Default dimensions to standard 32x32
    if dimensions is None:
        dimensions = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]

    # Default to standard 32x32 tiles
    if tile_dimensions is None:
        tile_dimensions = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]

    tile_rows, tile_cols = tile_dimensions
    rows, cols = dimensions

    if tile_rows == DEFAULT_TILE_R_DIM and tile_cols == DEFAULT_TILE_C_DIM:
        face_r_dim = MAX_FACE_R_DIM
        num_faces_r_dim = MAX_NUM_FACES_R_DIM
        num_faces_c_dim = MAX_NUM_FACES_C_DIM
    else:
        face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)

    if rows <= 0 or cols <= 0:
        raise ValueError(f"Dimensions {dimensions} must be positive.")

    if rows % tile_rows != 0 or cols % tile_cols != 0:
        raise ValueError(
            f"Dimensions {dimensions} must be divisible by tile dimensions "
            f"[{tile_rows}, {tile_cols}]."
        )

    # Calculate number of tiles
    row_tiles = rows // tile_rows
    col_tiles = cols // tile_cols

    expected_elements = rows * cols
    if input_tensor.numel() != expected_elements:
        raise ValueError(
            f"Cannot untilize tensor of size {input_tensor.numel()}. "
            f"Expected {expected_elements} elements for {row_tiles * col_tiles} tiles of "
            f"{tile_rows}x{tile_cols}."
        )

    return (
        input_tensor.reshape(
            row_tiles,
            col_tiles,
            num_faces_r_dim,
            num_faces_c_dim,
            face_r_dim,
            FACE_C_DIM,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .to(
            dtype=format_dict[stimuli_format],
            memory_format=torch.contiguous_format,
            copy=True,
        )
        .view(rows, cols)
    )
