# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

from .format_config import DataFormat
from .llk_params import format_dict
from .tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    FACE_C_DIM,
    MAX_FACE_ELEMENTS,
    MAX_FACE_R_DIM,
    MAX_NUM_FACES,
    MAX_NUM_FACES_C_DIM,
    MAX_NUM_FACES_R_DIM,
    MAX_TILE_ELEMENTS,
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

    input_reshaped = input_tensor.view(dimensions[0], dimensions[1])
    if input_reshaped.ndim != 2:
        raise ValueError(
            f"Expected a 2D tensor for tilize_block, got shape {input_tensor.shape}"
        )

    rows, cols = input_reshaped.shape

    # Determine tile dimensions
    if tile_dimensions is not None:
        tile_rows, tile_cols = tile_dimensions
        # get_tile_params validates tile_dimensions internally
        face_r_dim, _, _ = get_tile_params(tile_dimensions)
    else:
        # Default to standard 32x32 tiles
        tile_rows, tile_cols = DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM

    if rows % tile_rows != 0 or cols % tile_cols != 0:
        raise ValueError(
            f"Input dimensions {dimensions} must be divisible by tile dimensions "
            f"[{tile_rows}, {tile_cols}]."
        )

    # Calculate number of tiles
    row_tiles = rows // tile_rows
    col_tiles = cols // tile_cols
    total_tiles = row_tiles * col_tiles

    # Calculate elements per tile and face
    elements_per_tile = tile_rows * tile_cols
    elements_per_face = face_r_dim * FACE_C_DIM

    # Reshape into tiles: (row_tiles, tile_rows, col_tiles, tile_cols)
    blocked_tensor = input_reshaped.reshape(row_tiles, tile_rows, col_tiles, tile_cols)

    # Permute to get tiles in row-major order: (row_tiles, col_tiles, tile_rows, tile_cols)
    blocked_tensor = blocked_tensor.permute(0, 2, 1, 3)

    # Reshape to get all tiles as sequential entities: (total_tiles, tile_rows, tile_cols)
    all_tiles = blocked_tensor.reshape(total_tiles, tile_rows, tile_cols)

    # Flatten each tile for tilization
    flat_tiles = all_tiles.reshape(total_tiles, -1)

    # Tilize each tile
    # For standard 32x32 tiles, use original tilize signature for backward compatibility
    if tile_rows == DEFAULT_TILE_R_DIM and tile_cols == DEFAULT_TILE_C_DIM:
        tilized_tiles = torch.stack(
            [
                tilize(tile, stimuli_format=stimuli_format, num_faces=num_faces)
                for tile in flat_tiles
            ]
        )
    else:
        # For smaller tiles, pass tile_dimensions to enable proper face slicing
        tilized_tiles = torch.stack(
            [
                tilize(
                    tile,
                    stimuli_format=stimuli_format,
                    num_faces=num_faces,
                    face_r_dim=face_r_dim,
                    tile_dimensions=[tile_rows, tile_cols],
                )
                for tile in flat_tiles
            ]
        )

    # Each tilized tile has num_faces * elements_per_face elements
    tilized_elements_per_tile = num_faces * elements_per_face
    expected_elements = total_tiles * tilized_elements_per_tile

    tilized_output = (
        tilized_tiles.flatten()[:expected_elements]
        .reshape(total_tiles, tilized_elements_per_tile)
        .to(format_dict[stimuli_format])
    )

    return tilized_output


def tilize(
    original_tensor,
    stimuli_format=DataFormat.Float16_b,
    num_faces=MAX_NUM_FACES,
    face_r_dim=MAX_FACE_R_DIM,
    tile_dimensions=None,
):
    """Tilize a tensor into face-based layout.

    Args:
        original_tensor: Input tensor to tilize
        stimuli_format: Data format for output
        num_faces: Number of faces (1, 2, or 4)
        face_r_dim: Number of rows per face (1, 2, 4, 8, or 16)
        tile_dimensions: Optional [tile_rows, tile_cols] to auto-calculate parameters
                         Only used for non-32x32 tiles when num_faces and face_r_dim
                         are at their default values.

    For a tile with dimensions [tile_rows, tile_cols]:
        - face_r_dim = min(tile_rows, MAX_FACE_R_DIM) for single row of faces
        - num_faces_r_dim = ceil(tile_rows / MAX_FACE_R_DIM) (1 or 2)
        - num_faces_c_dim = tile_cols // FACE_C_DIM (typically 2 for 32-col tiles)
        - num_faces = num_faces_r_dim * num_faces_c_dim

    Examples:
        - 32x32 tile: 4 faces of 16x16 each (standard)
        - 32x16 tile: 2 faces of 16x16 each (vertical: f0, f2)
        - 16x32 tile: 2 faces of 16x16 each (horizontal: f0, f1)
        - 8x32 tile: 2 faces of 8x16 each (horizontal: f0, f1)
        - 4x32 tile: 2 faces of 4x16 each (horizontal: f0, f1)
    """
    # Auto-calculate from tile_dimensions for non-32x32 tiles
    # This preserves backward compatibility for 32x32 tiles
    num_faces_r_dim = 1
    num_faces_c_dim = 1
    default_tile_dims = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]
    if tile_dimensions is not None and tile_dimensions != default_tile_dims:
        face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
        num_faces = num_faces_r_dim * num_faces_c_dim

    elements_per_face = face_r_dim * FACE_C_DIM

    # For standard 32x32 tiles (backward compatibility)
    if face_r_dim == MAX_FACE_R_DIM and num_faces == MAX_NUM_FACES:
        if original_tensor.size(0) != MAX_TILE_ELEMENTS:
            raise ValueError(
                f"Input tensor must have {MAX_TILE_ELEMENTS} elements for "
                f"{DEFAULT_TILE_R_DIM}x{DEFAULT_TILE_C_DIM} tiles."
            )
        matrix = original_tensor.view(DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM)
        face_slices = [
            matrix[:MAX_FACE_R_DIM, :FACE_C_DIM],  # f0
            matrix[:MAX_FACE_R_DIM, FACE_C_DIM:],  # f1
            matrix[MAX_FACE_R_DIM:, :FACE_C_DIM],  # f2
            matrix[MAX_FACE_R_DIM:, FACE_C_DIM:],  # f3
        ]
        selected_faces = [face.flatten() for face in face_slices[:num_faces]]
        result = (
            torch.cat(selected_faces) if len(selected_faces) > 1 else selected_faces[0]
        )
        return result.to(dtype=format_dict[stimuli_format])

    # For tiles with 2 faces - need to distinguish layout direction
    if num_faces == 2:
        # Determine tile dimensions based on face arrangement
        if num_faces_r_dim == MAX_NUM_FACES_R_DIM and num_faces_c_dim == 1:
            # Vertical arrangement: 32x16 tile (2 rows of faces, 1 column)
            # Faces stacked vertically: f0 (top), f2 (bottom)
            tile_rows = DEFAULT_TILE_R_DIM
            tile_cols = FACE_C_DIM
            expected_elements = tile_rows * tile_cols

            if original_tensor.size(0) != expected_elements:
                raise ValueError(
                    f"Input tensor must have {expected_elements} elements for "
                    f"{tile_rows}x{tile_cols} tiles. Got {original_tensor.size(0)}."
                )

            matrix = original_tensor.view(tile_rows, tile_cols)
            face_slices = [
                matrix[:MAX_FACE_R_DIM, :FACE_C_DIM],  # f0: top half (rows 0-15)
                matrix[MAX_FACE_R_DIM:, :FACE_C_DIM],  # f2: bottom half (rows 16-31)
            ]
            selected_faces = [face.flatten() for face in face_slices]
            result = torch.cat(selected_faces)
            return result.to(dtype=format_dict[stimuli_format])

        elif num_faces_r_dim == 1 and num_faces_c_dim == MAX_NUM_FACES_C_DIM:
            # Horizontal arrangement: Nx32 tiles (1 row of faces, 2 columns)
            # Faces arranged horizontally: f0 (left), f1 (right)
            tile_rows = face_r_dim
            tile_cols = DEFAULT_TILE_C_DIM
            expected_elements = tile_rows * tile_cols

            if original_tensor.size(0) != expected_elements:
                raise ValueError(
                    f"Input tensor must have {expected_elements} elements for "
                    f"{tile_rows}x{tile_cols} tiles. Got {original_tensor.size(0)}."
                )

            matrix = original_tensor.view(tile_rows, tile_cols)
            face_slices = [
                matrix[:face_r_dim, :FACE_C_DIM],  # f0: left half
                matrix[:face_r_dim, FACE_C_DIM:],  # f1: right half
            ]
            selected_faces = [face.flatten() for face in face_slices]
            result = torch.cat(selected_faces)
            return result.to(dtype=format_dict[stimuli_format])

        else:
            # Fallback for legacy num_faces=2 without tile_dimensions
            # Assumes horizontal arrangement (backward compatibility)
            tile_rows = face_r_dim
            tile_cols = DEFAULT_TILE_C_DIM
            expected_elements = tile_rows * tile_cols

        if original_tensor.size(0) != expected_elements:
            raise ValueError(
                f"Input tensor must have {expected_elements} elements for "
                f"{tile_rows}x{tile_cols} tiles. Got {original_tensor.size(0)}."
            )

        matrix = original_tensor.view(tile_rows, tile_cols)
        face_slices = [
            matrix[:face_r_dim, :FACE_C_DIM],  # f0: left half
            matrix[:face_r_dim, FACE_C_DIM:],  # f1: right half
        ]
        selected_faces = [face.flatten() for face in face_slices]
        result = torch.cat(selected_faces)
        return result.to(dtype=format_dict[stimuli_format])

    # For single face tiles (e.g., 16x16)
    if num_faces == 1:
        expected_elements = elements_per_face
        if original_tensor.size(0) != expected_elements:
            raise ValueError(
                f"Input tensor must have {expected_elements} elements for "
                f"single face tiles. Got {original_tensor.size(0)}."
            )
        return original_tensor.to(dtype=format_dict[stimuli_format])

    raise ValueError(
        f"Unsupported combination: num_faces={num_faces}, face_r_dim={face_r_dim}"
    )


def untilize(
    tilized_tensor,
    stimuli_format=DataFormat.Float16_b,
    tile_dimensions=None,
    face_r_dim=MAX_FACE_R_DIM,
):
    """Untilize a tensor from face-based layout back to row-major.

    Args:
        tilized_tensor: Input tensor in tilized (face-based) layout
        stimuli_format: Data format for output
        tile_dimensions: Optional [tile_rows, tile_cols] for tile size
        face_r_dim: Number of rows per face (1, 2, 4, 8, or 16)

    Returns:
        Untilized tensor in row-major layout
    """
    # Default to standard 32x32 tiles for backward compatibility
    if tile_dimensions is None:
        tile_dimensions = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]

    tile_rows, tile_cols = tile_dimensions
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim
    elements_per_face = face_r_dim * FACE_C_DIM
    expected_elements = num_faces * elements_per_face

    if tilized_tensor.size(0) != expected_elements:
        raise ValueError(
            f"Input tensor must have {expected_elements} elements for "
            f"{tile_rows}x{tile_cols} tiles. It has: {tilized_tensor.size(0)}"
        )

    tilized_tensor = tilized_tensor.view(-1)

    # Standard 32x32 tiles (4 faces)
    if num_faces == MAX_NUM_FACES and face_r_dim == MAX_FACE_R_DIM:
        f0 = tilized_tensor[:MAX_FACE_ELEMENTS].view(MAX_FACE_R_DIM, FACE_C_DIM)
        f1 = tilized_tensor[MAX_FACE_ELEMENTS : 2 * MAX_FACE_ELEMENTS].view(
            MAX_FACE_R_DIM, FACE_C_DIM
        )
        f2 = tilized_tensor[2 * MAX_FACE_ELEMENTS : 3 * MAX_FACE_ELEMENTS].view(
            MAX_FACE_R_DIM, FACE_C_DIM
        )
        f3 = tilized_tensor[3 * MAX_FACE_ELEMENTS :].view(MAX_FACE_R_DIM, FACE_C_DIM)

        top = torch.cat((f0, f1), dim=1)
        bottom = torch.cat((f2, f3), dim=1)
        original_tensor = torch.cat((top, bottom), dim=0).view(MAX_TILE_ELEMENTS)
        return original_tensor.to(dtype=format_dict[stimuli_format])

    # 2-face tiles: vertical arrangement (32x16)
    if num_faces_r_dim == MAX_NUM_FACES_R_DIM and num_faces_c_dim == 1:
        f0 = tilized_tensor[:elements_per_face].view(face_r_dim, FACE_C_DIM)
        f2 = tilized_tensor[elements_per_face:].view(face_r_dim, FACE_C_DIM)

        original_tensor = torch.cat((f0, f2), dim=0).view(tile_rows * tile_cols)
        return original_tensor.to(dtype=format_dict[stimuli_format])

    # 2-face tiles: horizontal arrangement (Nx32)
    if num_faces_r_dim == 1 and num_faces_c_dim == MAX_NUM_FACES_C_DIM:
        f0 = tilized_tensor[:elements_per_face].view(face_r_dim, FACE_C_DIM)
        f1 = tilized_tensor[elements_per_face:].view(face_r_dim, FACE_C_DIM)

        original_tensor = torch.cat((f0, f1), dim=1).view(tile_rows * tile_cols)
        return original_tensor.to(dtype=format_dict[stimuli_format])

    # Single face tiles (e.g., 16x16)
    if num_faces == 1:
        return tilized_tensor.to(dtype=format_dict[stimuli_format])

    raise ValueError(
        f"Unsupported tile configuration: {tile_rows}x{tile_cols}, "
        f"num_faces={num_faces}, face_r_dim={face_r_dim}"
    )


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

    if rows % tile_rows != 0 or cols % tile_cols != 0:
        raise ValueError(
            f"Dimensions {dimensions} must be divisible by tile dimensions "
            f"[{tile_rows}, {tile_cols}]."
        )

    # Calculate number of tiles
    row_tiles = rows // tile_rows
    col_tiles = cols // tile_cols
    total_tiles = row_tiles * col_tiles

    # Calculate face parameters using get_tile_params (also validates tile_dimensions)
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim
    elements_per_face = face_r_dim * FACE_C_DIM
    tilized_elements_per_tile = num_faces * elements_per_face

    expected_elements = total_tiles * tilized_elements_per_tile
    if input_tensor.numel() != expected_elements:
        raise ValueError(
            f"Cannot untilize tensor of size {input_tensor.numel()}. "
            f"Expected {expected_elements} elements for {total_tiles} tiles of "
            f"{tile_rows}x{tile_cols}."
        )

    # Reshape input to have one tile per row
    input_reshaped = input_tensor.reshape(total_tiles, tilized_elements_per_tile)

    # Untilize each tile
    untilized_tiles = torch.stack(
        [
            untilize(
                tile,
                stimuli_format=stimuli_format,
                tile_dimensions=tile_dimensions,
                face_r_dim=face_r_dim,
            )
            for tile in input_reshaped
        ]
    )

    # Reshape to (row_tiles, col_tiles, tile_rows, tile_cols)
    output = untilized_tiles.reshape(row_tiles, col_tiles, tile_rows, tile_cols)

    # Permute and reshape to get the final dimensions
    output = output.permute(0, 2, 1, 3).reshape(rows, cols)

    return output.to(dtype=format_dict[stimuli_format])
