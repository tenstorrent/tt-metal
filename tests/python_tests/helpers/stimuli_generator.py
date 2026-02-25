# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

from .format_config import (
    MXFP8_E4M3_MAX_NORMAL,
    MXFP8_E5M2_MAX_NORMAL,
    DataFormat,
)
from .llk_params import format_dict
from .tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    FACE_C_DIM,
    MAX_FACE_R_DIM,
    MAX_NUM_FACES,
    get_tile_params,
    validate_tile_dimensions,
)


def flatten_list(sublists):
    return [item for sublist in sublists for item in sublist]


def _mask_tile(
    tile: torch.Tensor, num_faces: int, is_matrix_B: bool, face_r_dim: int = 16
) -> torch.Tensor:
    masked = tile.clone()
    if num_faces == 1:
        # Keep only f0
        masked[:16, 16:] = 0  # Zero f1
        masked[face_r_dim:, :] = 0  # Zero f2, f3 and part of f0
    elif num_faces == 2:
        if is_matrix_B:
            # matrix B (In1/SrcA): keep partial f0, f2
            if face_r_dim < 16:
                masked[face_r_dim:16, :16] = 0  # Zero part of f0
                masked[16 + face_r_dim :, :16] = 0  # Zero part of f2
            masked[:16, 16:] = 0  # Zero f1
            masked[16:, 16:] = 0  # Zero f3
        else:
            # matrix A (In0/SrcB): keep f0, f1
            masked[face_r_dim:, :] = 0  # Zero part of f0 and f1
    return masked


def generate_random_face(
    stimuli_format=DataFormat.Float16_b,
    const_value=1,
    const_face=False,
    sfpu=True,
    face_r_dim=MAX_FACE_R_DIM,
    negative_values=False,
):
    size = face_r_dim * FACE_C_DIM  # face_r_dim rows × FACE_C_DIM columns

    if stimuli_format in [DataFormat.MxFp8R, DataFormat.MxFp8P]:
        # MXFP8 optimized stimuli generation
        return _generate_mxfp8_face(stimuli_format, size, const_face, const_value, sfpu)
    elif stimuli_format != DataFormat.Bfp8_b:
        if stimuli_format.is_integer():
            max_value = 127 if stimuli_format == DataFormat.Int8 else 255
            min_value = -(max_value + 1) if negative_values else 0
            srcA_face = torch.randint(
                low=min_value,
                high=max_value,
                size=(size,),
                dtype=format_dict[stimuli_format],
            )
        else:
            if const_face:
                srcA_face = (
                    torch.ones(size, dtype=format_dict[stimuli_format]) * const_value
                )
            else:
                # random for both faces
                srcA_face = torch.rand(size, dtype=format_dict[stimuli_format])
                if negative_values:
                    srcA_face = srcA_face * 2 - 1  # Scaling for negative values.
                if sfpu:
                    srcA_face += 0.1
    else:
        if const_face:
            srcA_face = torch.ones(size, dtype=torch.bfloat16) * const_value
        else:
            low = -1 if negative_values else 0
            integer_part = torch.randint(low, 3, (size,))
            fraction = torch.randint(0, 16, (size,)).to(dtype=torch.bfloat16) / 16.0
            srcA_face = integer_part.to(dtype=torch.bfloat16) + fraction

    return srcA_face


def _generate_mxfp8_face(stimuli_format, size, const_face, const_value, sfpu):
    """
    Generate test data for MXFP8 formats using normal distribution scaled to format range.

    Uses conservative scaling (5% of max normal) to avoid saturation while creating
    diverse test data with realistic dynamic range. Max values from format_config.py.
    """
    if const_face:
        return torch.ones(size, dtype=torch.bfloat16) * const_value

    # Scale factor: use 5% of format's max normal value
    # This ensures values are well within representable range while maintaining diversity
    if stimuli_format == DataFormat.MxFp8R:
        scale = 0.05 * MXFP8_E5M2_MAX_NORMAL
    else:  # MxFp8P
        scale = 0.05 * MXFP8_E4M3_MAX_NORMAL

    face_data = torch.randn(size, dtype=torch.bfloat16) * scale

    # Add SFPU-friendly offset if needed
    if sfpu:
        face_data += 0.1

    return face_data


def generate_identity_face_tensor(
    stimuli_format: DataFormat, rows: int, cols: int
) -> torch.Tensor:
    assert rows % 16 == 0 and cols % 16 == 0, "Matrix size must be divisible by 16"

    num_faces_row = rows // 16
    num_faces_col = cols // 16
    face_height, face_width = 16, 16

    # Create output array in float32
    matrix = torch.zeros((rows, cols), dtype=torch.float32)

    # Fill each face with identity matrix
    for face_r in range(num_faces_row):
        for face_c in range(num_faces_col):
            r_start = face_r * face_height
            c_start = face_c * face_width

            # Set diagonal elements within the face
            for i in range(face_height):
                matrix[r_start + i, c_start + i] = 1

    return matrix.to(dtype=format_dict[stimuli_format])


def generate_incrementing_face_tensor(
    stimuli_format: DataFormat, rows: int, cols: int
) -> torch.Tensor:
    assert rows % 16 == 0 and cols % 16 == 0, "Matrix size must be divisible by 16"

    num_faces_row = rows // 16
    num_faces_col = cols // 16
    face_height, face_width = 16, 16

    # Create output array in float32
    matrix = torch.zeros((rows, cols), dtype=torch.float32)

    # Fill each face with its index
    for face_r in range(num_faces_row):
        for face_c in range(num_faces_col):
            face_val = float(face_r * num_faces_col + face_c + 1)
            r_start = face_r * face_height
            c_start = face_c * face_width
            matrix[r_start : r_start + face_height, c_start : c_start + face_width] = (
                face_val
            )

    return matrix.to(dtype=format_dict[stimuli_format])


def generate_face_matmul_data(
    num_faces: int,
    stimuli_format: DataFormat,
    input_dimensions=[32, 32],  # Add input_dimensions parameter
    is_matrix_A=True,  # True for matrix A (SrcB), False for matrix B (SrcA)
    face_r_dim=16,
) -> torch.Tensor:

    # Validate num_faces
    if num_faces not in [1, 2, 4]:
        raise ValueError(f"num_faces must be 1, 2, or 4, got {num_faces}")

    # Validate input_dimensions
    rows, cols = input_dimensions
    if rows % 32 != 0 or cols % 32 != 0:
        raise ValueError(
            f"Input dimensions must be multiples of 32, got {input_dimensions}"
        )

    # Calculate number of tiles needed
    tile_cnt = input_dimensions[0] // 32 * input_dimensions[1] // 32

    # Create list to store tiles --> generate each tile with the right faces zeroed out
    tiles = [
        _mask_tile(
            torch.rand(32, 32, dtype=format_dict[stimuli_format]),
            num_faces,
            not is_matrix_A,
            face_r_dim,
        ).flatten()
        for _ in range(tile_cnt)
    ]

    # Concatenate all tiles
    src = torch.cat(tiles)

    return src


def calculate_tile_and_face_counts(
    input_dimensions_A: list,
    input_dimensions_B: list,
    face_r_dim: int,
    num_faces: int,
) -> tuple[int, int, int]:
    """
    Calculate tile counts and faces to generate based on input dimensions and face configuration.
    This is the ORIGINAL function that always uses 32x32 tiles.

    Args:
        input_dimensions_A: [height, width] in elements for input A
        input_dimensions_B: [height, width] in elements for input B
        face_r_dim: Number of rows in a face (typically 16 for full faces)
        num_faces: Number of faces to generate for partial face case

    Returns:
        tuple: (tile_cnt_A, tile_cnt_B, faces_to_generate)
    """
    assert (
        face_r_dim == MAX_FACE_R_DIM or face_r_dim == input_dimensions_A[0]
    ), f"Invalid face_r_dim, got {face_r_dim}"

    # Handle partial faces
    if face_r_dim < MAX_FACE_R_DIM:
        # Partial face case: generate exactly num_faces worth of data
        tile_cnt_A, tile_cnt_B = 1, 1
        faces_to_generate = num_faces  # Generate exactly the right number of faces
    else:
        # Full tile case - always use 32x32 tiles
        tile_cnt_A = (
            input_dimensions_A[0]
            // DEFAULT_TILE_R_DIM
            * input_dimensions_A[1]
            // DEFAULT_TILE_C_DIM
        )
        tile_cnt_B = (
            input_dimensions_B[0]
            // DEFAULT_TILE_R_DIM
            * input_dimensions_B[1]
            // DEFAULT_TILE_C_DIM
        )
        faces_to_generate = MAX_NUM_FACES

    return tile_cnt_A, tile_cnt_B, faces_to_generate


def calculate_tile_and_face_counts_w_tile_dimensions(
    input_dimensions_A: list,
    input_dimensions_B: list,
    face_r_dim: int,
    num_faces: int,
    tile_dimensions: list,
) -> tuple[int, int, int]:
    """
    Calculate tile counts and faces to generate for variable tile dimensions (dense mode).

    Args:
        input_dimensions_A: [height, width] in elements for input A
        input_dimensions_B: [height, width] in elements for input B
        face_r_dim: Number of rows in a face (1, 2, 4, 8, or 16)
        num_faces: Number of faces per tile (1, 2, or 4)
        tile_dimensions: [rows, cols] for tile size

    Returns:
        tuple: (tile_cnt_A, tile_cnt_B, faces_to_generate)
    """
    validate_tile_dimensions(tile_dimensions)
    tile_r_dim, tile_c_dim = tile_dimensions

    # Calculate tile counts based on actual tile dimensions
    tile_cnt_A = (input_dimensions_A[0] // tile_r_dim) * (
        input_dimensions_A[1] // tile_c_dim
    )
    tile_cnt_B = (input_dimensions_B[0] // tile_r_dim) * (
        input_dimensions_B[1] // tile_c_dim
    )
    # Always generate all faces to fill the tile densely
    faces_to_generate = num_faces

    return tile_cnt_A, tile_cnt_B, faces_to_generate


def _get_dtype_for_format(stimuli_format: DataFormat) -> torch.dtype:
    """Get the torch dtype for a given data format."""
    if stimuli_format == DataFormat.Bfp8_b:
        return torch.bfloat16
    return format_dict[stimuli_format]


def _generate_source_tensor(
    stimuli_format: DataFormat,
    num_elements: int,
    faces_to_generate: int,
    tile_cnt: int,
    face_r_dim: int,
    const_face: bool,
    const_value: float,
    sfpu: bool,
    negative_values: bool,
    sequential: bool,
) -> torch.Tensor:
    """
    Generate a source tensor with random or sequential values.

    Args:
        stimuli_format: Data format for the tensor
        num_elements: Total number of elements to generate
        faces_to_generate: Number of faces per tile
        tile_cnt: Number of tiles
        face_r_dim: Number of rows per face
        const_face: Whether to use constant values
        const_value: Constant value to use if const_face is True
        sfpu: Whether to add SFPU-friendly offset
        negative_values: Whether to include negative values
        sequential: If True, generate sequential values (1, 2, 3, ...)

    Returns:
        torch.Tensor with generated values
    """
    dtype = _get_dtype_for_format(stimuli_format)

    if sequential:
        return torch.arange(1, num_elements + 1, dtype=dtype)

    src = []
    for _ in range(faces_to_generate * tile_cnt):
        face = generate_random_face(
            stimuli_format=stimuli_format,
            const_value=const_value,
            const_face=const_face,
            sfpu=sfpu,
            face_r_dim=face_r_dim,
            negative_values=negative_values,
        )
        src.extend(face.tolist())

    return torch.tensor(src[:num_elements], dtype=dtype)


def _clamp_mx_tensors(
    srcA_tensor: torch.Tensor,
    srcB_tensor: torch.Tensor,
    stimuli_format_A: DataFormat,
    stimuli_format_B: DataFormat,
    output_format: DataFormat = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Clamp tensors for MX format compatibility.

    Args:
        srcA_tensor: Source A tensor
        srcB_tensor: Source B tensor
        stimuli_format_A: Data format for source A
        stimuli_format_B: Data format for source B
        output_format: Optional output format for range constraints

    Returns:
        tuple: (clamped_srcA_tensor, clamped_srcB_tensor)
    """
    # Clamp inputs if both are different MX formats (use more restrictive MxFp8P)
    if stimuli_format_A.is_mx_format() and stimuli_format_B.is_mx_format():
        if stimuli_format_A != stimuli_format_B:
            srcA_tensor = torch.clamp(
                srcA_tensor, -MXFP8_E4M3_MAX_NORMAL, MXFP8_E4M3_MAX_NORMAL
            )
            srcB_tensor = torch.clamp(
                srcB_tensor, -MXFP8_E4M3_MAX_NORMAL, MXFP8_E4M3_MAX_NORMAL
            )

    # Clamp inputs based on output format to prevent excessive rounding errors
    if output_format == DataFormat.MxFp8P:
        srcA_tensor = torch.clamp(
            srcA_tensor, -MXFP8_E4M3_MAX_NORMAL, MXFP8_E4M3_MAX_NORMAL
        )
        srcB_tensor = torch.clamp(
            srcB_tensor, -MXFP8_E4M3_MAX_NORMAL, MXFP8_E4M3_MAX_NORMAL
        )
    elif output_format == DataFormat.MxFp8R:
        srcA_tensor = torch.clamp(
            srcA_tensor, -MXFP8_E5M2_MAX_NORMAL, MXFP8_E5M2_MAX_NORMAL
        )
        srcB_tensor = torch.clamp(
            srcB_tensor, -MXFP8_E5M2_MAX_NORMAL, MXFP8_E5M2_MAX_NORMAL
        )

    return srcA_tensor, srcB_tensor


def generate_stimuli(
    stimuli_format_A=DataFormat.Float16_b,
    input_dimensions_A=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
    stimuli_format_B=DataFormat.Float16_b,
    input_dimensions_B=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
    const_face=False,
    const_value_A=1,
    const_value_B=1,
    sfpu=True,
    face_r_dim=MAX_FACE_R_DIM,
    num_faces=MAX_NUM_FACES,
    negative_values=False,
    output_format=None,
    sequential_A=False,
    sequential_B=False,
):
    """
    Generate stimuli data for testing - ORIGINAL backward-compatible version.

    This is the original generate_stimuli that ALWAYS uses 32x32 tiles.
    - For full faces (face_r_dim == 16): generates 4 faces per tile (1024 elements)
    - For partial faces (face_r_dim < 16): generates num_faces worth of data

    Args:
        stimuli_format_A: Data format for source A
        input_dimensions_A: [height, width] for source A
        stimuli_format_B: Data format for source B
        input_dimensions_B: [height, width] for source B
        const_face: Whether to use constant values
        const_value_A: Constant value for source A
        const_value_B: Constant value for source B
        sfpu: Whether to add SFPU-friendly offset
        face_r_dim: Number of rows per face (typically 16 for full faces)
        num_faces: Number of faces for partial face case
        negative_values: Whether to include negative values
        output_format: Optional output format for MX range constraints
        sequential_A: If True, generate sequential values for src_A
        sequential_B: If True, generate sequential values for src_B

    Returns:
        tuple: (srcA_tensor, tile_cnt_A, srcB_tensor, tile_cnt_B)
    """
    tile_cnt_A, tile_cnt_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions_A, input_dimensions_B, face_r_dim, num_faces
    )

    num_elements_A = input_dimensions_A[0] * input_dimensions_A[1]
    num_elements_B = input_dimensions_B[0] * input_dimensions_B[1]

    srcA_tensor = _generate_source_tensor(
        stimuli_format=stimuli_format_A,
        num_elements=num_elements_A,
        faces_to_generate=faces_to_generate,
        tile_cnt=tile_cnt_A,
        face_r_dim=face_r_dim,
        const_face=const_face,
        const_value=const_value_A,
        sfpu=sfpu,
        negative_values=negative_values,
        sequential=sequential_A,
    )

    srcB_tensor = _generate_source_tensor(
        stimuli_format=stimuli_format_B,
        num_elements=num_elements_B,
        faces_to_generate=faces_to_generate,
        tile_cnt=tile_cnt_B,
        face_r_dim=face_r_dim,
        const_face=const_face,
        const_value=const_value_B,
        sfpu=sfpu,
        negative_values=negative_values,
        sequential=sequential_B,
    )

    srcA_tensor, srcB_tensor = _clamp_mx_tensors(
        srcA_tensor, srcB_tensor, stimuli_format_A, stimuli_format_B, output_format
    )

    return srcA_tensor, tile_cnt_A, srcB_tensor, tile_cnt_B


def generate_stimuli_w_tile_dimensions(
    stimuli_format_A=DataFormat.Float16_b,
    input_dimensions_A=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
    stimuli_format_B=DataFormat.Float16_b,
    input_dimensions_B=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
    const_face=False,
    const_value_A=1,
    const_value_B=1,
    sfpu=True,
    tile_dimensions=None,
    negative_values=False,
    output_format=None,
    sequential_A=False,
    sequential_B=False,
):
    """
    Generate stimuli data for testing - DENSE mode for variable tile dimensions.

    This variant generates DENSE data that fills all elements based on tile_dimensions.
    For example: tile_dimensions=[8, 32] with input_dimensions=[64, 64] produces:
    - tile_cnt = (64//8) * (64//32) = 8 * 2 = 16 tiles
    - Each tile is 8×32 = 256 elements
    - Total = 64×64 = 4096 elements (all filled)

    Args:
        stimuli_format_A: Data format for source A
        input_dimensions_A: [height, width] for source A
        stimuli_format_B: Data format for source B
        input_dimensions_B: [height, width] for source B
        const_face: Whether to use constant values
        const_value_A: Constant value for source A
        const_value_B: Constant value for source B
        sfpu: Whether to add SFPU-friendly offset
        tile_dimensions: [rows, cols] for tile size (e.g., [8, 32], [32, 16])
        negative_values: Whether to include negative values
        output_format: Optional output format for MX range constraints
        sequential_A: If True, generate sequential values for src_A
        sequential_B: If True, generate sequential values for src_B

    Returns:
        tuple: (srcA_tensor, tile_cnt_A, srcB_tensor, tile_cnt_B)
    """
    # Compute face_r_dim and num_faces from tile_dimensions
    if tile_dimensions is None:
        tile_dimensions = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]

    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim

    tile_cnt_A, tile_cnt_B, faces_to_generate = (
        calculate_tile_and_face_counts_w_tile_dimensions(
            input_dimensions_A,
            input_dimensions_B,
            face_r_dim,
            num_faces,
            tile_dimensions,
        )
    )

    num_elements_A = input_dimensions_A[0] * input_dimensions_A[1]
    num_elements_B = input_dimensions_B[0] * input_dimensions_B[1]

    srcA_tensor = _generate_source_tensor(
        stimuli_format=stimuli_format_A,
        num_elements=num_elements_A,
        faces_to_generate=faces_to_generate,
        tile_cnt=tile_cnt_A,
        face_r_dim=face_r_dim,
        const_face=const_face,
        const_value=const_value_A,
        sfpu=sfpu,
        negative_values=negative_values,
        sequential=sequential_A,
    )

    srcB_tensor = _generate_source_tensor(
        stimuli_format=stimuli_format_B,
        num_elements=num_elements_B,
        faces_to_generate=faces_to_generate,
        tile_cnt=tile_cnt_B,
        face_r_dim=face_r_dim,
        const_face=const_face,
        const_value=const_value_B,
        sfpu=sfpu,
        negative_values=negative_values,
        sequential=sequential_B,
    )

    srcA_tensor, srcB_tensor = _clamp_mx_tensors(
        srcA_tensor, srcB_tensor, stimuli_format_A, stimuli_format_B, output_format
    )

    return srcA_tensor, tile_cnt_A, srcB_tensor, tile_cnt_B


def convert_to_l1_view(
    tilized_tensor: torch.Tensor,
    input_dimensions: list,
    tile_dimensions: list = None,
) -> torch.Tensor:
    """
    Convert a tilized tensor to its L1 memory view by condensing data based on tile dimensions.

    This function extracts only the data that corresponds to the specified tile dimensions
    and places it at the beginning of each tile, with the remaining space zeroed out.
    The full tile size (1024 elements) is preserved.

    Tilized format: faces are stored sequentially [f0 (256), f1 (256), f2 (256), f3 (256)]
    Within each face, data is stored row-major (16 rows × 16 cols).

    Face layout in a 32×32 tile:
    - f0: rows 0-15, cols 0-15  (top-left)
    - f1: rows 0-15, cols 16-31 (top-right)
    - f2: rows 16-31, cols 0-15 (bottom-left)
    - f3: rows 16-31, cols 16-31 (bottom-right)

    Examples:
    - tile_dimensions=[32, 32]: full tile, no change [f0, f1, f2, f3]
    - tile_dimensions=[16, 32]: top half [f0, f1, 0, 0]
    - tile_dimensions=[32, 16]: left half [f0, f2, 0, 0]
    - tile_dimensions=[16, 16]: top-left only [f0, 0, 0, 0]
    - tile_dimensions=[8, 32]: first 8 rows [f0_rows0-7, f1_rows0-7, 0, ...]

    Args:
        tilized_tensor: Input tensor in tilized format (faces stored sequentially per tile)
        input_dimensions: [rows, cols] of the full input matrix
        tile_dimensions: [rows, cols] to keep per tile (default [32, 32])
                        rows must be one of: 1, 2, 4, 8, 16, 32
                        cols must be one of: 16, 32

    Returns:
        Tensor with condensed data at the beginning (face by face), zeros at the end
    """
    if tile_dimensions is None:
        tile_dimensions = [32, 32]

    tile_rows, tile_cols = tile_dimensions

    valid_rows = {1, 2, 4, 8, 16, 32}
    valid_cols = {16, 32}

    if tile_rows not in valid_rows:
        raise ValueError(
            f"tile_dimensions[0] (rows) must be one of {sorted(valid_rows)}, got {tile_rows}"
        )
    if tile_cols not in valid_cols:
        raise ValueError(
            f"tile_dimensions[1] (cols) must be one of {sorted(valid_cols)}, got {tile_cols}"
        )

    rows, cols = input_dimensions
    if rows % 32 != 0 or cols % 32 != 0:
        raise ValueError(
            f"Input dimensions must be multiples of 32, got {input_dimensions}"
        )

    # If using full tile dimensions, no conversion needed
    if tile_rows == 32 and tile_cols == 32:
        return tilized_tensor.flatten()

    # Calculate number of tiles
    tile_cnt = (rows // 32) * (cols // 32)
    face_rows = 16
    face_cols = 16

    # Reshape to [num_tiles, 4, 16, 16] for easier face/row manipulation
    # Face order in tilized format: [f0, f1, f2, f3]
    tensor_by_tiles = tilized_tensor.flatten().view(tile_cnt, 4, face_rows, face_cols)

    # Create output tensor with same shape, initialized to zeros
    output = torch.zeros_like(tensor_by_tiles)

    # Determine which faces to use and how many rows from each
    # tile_rows <= 16: only top faces (f0, f1), take tile_rows from each
    # tile_rows == 32: all faces, take all 16 rows from each
    # tile_cols == 16: only left faces (f0, f2)
    # tile_cols == 32: both left and right faces
    use_bottom_faces = tile_rows == 32
    use_right_faces = tile_cols == 32
    rows_per_face = tile_rows if tile_rows <= 16 else 16

    # Extract data face by face (not interleaved)
    for tile_idx in range(tile_cnt):
        out_flat = []

        # f0: always used - extract rows_per_face rows
        for row in range(rows_per_face):
            out_flat.extend(tensor_by_tiles[tile_idx, 0, row, :].tolist())

        # f1: used if tile_cols == 32 - extract rows_per_face rows
        if use_right_faces:
            for row in range(rows_per_face):
                out_flat.extend(tensor_by_tiles[tile_idx, 1, row, :].tolist())

        # f2: used if tile_rows == 32 - extract all 16 rows
        if use_bottom_faces:
            for row in range(16):
                out_flat.extend(tensor_by_tiles[tile_idx, 2, row, :].tolist())

        # f3: used if tile_rows == 32 and tile_cols == 32 - extract all 16 rows
        if use_bottom_faces and use_right_faces:
            for row in range(16):
                out_flat.extend(tensor_by_tiles[tile_idx, 3, row, :].tolist())

        # Place condensed data at the beginning of the tile
        out_flat_tensor = torch.tensor(out_flat, dtype=tilized_tensor.dtype)
        output_flat = output[tile_idx].flatten()
        output_flat[: len(out_flat)] = out_flat_tensor
        output[tile_idx] = output_flat.view(4, face_rows, face_cols)

    # Flatten and return
    return output.flatten()
