# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

from .format_config import (
    MXFP8_E4M3_MAX_NORMAL,
    MXFP8_E5M2_MAX_NORMAL,
    DataFormat,
)
from .llk_params import format_dict


def flatten_list(sublists):
    return [item for sublist in sublists for item in sublist]


def _mask_tile(tile: torch.Tensor, num_faces: int, is_matrix_A: bool) -> torch.Tensor:
    masked = tile.clone()
    if num_faces == 1:
        # Keep only f0
        masked[:16, 16:] = 0  # Zero f1
        masked[16:, :] = 0  # Zero f2, f3
    elif num_faces == 2:
        if is_matrix_A:
            # Matrix A: keep f0, f2
            masked[:16, 16:] = 0  # Zero f1
            masked[16:, 16:] = 0  # Zero f3
        else:
            # Matrix B: keep f0, f1
            masked[16:, :] = 0  # Zero f2, f3
    return masked


def generate_random_face(
    stimuli_format=DataFormat.Float16_b,
    const_value=1,
    const_face=False,
    sfpu=True,
    face_r_dim=16,
    negative_values=False,
):
    size = face_r_dim * 16  # face_r_dim rows × 16 columns

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


def generate_face_matmul_data(
    num_faces: int,
    stimuli_format: DataFormat,
    input_dimensions=[32, 32],  # Add input_dimensions parameter
    is_matrix_A=True,  # True for matrix A (SrcB), False for matrix B (SrcA)
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
            is_matrix_A,
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

    Args:
        input_dimensions_A: [height, width] in elements for input A
        input_dimensions_B: [height, width] in elements for input B
        face_r_dim: Number of rows in a face (typically 16 for full faces)
        num_faces: Number of faces to generate for partial face case

    Returns:
        tuple: (tile_cnt_A, tile_cnt_B, faces_to_generate)
    """
    assert (
        face_r_dim == 16 or face_r_dim == input_dimensions_A[0]
    ), f"Invalid face_r_dim, got {face_r_dim}"

    # Handle partial faces
    if face_r_dim < 16:
        # Partial face case: generate exactly num_faces worth of data
        tile_cnt_A, tile_cnt_B = 1, 1
        faces_to_generate = num_faces  # Generate exactly the right number of faces
    else:
        # Full tile case
        tile_cnt_A = input_dimensions_A[0] // 32 * input_dimensions_A[1] // 32
        tile_cnt_B = input_dimensions_B[0] // 32 * input_dimensions_B[1] // 32
        faces_to_generate = 4

    return tile_cnt_A, tile_cnt_B, faces_to_generate


def generate_stimuli(
    stimuli_format_A=DataFormat.Float16_b,
    input_dimensions_A=[32, 32],
    stimuli_format_B=DataFormat.Float16_b,
    input_dimensions_B=[32, 32],
    const_face=False,
    const_value_A=1,
    const_value_B=1,
    sfpu=True,
    face_r_dim=16,  # Add face_r_dim parameter
    num_faces=4,  # Add num_faces parameter for partial faces
    negative_values=False,
    output_format=None,  # Optional output format to consider for range constraints (MX)
    sequential_A=False,  # Generate sequential values (1, 2, 3, ...) for src_A
    sequential_B=False,  # Generate sequential values (1, 2, 3, ...) for src_B
):
    """
    Generate stimuli data for testing.

    Args:
        sequential_A: If True, generates sequential values starting from 1 for src_A.
                     src_A will have values 1, 2, 3, ...
        sequential_B: If True, generates sequential values starting from 1 for src_B.
                     src_B will have values 1, 2, 3, ...
        output_format: Optional output format to consider for range constraints (MX formats)
    """

    tile_cnt_A, tile_cnt_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions_A, input_dimensions_B, face_r_dim, num_faces
    )

    dtype_A = (
        format_dict[stimuli_format_A]
        if stimuli_format_A != DataFormat.Bfp8_b
        else torch.bfloat16
    )
    dtype_B = (
        format_dict[stimuli_format_B]
        if stimuli_format_B != DataFormat.Bfp8_b
        else torch.bfloat16
    )

    num_elements_A = input_dimensions_A[0] * input_dimensions_A[1]
    num_elements_B = input_dimensions_B[0] * input_dimensions_B[1]

    # Generate src_A
    if sequential_A:
        srcA_tensor = torch.arange(1, num_elements_A + 1, dtype=dtype_A)
    else:
        srcA = []
        for _ in range(faces_to_generate * tile_cnt_A):
            face_a = generate_random_face(
                stimuli_format=stimuli_format_A,
                const_value=const_value_A,
                const_face=const_face,
                sfpu=sfpu,
                face_r_dim=face_r_dim,
                negative_values=negative_values,
            )
            srcA.extend(face_a.tolist())
        srcA_tensor = torch.tensor(srcA[:num_elements_A], dtype=dtype_A)

    # Generate src_B
    if sequential_B:
        srcB_tensor = torch.arange(1, num_elements_B + 1, dtype=dtype_B)
    else:
        srcB = []
        for _ in range(faces_to_generate * tile_cnt_B):
            face_b = generate_random_face(
                stimuli_format=stimuli_format_B,
                const_value=const_value_B,
                const_face=const_face,
                sfpu=sfpu,
                face_r_dim=face_r_dim,
                negative_values=negative_values,
            )
            srcB.extend(face_b.tolist())
        srcB_tensor = torch.tensor(srcB[:num_elements_B], dtype=dtype_B)

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

    return srcA_tensor, tile_cnt_A, srcB_tensor, tile_cnt_B
