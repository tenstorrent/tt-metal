# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

from .format_arg_mapping import format_dict
from .format_config import DataFormat


def flatten_list(sublists):
    return [item for sublist in sublists for item in sublist]


def generate_random_face(
    stimuli_format=DataFormat.Float16_b, const_value=1, const_face=False
):

    if stimuli_format in [DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32]:
        if const_face:
            srcA_face = torch.ones(256, dtype=format_dict[stimuli_format]) * const_value
        else:  # random for both faces
            srcA_face = torch.rand(256, dtype=format_dict[stimuli_format]) + 0.1

    elif stimuli_format == DataFormat.Bfp8_b:
        size = 256
        integer_part = torch.randint(0, 3, (size,))
        fraction = torch.randint(0, 16, (size,)).to(dtype=torch.bfloat16) / 16.0
        if const_face:
            srcA_face = torch.ones(256, dtype=torch.bfloat16) * const_value
        else:
            srcA_face = integer_part.to(dtype=torch.bfloat16) + fraction

    elif stimuli_format == DataFormat.Int32:
        srcA_face = torch.arange(256)

    return srcA_face


def generate_random_face_ab(
    stimuli_format_A,
    stimuli_format_B,
    const_face=False,
    const_value_A=1,
    const_value_B=2,
):
    return generate_random_face(
        stimuli_format_A, const_value_A, const_face
    ), generate_random_face(stimuli_format_B, const_value_B, const_face)


def generate_stimuli(
    stimuli_format_A=DataFormat.Float16_b,
    stimuli_format_B=DataFormat.Float16_b,
    tile_cnt=1,
    const_face=False,
    const_value_A=1,
    const_value_B=1,
):

    srcA = []
    srcB = []

    for _ in range(4 * tile_cnt):
        face_a, face_b = generate_random_face_ab(
            stimuli_format_A, stimuli_format_B, const_face, const_value_A, const_value_B
        )
        srcA.extend(face_a.tolist())
        srcB.extend(face_b.tolist())

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
    return torch.tensor(srcA, dtype=dtype_A), torch.tensor(srcB, dtype=dtype_B)
