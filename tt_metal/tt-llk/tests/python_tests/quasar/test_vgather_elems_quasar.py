# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import format_dict
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig

GATHER_ELEM_CNT = 32
TILE_DIMENSIONS = [1, GATHER_ELEM_CNT]
NUM_FACES = 2
FACE_R_DIM = 1

TEST_INDICES = [
    26,
    12,
    25,
    25,
    29,
    6,
    10,
    24,
    8,
    17,
    15,
    20,
    11,
    16,
    30,
    29,
    20,
    22,
    4,
    19,
    0,
    7,
    4,
    25,
    5,
    12,
    4,
    3,
    31,
    6,
    16,
    26,
]


@pytest.mark.quasar
def test_vgather_elems_quasar():
    formats = InputOutputFormat(
        input_format=DataFormat.Float16_b,
        output_format=DataFormat.Float16_b,
    )

    torch.manual_seed(42)
    src_A = torch.randn(GATHER_ELEM_CNT).to(torch.bfloat16)
    indices = torch.tensor(TEST_INDICES, dtype=torch.uint8)

    golden_tensor = src_A[indices.to(torch.int64)]

    configuration = TestConfig(
        "sources/quasar/vgather_elems_quasar_test.cpp",
        formats,
        templates=[],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            indices,
            DataFormat.UInt8,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=NUM_FACES,
            face_r_dim=FACE_R_DIM,
            tile_dimensions=TILE_DIMENSIONS,
            use_dense_tile_dimensions=True,
        ),
        requires_vector_ext=True,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    mismatches = [
        (i, TEST_INDICES[i], golden_tensor[i].item(), res_tensor[i].item())
        for i in range(GATHER_ELEM_CNT)
        if res_tensor[i] != golden_tensor[i]
    ]
    assert not mismatches, (
        "UNPACR_STRIDE gather returned the wrong element for "
        f"{len(mismatches)} of {GATHER_ELEM_CNT} lanes "
        "(lane, index, expected, actual): " + str(mismatches)
    )
