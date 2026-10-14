# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig

VECTOR_ELEMS = 32
TILE_DIMENSIONS = [1, VECTOR_ELEMS]
NUM_FACES = 2
FACE_R_DIM = 1


@pytest.mark.quasar
def test_vector_add_l1_quasar():
    formats = InputOutputFormat(
        input_format=DataFormat.Int32,
        output_format=DataFormat.Int32,
    )

    torch.manual_seed(7)
    src_a = torch.randint(0, 1000, (VECTOR_ELEMS,), dtype=torch.int32)
    src_b = torch.randint(0, 1000, (VECTOR_ELEMS,), dtype=torch.int32)
    golden = src_a + src_b

    configuration = TestConfig(
        "sources/quasar/vector_add_l1_quasar_test.cpp",
        formats,
        templates=[],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_a,
            DataFormat.Int32,
            src_b,
            DataFormat.Int32,
            DataFormat.Int32,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=NUM_FACES,
            face_r_dim=FACE_R_DIM,
            tile_dimensions=TILE_DIMENSIONS,
            use_dense_tile_dimensions=True,
            twos_complement=True,
        ),
        dest_acc=DestAccumulation.Yes,
        disable_format_inference=True,
        requires_vector_ext=True,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == VECTOR_ELEMS
    res = torch.tensor(res_from_L1, dtype=torch.int32)
    mismatches = [
        (i, src_a[i].item(), src_b[i].item(), golden[i].item(), res[i].item())
        for i in range(VECTOR_ELEMS)
        if res[i] != golden[i]
    ]
    assert not mismatches, (
        f"TRISC0 vector add wrote wrong values for {len(mismatches)} of {VECTOR_ELEMS} lanes "
        "(lane, a, b, expected, actual): " + str(mismatches)
    )
