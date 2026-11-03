# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, VectorMode
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    VECTOR_MODE,
    generate_input_dim,
)

pytestmark = [skip_for_quasar]

ELEMENTS_PER_TILE = 1024
FORMATS = input_output_formats([DataFormat.Float32], same=True)[0]


@pytest.mark.parametrize("approx_mode", [ApproximationMode.No, ApproximationMode.Yes])
def test_sfpu_lgamma_stirling_fp32_large_reflected_argument(approx_mode):
    exponents = torch.arange(24 + 127, 255, dtype=torch.int32)
    mantissas = torch.tensor([0, 1, 0x2AAAAA, 0x400000, 0x7FFFFF], dtype=torch.int32)
    magnitudes = (
        ((exponents[:, None] << 23) | mantissas[None, :]).flatten().view(torch.float32)
    )
    x = (-magnitudes).repeat(ELEMENTS_PER_TILE // magnitudes.numel() + 1)[
        :ELEMENTS_PER_TILE
    ]
    log_z = torch.log(1.0 - x.double()).float()
    src_A = torch.cat([x, log_z]).contiguous()
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = TestConfig(
        "sources/sfpu_lgamma_stirling_fp32_test.cpp",
        FORMATS,
        templates=[
            generate_input_dim([64, 32], [64, 32]),
            APPROX_MODE(approx_mode),
            VECTOR_MODE(VectorMode.RC),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            FORMATS.input_format,
            src_B,
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=2,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
        compile_time_formats=True,
    )
    res = torch.tensor(
        configuration.run().result[:ELEMENTS_PER_TILE], dtype=torch.float32
    )

    bad = ~(torch.isfinite(res) & (res > 0))
    assert (
        not bad.any()
    ), f"lgamma(1 - x) for x = {x[bad][:4].tolist()} came out as {res[bad][:4].tolist()}, expected finite and positive"
