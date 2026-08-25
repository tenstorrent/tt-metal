# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.golden_generators import ELEMENTS_PER_TILE
from helpers.llk_params import DestAccumulation, PerfRunType, format_dict
from helpers.param_config import generate_perf_input_dimensions, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    SFPU_SCALE_EN,
    SFPU_UNARY_SCALAR,
    TILE_COUNT,
)
from test_sfpu_sdpa_exp_unclamped import BF16_ONE, FORMATS


@skip_for_wormhole
@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No],
    input_dimensions=lambda dest_acc: generate_perf_input_dimensions(dest_acc),
    scale_en=[False, True],
)
def test_perf_sfpu_sdpa_exp_unclamped(
    perf_report, formats, dest_acc, input_dimensions, scale_en
):
    torch.manual_seed(0)
    torch_format = format_dict[formats.input_format]
    tile_cnt = (input_dimensions[0] * input_dimensions[1]) // ELEMENTS_PER_TILE
    src_A = torch.empty(tile_cnt * ELEMENTS_PER_TILE, dtype=torch_format).uniform_(
        -20.0, 0.0
    )
    src_B = torch.zeros_like(src_A)

    configuration = PerfConfig(
        "sources/sfpu_sdpa_exp_unclamped_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            SFPU_SCALE_EN(scale_en=scale_en),
            SFPU_UNARY_SCALAR(value_bits=BF16_ONE),
        ],
        runtimes=[TILE_COUNT(tile_cnt)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=tile_cnt,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )
    configuration.run(perf_report)
