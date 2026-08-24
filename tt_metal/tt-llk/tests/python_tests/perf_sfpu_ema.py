# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    PerfRunType,
    format_dict,
)
from helpers.param_config import generate_perf_input_dimensions, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import APPROX_MODE, EMA_ALPHA_BETA, TILE_COUNT
from helpers.tilize_untilize import tilize_block
from test_sfpu_ema import EMA_ALPHA, EMA_BETA, _f32_bits


@pytest.mark.perf
@parametrize(
    dest_acc=[DestAccumulation.No],
    input_dimensions=lambda dest_acc: [
        dims for dims in generate_perf_input_dimensions(dest_acc) if dims[1] == TILE_DIM
    ],
)
def test_perf_sfpu_ema(perf_report, dest_acc, input_dimensions):
    torch.manual_seed(0)
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    torch_format = format_dict[formats.input_format]
    tile_cnt = (input_dimensions[0] * input_dimensions[1]) // ELEMENTS_PER_TILE
    src_A = torch.empty(tile_cnt * ELEMENTS_PER_TILE, dtype=torch_format).uniform_(
        -4.0, 4.0
    )
    src_B = torch.zeros_like(src_A)
    src_A_tilized = tilize_block(
        src_A, input_dimensions, stimuli_format=formats.input_format
    ).flatten()

    configuration = PerfConfig(
        "sources/sfpu_ema_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            APPROX_MODE(ApproximationMode.No),
            EMA_ALPHA_BETA(
                alpha_bits=_f32_bits(EMA_ALPHA), beta_bits=_f32_bits(EMA_BETA)
            ),
        ],
        runtimes=[TILE_COUNT(tile_cnt)],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
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
        disable_format_inference=True,
        compile_time_formats=True,
    )
    configuration.run(perf_report)
