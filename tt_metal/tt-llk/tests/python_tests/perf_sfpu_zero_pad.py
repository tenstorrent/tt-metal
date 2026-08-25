# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.golden_generators import ELEMENTS_PER_TILE
from helpers.llk_params import PerfRunType, format_dict
from helpers.param_config import generate_perf_input_dimensions, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import TILE_COUNT, ZERO_PAD_ROWS
from test_sfpu_zero_pad import FORMATS, _build_input_tile, _valid_dest_acc


@skip_for_wormhole
@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    dest_acc=lambda formats: _valid_dest_acc(formats),
    input_dimensions=lambda dest_acc: generate_perf_input_dimensions(dest_acc),
    row_range=[(8, 32)],
)
def test_perf_sfpu_zero_pad(
    perf_report, formats, dest_acc, input_dimensions, row_range
):
    valid_rows, total_rows = row_range
    torch_format = format_dict[formats.input_format]
    tile_cnt = (input_dimensions[0] * input_dimensions[1]) // ELEMENTS_PER_TILE
    one_tile = _build_input_tile(torch_format)
    src_A = torch.cat([one_tile] * tile_cnt)
    src_B = torch.zeros_like(src_A)

    configuration = PerfConfig(
        "sources/sfpu_zero_pad_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[ZERO_PAD_ROWS(valid_rows=valid_rows, total_rows=total_rows)],
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
        unpack_to_dest=formats.input_format.is_32_bit(),
    )
    configuration.run(perf_report)
