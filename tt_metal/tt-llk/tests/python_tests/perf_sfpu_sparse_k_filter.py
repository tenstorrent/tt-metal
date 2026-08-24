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
from helpers.test_variant_parameters import SPARSE_K_CONFIG, TILE_COUNT
from test_sfpu_sparse_k_filter import FORMATS, _build_indices


@skip_for_wormhole
@pytest.mark.perf
@parametrize(
    dest_acc=[DestAccumulation.Yes],
    layout=[(0x3F, 14, 0x3FFF)],
    input_dimensions=lambda dest_acc: generate_perf_input_dimensions(dest_acc),
)
def test_perf_sfpu_sparse_k_filter(perf_report, dest_acc, layout, input_dimensions):
    torch_format = format_dict[FORMATS.input_format]
    tile_cnt = (input_dimensions[0] * input_dimensions[1]) // ELEMENTS_PER_TILE
    one_tile = _build_indices(layout, my_bank=0, torch_format=torch_format)
    src_A = torch.cat([one_tile] * tile_cnt)
    src_B = torch.zeros_like(src_A)
    bank_mask, shift, within_mask = layout

    configuration = PerfConfig(
        "sources/sfpu_sparse_k_filter_test.cpp",
        FORMATS,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            SPARSE_K_CONFIG(
                sparse_k_iterations=32,
                bank_mask=bank_mask,
                my_bank=0,
                global_bank_shift=shift,
                within_bank_mask=within_mask,
                out_shift=0,
            ),
        ],
        runtimes=[TILE_COUNT(tile_cnt)],
        variant_stimuli=StimuliConfig(
            src_A,
            FORMATS.input_format,
            src_B,
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=tile_cnt,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=True,
    )
    configuration.run(perf_report)
