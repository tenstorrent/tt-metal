# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.llk_params import L1Accumulation, PerfRunType, Tilize
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from test_pack_dest_bank import create_pack_dest_bank_config, pack_dest_bank_cases


@pytest.mark.perf
@parametrize(
    case=pack_dest_bank_cases,
    l1_acc=[L1Accumulation.No, L1Accumulation.Yes],
    num_faces=4,
    tilize=[Tilize.No],
    dest_index=0,
    loop_factor=[8, 128, 512],
)
def test_perf_pack_dest_bank(
    perf_report,
    case,
    l1_acc,
    num_faces,
    tilize,
    dest_index,
    loop_factor,
):
    formats = case.formats
    dest_acc = case.dest_acc
    tile_cnt = case.num_blocks * case.num_tiles_in_block

    src_A = torch.ones(tile_cnt * 1024, dtype=torch.bfloat16)

    # src_B is not used in this test but needed for StimuliConfig
    src_B = torch.zeros_like(src_A)
    tile_cnt_B = tile_cnt

    configuration = create_pack_dest_bank_config(
        formats,
        dest_acc,
        l1_acc,
        num_faces,
        tilize,
        dest_index,
        tile_cnt,
        case.num_blocks,
        case.num_tiles_in_block,
        StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt,
            num_faces=num_faces,
        ),
        loop_factor=loop_factor,
        is_perf=True,
        run_types=[PerfRunType.PACK_ISOLATE],
    )

    configuration.run(perf_report)
