# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, L1Accumulation, PerfRunType, Tilize
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from test_pack_dest_bank import create_pack_dest_bank_config


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            # DataFormat.Float16,
            # DataFormat.Float32,
            # DataFormat.Bfp8_b,  # Can be enabled if needed
        ]
    ),
    dest_acc=[DestAccumulation.No],
    l1_acc=[L1Accumulation.No, L1Accumulation.Yes],
    num_faces=4,
    tilize=[Tilize.No],
    dest_index=0,
    num_blocks=[1, 2],
    num_tiles_in_block=[4, 8],
    loop_factor=[8, 128, 512],
)
def test_perf_pack_dest_bank(
    perf_report,
    formats,
    dest_acc,
    l1_acc,
    num_faces,
    tilize,
    dest_index,
    num_blocks,
    num_tiles_in_block,
    loop_factor,
):
    if (num_blocks, num_tiles_in_block) not in {(1, 4), (2, 4), (1, 8)}:
        pytest.skip(
            "Local perf sweep only uses 1x4, 2x4, and 1x8 blocked-pack patterns"
        )

    tile_cnt = num_blocks * num_tiles_in_block

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
        num_blocks,
        num_tiles_in_block,
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
