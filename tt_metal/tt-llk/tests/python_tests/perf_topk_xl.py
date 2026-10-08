# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
The topk_xl chunk pipeline of topk_large_indices (sources/topk_xl_perf.cpp, the perf twin of topk_xl_test.cpp) on one
row: K' 512, 1024 and 2048 with 2, 8 and 32 chunks fused end to end, and the unfused row-major op path. TILE_LOOP is
cycles per input tile of the row (a chunk is K' / 1024 tiles, at least one).
"""

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, PerfRunType
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
    TILE_COUNT,
    TOPK_XL_PERF,
)

pytestmark = [skip_for_quasar, skip_for_wormhole]

ELEMENTS_PER_TILE = 1024

# (K', chunks, fused end to end)
ROWS = [(k, c, True) for k in (512, 1024, 2048) for c in (2, 8, 32)] + [
    (k, c, False) for k in (512, 1024) for c in (2, 8)
]
ROW_IDS = [f"k{k}-chunks{c}-{'fused' if f else 'unfused'}" for k, c, f in ROWS]


@pytest.mark.perf
@pytest.mark.parametrize("row", ROWS, ids=ROW_IDS)
def test_perf_topk_xl(perf_report, row):
    K, chunks, fused_e2e = row
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.UInt32)
    tiles_per_seq = (K + ELEMENTS_PER_TILE - 1) // ELEMENTS_PER_TILE
    total_tiles = chunks * tiles_per_seq
    generator = torch.Generator().manual_seed(1234)
    src_A = torch.randn(total_tiles * ELEMENTS_PER_TILE, generator=generator).to(
        torch.bfloat16
    )
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.bfloat16)
    PerfConfig(
        "sources/topk_xl_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            # 32-bit fused words, two chunks of K' 2048 in Dest at once.
            DEST_SYNC(DestSync.Full),
            TOPK_XL_PERF(
                topk_xl_k=K, topk_xl_chunks=chunks, topk_xl_fused_e2e=fused_e2e
            ),
        ],
        runtimes=[TILE_COUNT(total_tiles), LOOP_FACTOR(4)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=total_tiles,
            tile_count_B=1,
            tile_count_res=2 * tiles_per_seq,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=False,
    ).run(perf_report)
