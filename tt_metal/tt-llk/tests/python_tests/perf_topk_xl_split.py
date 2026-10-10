# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
perf_topk_xl.py's fused K' 512 and 2048 rows (2, 8 and 32 chunks) with the chunk split across threads
(sources/topk_xl_split_perf.cpp): the SFPU work on PACK, the copy and the face transposes on MATH, two chunks in
flight. TILE_LOOP is cycles per input tile of the row, comparable row for row with perf_topk_xl.py.
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
ROWS = [(k, c) for k in (512, 2048) for c in (2, 8, 32)]


@pytest.mark.perf
@pytest.mark.parametrize("row", ROWS, ids=[f"k{k}-chunks{c}-split" for k, c in ROWS])
def test_perf_topk_xl_split(perf_report, row):
    K, chunks = row
    tiles_per_seq = (K + ELEMENTS_PER_TILE - 1) // ELEMENTS_PER_TILE
    total_tiles = chunks * tiles_per_seq
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.UInt32)
    generator = torch.Generator().manual_seed(1234)
    src_A = torch.randn(total_tiles * ELEMENTS_PER_TILE, generator=generator).to(
        torch.bfloat16
    )
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.bfloat16)
    PerfConfig(
        "sources/topk_xl_split_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            DEST_SYNC(DestSync.Full),
            TOPK_XL_PERF(topk_xl_k=K, topk_xl_chunks=chunks, topk_xl_fused_e2e=True),
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
