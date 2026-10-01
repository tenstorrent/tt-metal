# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
The generic TopK pipeline (sources/topk_perf.cpp, the perf twin of topk_test.cpp) in the four run types.
TILE_LOOP is cycles per value tile of the row; the per-phase split of the same kernel is perf_topk_phase.py.
"""

import pytest
from conftest import skip_for_quasar
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType, TopKSortDirection
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    INPUT_DIMENSIONS,
    LOOP_FACTOR,
    TILE_COUNT,
    TOPK,
    TOPK_PERF,
)

pytestmark = [skip_for_quasar]

DESC, ASC = TopKSortDirection.Descending, TopKSortDirection.Ascending
RING_TILES = 16

# (W, K, direction, sort mode)
PIPELINE_ROWS = [
    (1024, 32, DESC, "unstable"),
    (4096, 32, DESC, "unstable"),
    (8192, 32, DESC, "unstable"),
    (4096, 64, DESC, "unstable"),
    (4096, 32, ASC, "unstable"),
    (4096, 32, DESC, "stable"),
    (4096, 32, DESC, "fused"),
]
PIPELINE_IDS = [f"w{w}-k{k}-{'desc' if d == DESC else 'asc'}-{m}" for w, k, d, m in PIPELINE_ROWS]


@pytest.mark.perf
@pytest.mark.parametrize("row", PIPELINE_ROWS, ids=PIPELINE_IDS)
def test_perf_topk_pipeline(perf_report, row):
    W, K, direction, sort_mode = row
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    stable = sort_mode == "stable"
    fused = sort_mode == "fused"
    rank_stamped = sort_mode == "rank_stamped"
    wt = W // 32  # value tiles per tile row
    PerfConfig(
        "sources/topk_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE, PerfRunType.MATH_ISOLATE, PerfRunType.PACK_ISOLATE],
        templates=[
            DEST_SYNC(),
            TOPK(
                topk_k=K,
                topk_matrix_width=2 * W,
                topk_sort_direction=direction,
                topk_stable_sort=stable,
                topk_fused_stable=fused,
                topk_rank_stamped=rank_stamped,
            ),
            TOPK_PERF(topk_phase="full", topk_drop_copy=False, topk_tile0_sorted=False),
        ],
        runtimes=[
            INPUT_DIMENSIONS(1, 2 * wt),
            TILE_COUNT(wt),
            LOOP_FACTOR(16),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=RING_TILES,
            tile_count_B=RING_TILES,
            tile_count_res=RING_TILES,
        ),
        unpack_to_dest=False,
        # Fused and rank-stamped keys are 32-bit words.
        dest_acc=DestAccumulation.Yes if (fused or rank_stamped) else DestAccumulation.No,
    ).run(perf_report)
