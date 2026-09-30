# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
The phases of the generic TopK network per call, on the pipeline perf kernel (sources/topk_perf.cpp) in
MATH_ISOLATE with the datacopies dropped: one phase per build (local sort, merge, rebuild, or nothing: the loop
skeleton), so per call = (cycles per row of the phase build - the skeleton) / calls per row (local sort Wt / 2,
merge Wt - 1, rebuild Wt / 2 for a row of Wt value tiles). The rows with the copies kept give the datacopy carrier
of a step; the tile0_sorted rows time the local sort with its phases 0 to 4 on the second tile only. The whole
pipeline in the four run types is perf_topk.py.
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

DESC = TopKSortDirection.Descending
RING_TILES = 16

# (W, K, direction, sort mode, phase, drop_copy, tile0_sorted)
PHASE_ROWS = (
    [(4096, 32, DESC, "unstable", ph, True, False) for ph in ("full", "sort", "merge", "rebuild", "copy")]
    + [(4096, 32, DESC, "unstable", ph, False, False) for ph in ("full", "copy")]
    + [(4096, 64, DESC, "unstable", ph, True, False) for ph in ("full", "sort", "merge", "rebuild")]
    + [(4096, 32, DESC, "stable", ph, True, False) for ph in ("sort", "merge", "rebuild")]
    + [(4096, 32, DESC, "fused", ph, True, False) for ph in ("sort", "merge", "rebuild", "fuse")]
    + [(4096, 32, DESC, "rank_stamped", ph, True, False) for ph in ("sort", "merge", "rebuild")]
    + [(4096, 64, DESC, m, "sort", True, False) for m in ("stable", "rank_stamped")]
    + [(4096, 64, DESC, m, "sort", True, True) for m in ("unstable", "stable", "rank_stamped")]
)
PHASE_IDS = [
    f"w{w}-k{k}-{m}-{ph}{'-nocopy' if drop else ''}{'-tile0sorted' if t0 else ''}"
    for w, k, d, m, ph, drop, t0 in PHASE_ROWS
]


@pytest.mark.perf
@pytest.mark.parametrize("row", PHASE_ROWS, ids=PHASE_IDS)
def test_perf_topk_phase(perf_report, row):
    W, K, direction, sort_mode, phase, drop_copy, tile0_sorted = row
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    stable = sort_mode == "stable"
    fused = sort_mode == "fused"
    rank_stamped = sort_mode == "rank_stamped"
    wt = W // 32
    PerfConfig(
        "sources/topk_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
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
            TOPK_PERF(topk_phase=phase, topk_drop_copy=drop_copy, topk_tile0_sorted=tile0_sorted),
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
        dest_acc=DestAccumulation.Yes if (fused or rank_stamped) else DestAccumulation.No,
    ).run(perf_report)
