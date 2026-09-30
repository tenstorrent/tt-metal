# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
The generic TopK pipeline (sources/topk_perf.cpp, the perf twin of topk_test.cpp): transposed unpack,
datacopies, local sort, merge and rebuild per 2-tile slab, pack, over a tile row of W values with uint16
indices. TILE_LOOP is cycles per value tile of the row; a row of Wt value tiles has Wt - 1 tile-pair steps.

test_perf_topk_pipeline times the whole pipeline in the four run types over the width, K, the direction and
the sort mode. test_perf_topk_phase times the network in MATH_ISOLATE with the datacopies dropped, one phase
per build (local sort, merge, rebuild, nothing), so the phases come out per call at the parameters the
kernels use; the rows with the copies kept give the datacopy carrier of a step, and the tile0_sorted rows the
local sort with its phases 0 to 4 on the second tile only.
"""

from dataclasses import dataclass

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
    TemplateParameter,
)

pytestmark = [skip_for_quasar]

PHASES = {"full": 0, "sort": 1, "merge": 2, "rebuild": 3, "copy": 4, "fuse": 5}
DESC, ASC = TopKSortDirection.Descending, TopKSortDirection.Ascending
RING_TILES = 16


@dataclass
class TOPK_PERF(TemplateParameter):
    # Field names double as perf-report columns (helpers/perf/wide_schema.py).
    topk_phase: str = "full"
    topk_drop_copy: bool = False
    topk_tile0_sorted: bool = False

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr int TOPK_PERF_PHASE = {PHASES[self.topk_phase]};",
                f"constexpr bool TOPK_PERF_DROP_COPY = {str(self.topk_drop_copy).lower()};",
                f"constexpr bool TOPK_PERF_TILE0_SORTED = {str(self.topk_tile0_sorted).lower()};",
            ]
        )


def topk_perf_config(W, K, direction, sort_mode, phase, drop_copy, tile0_sorted, run_types, loop_factor=16):
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    stable = sort_mode == "stable"
    fused = sort_mode == "fused"
    rank_stamped = sort_mode == "rank_stamped"
    wt = W // 32  # value tiles per tile row; the matrix holds the index tiles too
    return PerfConfig(
        "sources/topk_perf.cpp",
        formats,
        run_types=run_types,
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
            LOOP_FACTOR(loop_factor),
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
    )


# (W, K, direction, sort mode): the pipeline rows. The per-step cost does not depend on W (the stage 2 sweep
# measured 2840 to 2899 cycles per step from W 1024 to 8192), so three widths stand for the sweep.
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

# (W, K, direction, sort mode, phase, drop_copy, tile0_sorted): the phase rows, MATH_ISOLATE.
PHASE_ROWS = (
    [(4096, 32, DESC, "unstable", ph, True, False) for ph in ("full", "sort", "merge", "rebuild", "copy")]
    + [(4096, 32, DESC, "unstable", ph, False, False) for ph in ("full", "copy")]
    + [(4096, 64, DESC, "unstable", ph, True, False) for ph in ("full", "sort", "merge", "rebuild")]
    + [(4096, 32, DESC, "stable", ph, True, False) for ph in ("sort", "merge", "rebuild")]
    + [(4096, 32, DESC, "fused", ph, True, False) for ph in ("sort", "merge", "rebuild", "fuse")]
    + [(4096, 32, DESC, "rank_stamped", ph, True, False) for ph in ("sort", "merge", "rebuild")]
    + [(4096, 64, DESC, m, "sort", True, True) for m in ("unstable", "stable", "rank_stamped")]
)
PHASE_IDS = [
    f"w{w}-k{k}-{'desc' if d == DESC else 'asc'}-{m}-{ph}{'-nocopy' if drop else ''}{'-tile0sorted' if t0 else ''}"
    for w, k, d, m, ph, drop, t0 in PHASE_ROWS
]


@pytest.mark.perf
@pytest.mark.parametrize("row", PIPELINE_ROWS, ids=PIPELINE_IDS)
def test_perf_topk_pipeline(perf_report, row):
    W, K, direction, sort_mode = row
    topk_perf_config(
        W,
        K,
        direction,
        sort_mode,
        "full",
        False,
        False,
        [PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE, PerfRunType.MATH_ISOLATE, PerfRunType.PACK_ISOLATE],
    ).run(perf_report)


@pytest.mark.perf
@pytest.mark.parametrize("row", PHASE_ROWS, ids=PHASE_IDS)
def test_perf_topk_phase(perf_report, row):
    W, K, direction, sort_mode, phase, drop_copy, tile0_sorted = row
    topk_perf_config(W, K, direction, sort_mode, phase, drop_copy, tile0_sorted, [PerfRunType.MATH_ISOLATE]).run(perf_report)
