# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
MATH_ISOLATE cost of the topk bitonic network API calls (sources/topk_network_perf.cpp).

perf_eltwise_unary_sfpu.py times TopKLocalSort / TopKMerge / TopKRebuild at one argument set
each (idir = 0, K = 64 local sort, and a datacopy per tile). This module times the bare SFPU
call, one call per "tile", across the arguments the ttnn callers use: K (local sort end phase
logK - 1), idir, and every sort mode in 16-bit and 32-bit DEST. TILE_LOOP is cycles per call.
"""

from dataclasses import dataclass

import pytest
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    TILE_COUNT,
    TemplateParameter,
)

pytestmark = [skip_for_quasar]

# (sort mode, 32-bit DEST): fused / rank-stamped keys only exist in 32-bit DEST.
MODE_DEST = [
    ("unstable", DestAccumulation.No),
    ("unstable", DestAccumulation.Yes),
    ("stable", DestAccumulation.No),
    ("stable", DestAccumulation.Yes),
    ("fused", DestAccumulation.Yes),
    ("rank_stamped", DestAccumulation.Yes),
]
MODE_DEST_IDS = [
    f"{m}-{'fp32' if d == DestAccumulation.Yes else 'fp16'}dest" for m, d in MODE_DEST
]

# local sort at every ttnn end phase, rebuild at the K values that reach phase 3 and above
# (K = 32 / skip_second = 1 is the router / grouped-gate call), merge as the unchanged control.
NETWORK_CALLS = (
    [f"sort_k{k}" for k in (4, 8, 16, 32, 64)]
    + [f"rebuild_k{k}_skip1" for k in (16, 32, 64)]
    + ["merge_k32"]
)


_NETWORK_OPS = {"sort": "TopKLocalSort", "merge": "TopKMerge", "rebuild": "TopKRebuild"}


@dataclass
class TOPK_NETWORK_PERF(TemplateParameter):
    # Field names double as perf-report columns (helpers/perf/wide_schema.py).
    mathop: str = "TopKLocalSort"
    topk_k: int = 32
    topk_idir: int = 0
    stable_sort: str = "No"
    topk_fused_stable: bool = False
    topk_rank_stamped: bool = False
    topk_rebuild_skip_second: int = 1

    def convert_to_cpp(self) -> str:
        network_op = list(_NETWORK_OPS.values()).index(self.mathop)
        logk = self.topk_k.bit_length() - 1
        return "\n".join(
            [
                f"constexpr int TOPK_NETWORK_OP = {network_op};",
                f"constexpr int TOPK_K = {self.topk_k};",
                f"constexpr int TOPK_LOGK = {logk};",
                f"constexpr int TOPK_IDIR = {self.topk_idir};",
                f"constexpr bool TOPK_STABLE_SORT = {str(self.stable_sort == 'Yes').lower()};",
                f"constexpr bool TOPK_FUSED_STABLE = {str(self.topk_fused_stable).lower()};",
                f"constexpr bool TOPK_RANK_STAMPED = {str(self.topk_rank_stamped).lower()};",
                f"constexpr int TOPK_REBUILD_SKIP_SECOND = {self.topk_rebuild_skip_second};",
            ]
        )


@pytest.mark.perf
@pytest.mark.parametrize("mode_dest", MODE_DEST, ids=MODE_DEST_IDS)
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    network=NETWORK_CALLS,
    idir=[0, 1],
    loop_factor=[32],
)
def test_perf_topk_network(perf_report, formats, mode_dest, network, idir, loop_factor):
    sort_mode, dest_acc = mode_dest
    if network.startswith("merge") and idir != 0:
        pytest.skip("merge is the unchanged control; one direction is enough")
    mathop = _NETWORK_OPS[network.split("_")[0]]
    K = int(network.split("_k")[1].split("_")[0])

    tile_count = 1
    configuration = PerfConfig(
        "sources/topk_network_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            TOPK_NETWORK_PERF(
                mathop=mathop,
                topk_k=K,
                topk_idir=idir,
                stable_sort="Yes" if sort_mode == "stable" else "No",
                topk_fused_stable=sort_mode == "fused",
                topk_rank_stamped=sort_mode == "rank_stamped",
            ),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            LOOP_FACTOR(loop_factor),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
