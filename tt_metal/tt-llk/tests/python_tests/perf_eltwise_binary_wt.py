# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723 third review, CI only): harness rows of the whole-tile programs, fidelity phase outer, for
partial-face tiles (1x32, 8x32) and the column broadcast on 1 x 2-face 16x32 tiles, against main's per-face program
(eb_handoff "face"), 8 tiles per run, 32 loops, bf16 in and out, 16-bit and fp32 DEST."""
from dataclasses import dataclass

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import BroadcastType, DestAccumulation, DestSync, MathFidelity, MathOperation, Transpose
from helpers.param_config import parametrize
from helpers.perf.core import PerfRunType
from helpers.test_variant_parameters import TemplateParameter
from test_eltwise_binary import _run_eltwise_binary_test


@dataclass
class EB_HANDOFF(TemplateParameter):
    eb_handoff: str = "face"

    def convert_to_cpp(self) -> str:
        return f"constexpr bool EB_HANDOFF_TILE = {'true' if self.eb_handoff == 'tile' else 'false'};"


M, A = MathOperation.Elwmul, MathOperation.Elwadd
N, C, R, S = BroadcastType.None_, BroadcastType.Column, BroadcastType.Row, BroadcastType.Scalar
L, H2, H3, H4 = MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4
CASES = {
    "mul_none_8x32_lofi": (M, N, [8, 32], L),
    "mul_none_8x32_hifi2": (M, N, [8, 32], H2),
    "mul_none_8x32_hifi3": (M, N, [8, 32], H3),
    "mul_none_8x32_hifi4": (M, N, [8, 32], H4),
    "add_none_8x32_lofi": (A, N, [8, 32], L),
    "mul_col_8x32_lofi": (M, C, [8, 32], L),
    "mul_col_8x32_hifi2": (M, C, [8, 32], H2),
    "mul_col_8x32_hifi4": (M, C, [8, 32], H4),
    "mul_col_16x32_lofi": (M, C, [16, 32], L),
    "mul_col_16x32_hifi2": (M, C, [16, 32], H2),
    "mul_col_16x32_hifi4": (M, C, [16, 32], H4),
    "mul_row_8x32_hifi4": (M, R, [8, 32], H4),
    "mul_scalar_8x32_hifi4": (M, S, [8, 32], H4),
    "mul_none_1x32_hifi4": (M, N, [1, 32], H4),
    "add_none_1x32_lofi": (A, N, [1, 32], L),
}
RUN_TYPES = [PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE, PerfRunType.MATH_ISOLATE]


@pytest.mark.perf
@parametrize(
    case=list(CASES),
    handoff=["face", "tile"],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    run_types=[RUN_TYPES],
)
def test_perf_eltwise_binary_wt(perf_report, case, handoff, dest_acc, run_types):
    op, bcast, tile, fid = CASES[case]
    _run_eltwise_binary_test(
        dest_acc,
        DestSync.Half,
        False,
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        bcast,
        op,
        fid,
        Transpose.No,
        [8 * tile[0], 32],
        tile,
        False,
        is_perf=True,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=32,
        per_face_handoff=handoff == "face",
        extra_templates=(EB_HANDOFF(handoff),),
    )
