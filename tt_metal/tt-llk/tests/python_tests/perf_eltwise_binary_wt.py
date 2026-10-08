# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723 fourth review, CI only): harness rows of the whole-tile program, fidelity phase outer, for the
column broadcast multiply of partial-face tiles (1x32 to 8x32), against main's per-face program (eb_handoff "face"), with the
standard form, the row broadcast and the 16x32 column broadcast as same-program controls; 8 tiles per run, 32 loops, bf16 in
and out, 16-bit and fp32 DEST."""
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
CASES = {f"mul_col_{h}x32_{f.name.lower()}": (M, C, [h, 32], f) for h in (1, 2, 4, 8) for f in (L, H2, H3, H4)}
# same-program controls: the standard form and the row broadcast keep the per-face program for partial faces
CASES.update({"mul_none_8x32_hifi4": (M, N, [8, 32], H4), "mul_row_8x32_hifi4": (M, R, [8, 32], H4), "mul_col_16x32_hifi4": (M, C, [16, 32], H4)})
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
