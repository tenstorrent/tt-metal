# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Regression audit of the eltwise binary per-tile hand-off: the recorded residual configurations (LoFi add and subtract at
# one tile per DEST section, 16-bit inputs into Bfp8_b, fp32 add) with the per-tile program forced where the tree has it,
# main's per-face program on main. Not for merge.
import inspect

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import BroadcastType, DestAccumulation, DestSync, MathFidelity, MathOperation, Transpose
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_eltwise_binary import _run_eltwise_binary_test

_KW = {"per_face_handoff": False} if "per_face_handoff" in inspect.signature(_run_eltwise_binary_test).parameters else {}


@pytest.mark.perf
@parametrize(
    cfg=[
        ("bf16_add_1", DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No, MathOperation.Elwadd, [32, 32]),
        ("bf16_sub_1", DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No, MathOperation.Elwsub, [32, 32]),
        ("bf16_add_8", DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No, MathOperation.Elwadd, [256, 32]),
        ("bf16_bfp8_add_8", DataFormat.Float16_b, DataFormat.Bfp8_b, DestAccumulation.No, MathOperation.Elwadd, [256, 32]),
        ("fp16_bfp8_add_8", DataFormat.Float16, DataFormat.Bfp8_b, DestAccumulation.No, MathOperation.Elwadd, [256, 32]),
        ("fp32_add_1", DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes, MathOperation.Elwadd, [32, 32]),
        ("fp32_add_4", DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes, MathOperation.Elwadd, [128, 32]),
    ],
    run_types=[ALL_PERF_RUN_TYPES],
)
def test_perf_eb_resid(perf_report, cfg, run_types):
    _, fin, fout, dest_acc, op, dims = cfg
    _run_eltwise_binary_test(
        dest_acc,
        DestSync.Half,
        False,
        InputOutputFormat(fin, fout),
        BroadcastType.None_,
        op,
        MathFidelity.LoFi,
        Transpose.No,
        dims,
        [32, 32],
        False,
        run_types=run_types,
        loop_factor=32,
        is_perf=True,
        perf_report=perf_report,
        **_KW,
    )
