# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Headroom round, eltwise binary: binary_ng's block sections in the harness (block unpack, whole-tile hand-off), per-tile
# math calls on the head and one block math call per DEST section where the tree has it. Per-tile rows as in-job controls.
# Not for merge.
import inspect

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import BroadcastType, DestAccumulation, DestSync, MathFidelity, MathOperation, Transpose
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_eltwise_binary import _run_eltwise_binary_test

_HAS_BLOCK = "math_block" in inspect.signature(_run_eltwise_binary_test).parameters
F16, BF8, F32 = DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Float32
ADD, SUB, MUL = MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul
LO, H2, H3, H4 = MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4
NO, YES = DestAccumulation.No, DestAccumulation.Yes

# name, input, output, dest acc, op, fidelity, dims, block (unpack and math), sync
CFG = [
    ("mul_h4_8", F16, F16, NO, MUL, H4, [256, 32], True, DestSync.Half),
    ("mul_h4_16", F16, F16, NO, MUL, H4, [512, 32], True, DestSync.Half),
    ("mul_h3_8", F16, F16, NO, MUL, H3, [256, 32], True, DestSync.Half),
    ("mul_h2_8", F16, F16, NO, MUL, H2, [256, 32], True, DestSync.Half),
    ("mul_lo_8", F16, F16, NO, MUL, LO, [256, 32], True, DestSync.Half),
    ("add_lo_8", F16, F16, NO, ADD, LO, [256, 32], True, DestSync.Half),
    ("sub_lo_8", F16, F16, NO, SUB, LO, [256, 32], True, DestSync.Half),
    ("add_lo_8_bfp8", BF8, BF8, NO, ADD, LO, [256, 32], True, DestSync.Half),
    ("mul_h4_4_fp32", F16, F16, YES, MUL, H4, [128, 32], True, DestSync.Half),
    ("add_lo_4_fp32", F16, F16, YES, ADD, LO, [128, 32], True, DestSync.Half),
    ("add_lo_16_full", F16, F16, NO, ADD, LO, [512, 32], True, DestSync.Full),
    ("mul_h4_16_full", F16, F16, NO, MUL, H4, [512, 32], True, DestSync.Full),
    ("pt_mul_h4_8", F16, F16, NO, MUL, H4, [256, 32], False, DestSync.Half),
    ("pt_add_lo_8", F16, F16, NO, ADD, LO, [256, 32], False, DestSync.Half),
]


@pytest.mark.perf
@parametrize(cfg=CFG, run_types=[ALL_PERF_RUN_TYPES])
def test_perf_eb_hr(perf_report, cfg, run_types):
    _, fin, fout, dest_acc, op, fid, dims, block, sync = cfg
    kw = {"per_face_handoff": False, "unpack_ab_block": 1 if block else 0}
    if _HAS_BLOCK:
        kw["math_block"] = block
    _run_eltwise_binary_test(
        dest_acc,
        sync,
        False,
        InputOutputFormat(fin, fout),
        BroadcastType.None_,
        op,
        fid,
        Transpose.No,
        dims,
        [32, 32],
        False,
        run_types=run_types,
        loop_factor=32,
        is_perf=True,
        perf_report=perf_report,
        **kw,
    )
