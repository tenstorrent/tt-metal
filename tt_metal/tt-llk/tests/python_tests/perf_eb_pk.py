# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# HW_FLOORS attribution, eltwise binary add LoFi on this head: block unpack and, by the scratch harness edit, the block
# pack under it: the DEST hand-off job 2 configuration (1x8, per-tile math) and the block math call (8x1); one row per
# report key.
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
    ("add_lo_8_row", F16, F16, NO, ADD, LO, [32, 256], False, DestSync.Half),
    ("add_lo_8_col_mb", F16, F16, NO, ADD, LO, [256, 32], True, DestSync.Half),
]


@pytest.mark.perf
@parametrize(cfg=CFG, run_types=[ALL_PERF_RUN_TYPES])
def test_perf_eb_pk(perf_report, cfg, run_types):
    _, fin, fout, dest_acc, op, fid, dims, block, sync = cfg
    kw = {"per_face_handoff": False, "unpack_ab_block": 1}
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
