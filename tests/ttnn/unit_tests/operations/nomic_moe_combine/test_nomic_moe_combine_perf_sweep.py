# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-knob sweep for nomic_moe_combine (run under `scripts/run_safe_pytest.sh --profile`; correctness is checked too).

Each case monkeypatches the op module's knobs (BATCH_READS, BATCH_DEPTH, READER_PLACEMENT, DEST_GROUP_TILES,
FUSED_TILIZE, ROWS_DEPTH_TILE_ROWS) and runs the golden perf setup of one LOOSE_CASES shape (real_text routing,
96 KB of L1 reserved). The profiler CSV's GenericOp rows come out in test order. Pick variants with
NMC_SWEEP="name1,name2" (default: baseline only).
"""

from __future__ import annotations

import importlib
import os

import pytest
import ttnn

from eval.golden_tests.nomic_moe_combine.feature_spec import LOOSE_CASES
from eval.golden_tests.nomic_moe_combine.helpers import run_nomic_moe_combine

op = importlib.import_module("ttnn.operations.nomic_moe_combine.nomic_moe_combine")

R0 = (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)
R1 = (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1)
R0N1 = (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1)
R1N0 = (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0)

VARIANTS = {
    "baseline": {},
    "b8d2": dict(BATCH_READS=8, BATCH_DEPTH=2),
    "b4d4": dict(BATCH_READS=4, BATCH_DEPTH=4),
    "b8d3": dict(BATCH_READS=8, BATCH_DEPTH=3),
    "b8d4": dict(BATCH_READS=8, BATCH_DEPTH=4),
    "b16d2": dict(BATCH_READS=16, BATCH_DEPTH=2),
    "b2d4": dict(BATCH_READS=2, BATCH_DEPTH=4),
    "b4d3": dict(BATCH_READS=4, BATCH_DEPTH=3),
    "swap_riscs": dict(READER_PLACEMENT=(R1, R0)),
    "swap_nocs": dict(READER_PLACEMENT=(R0N1, R1N0)),
    "dest2": dict(DEST_GROUP_TILES=2),
    "rows_depth2": dict(ROWS_DEPTH_TILE_ROWS=2),
    "r1": dict(FUSED_TILIZE=True),
    "index_concurrent": dict(INDEX_FIRST=False),
    "weights_all_at_start": dict(WEIGHTS_LOOKAHEAD_ROWS=None),
    "weights_lookahead2": dict(WEIGHTS_LOOKAHEAD_ROWS=2),
    "weights_lookahead1": dict(WEIGHTS_LOOKAHEAD_ROWS=1),
    "side_vc0": dict(SIDE_READ_VC=0),
    "weights_one_read": dict(WEIGHTS_ONE_READ=True),
    "side_after1": dict(SIDE_READS_AFTER_BATCHES=1),
    "side_after2": dict(SIDE_READS_AFTER_BATCHES=2),
    "side_after4": dict(SIDE_READS_AFTER_BATCHES=4),
    "side_after8": dict(SIDE_READS_AFTER_BATCHES=8),
    "side_vc2": dict(SIDE_READ_VC=2),
    "side_vc3": dict(SIDE_READ_VC=3),
    "bias_all_banks": dict(BIAS_COPY_OFFSET=0),
    "design_startup": dict(INDEX_FIRST=False, BIAS_COPY_OFFSET=0),
    # Refinement 2: group_shared_index_weights (default: both shared).
    "share_none": dict(SHARE_INDEX=False, SHARE_WEIGHTS=False),
    "share_index_only": dict(SHARE_INDEX=True, SHARE_WEIGHTS=False),
    "share_weights_only": dict(SHARE_INDEX=False, SHARE_WEIGHTS=True),
    "stagger_leaders": dict(STAGGER_LEADERS=True),
    # Refinement 3: compute block surface.
    "full_sync": dict(DST_FULL_SYNC=True),
    "hifi3": dict(MATH_FIDELITY=ttnn.MathFidelity.HiFi3),
    "hifi4": dict(MATH_FIDELITY=ttnn.MathFidelity.HiFi4),
    # measurement only: fails the precision bars (rel RMS 0.0036)
    "hifi2": dict(MATH_FIDELITY=ttnn.MathFidelity.HiFi2, PRECISION_FIDELITIES=(ttnn.MathFidelity.HiFi2,)),
    "full_sync_hifi3": dict(DST_FULL_SYNC=True, MATH_FIDELITY=ttnn.MathFidelity.HiFi3),
    "hifi3_bias_srcb": dict(MATH_FIDELITY=ttnn.MathFidelity.HiFi3, BIAS_SEED_SRCB=True),
    "bias_srcb": dict(BIAS_SEED_SRCB=True),
    "hifi3_dest2": dict(MATH_FIDELITY=ttnn.MathFidelity.HiFi3, DEST_GROUP_TILES=2),
    "hifi3_r1": dict(MATH_FIDELITY=ttnn.MathFidelity.HiFi3, FUSED_TILIZE=True),
    "hifi3_full_sync_bias_srcb": dict(MATH_FIDELITY=ttnn.MathFidelity.HiFi3, DST_FULL_SYNC=True, BIAS_SEED_SRCB=True),
}

SELECTED = [v for v in os.environ.get("NMC_SWEEP", "baseline").split(",") if v]
SHAPES = {f"{c['inputs'][0][0]}x{c['inputs'][0][2]}": c for c in LOOSE_CASES}


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("variant", SELECTED)
def test_perf_sweep(variant, shape, device, monkeypatch):
    for name, value in VARIANTS[variant].items():
        monkeypatch.setattr(op, name, value)
    monkeypatch.setattr(op, "_PLAN_CACHE", {})
    monkeypatch.setattr(op, "_CALL_CACHE", {})  # cached descriptors read the knobs when built
    case = SHAPES[shape]
    run_nomic_moe_combine(
        case["inputs"],
        dtype=case["dtype"],
        rows_buffer=case["rows_buffer"],
        output_layout=case["output_layout"],
        device=device,
        extras=case["extras"],
    )
