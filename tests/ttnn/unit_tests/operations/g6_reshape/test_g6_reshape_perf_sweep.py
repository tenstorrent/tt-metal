# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 knob co-tune harness: one call per (case, knob setting), bit-exact checked.
Run under `scripts/run_safe_pytest.sh --profile --no-py-zones` to read device-ns per setting from the
ops CSV (one generic_op row per test, in parametrize order). G6_PERF_SWEEP=1 enables the wide sweep;
by default only the shipped knobs run (cheap correctness pin for the perf guard set)."""

import importlib
import os

import pytest
import torch

import ttnn
from ttnn.operations.g6_reshape import g6_reshape

_op = importlib.import_module("ttnn.operations.g6_reshape.g6_reshape")

CASES = [
    ((1, 4096), (4096, 1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
    ((1, 3844), (3844, 1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
    ((16, 1, 5120), (1, 1, 16, 5120), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ((1, 1, 16, 4608), (16, 1, 4608), ttnn.bfloat16, ttnn.TILE_LAYOUT),
]

# "warmup" absorbs the first-call-in-module penalty (~200-300 ns observed); read "default" as the baseline.
SWEEP = {"warmup": {}, "default": {}}
CASES = CASES[:1] if os.environ.get("G6_FLAG_ONLY") else CASES
if os.environ.get("G6_PERF_SWEEP"):
    for c in (8, 16, 32, 64):
        SWEEP[f"cores{c}"] = {"MAX_CORES": c}
    SWEEP["movers1"] = {"MOVERS_PER_CORE": 1}
    SWEEP["cores32_movers1"] = {"MAX_CORES": 32, "MOVERS_PER_CORE": 1}
if os.environ.get("G6_PERF_SWEEP2"):
    SWEEP["noc_swapped"] = {"MOVER_NOCS": (ttnn.NOC.NOC_1, ttnn.NOC.NOC_0)}
    SWEEP["default_again"] = {}
    SWEEP["blk10_d2"] = {"MAX_BLOCK_ATOMS": 10, "STAGE_DEPTH": 2, "IMAGE_DEPTH": 2}
    SWEEP["blk10_d1"] = {"MAX_BLOCK_ATOMS": 10}
    SWEEP["bank_fold"] = {"RM_FOLD": "bank"}


def _id(case):
    return f"{'x'.join(map(str, case[0]))}-{'x'.join(map(str, case[1]))}-{case[2]}"


@pytest.mark.parametrize("knobs", SWEEP.values(), ids=SWEEP.keys())
@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_perf_sweep(case, knobs, device, monkeypatch):
    for name, value in knobs.items():
        monkeypatch.setattr(_op, name, value)
    src, dst, dtype, layout = case
    if dtype == ttnn.int32:
        x = torch.randint(-(2**31), 2**31 - 1, src, dtype=torch.int32)
    else:
        x = torch.randn(src).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    y = ttnn.to_torch(g6_reshape(t, list(dst)))
    if os.environ.get("G6_ABLATION"):  # stubbed-kernel ablation runs: timing only, output is garbage
        return
    assert torch.equal(y.reshape(dst), x.reshape(dst))
