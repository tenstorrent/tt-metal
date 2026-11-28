# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4 — flagged TILE profiles (Q2185 bf16->fp32 DRAM, P39 bf16->bf8b L1) plus guard shapes.

One device dispatch per case, so `run_safe_pytest.sh --profile` yields one device-kernel row per case. The
`knobs` parametrization sweeps BLOCK_TILES_CAP x BUFFER_DEPTH (None = the shipped default) and doubles as a
correctness pin for multi-block / deep-CB TILE pipelines on the flagged shapes.
"""

import importlib
import os

import pytest
import torch
import ttnn

from ttnn.operations.g6_typecast import g6_typecast

from .test_g6_typecast import check_bfp8_group

op_mod = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast")
pd = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast_program_descriptor")

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG

CASES = {
    "Q2185": ((16, 4096, 12), ttnn.float32, DRAM),
    "P39": ((1, 128, 256, 256), ttnn.bfloat8_b, L1),
    "Q2187": ((1, 1, 3844, 256), ttnn.bfloat8_b, DRAM),
    "i32_dram": ((8, 1024, 256), ttnn.int32, DRAM),
    "f32_l1": ((4, 512, 512), ttnn.float32, L1),
    "Q2191": ((16, 1, 12), ttnn.float32, DRAM),
    # Perf 1 live-face transfers: both tails, W tail > 16 with the bf8b column mask, bf8b H tail, rank 2.
    "tails_bf8b": ((2, 33, 40), ttnn.bfloat8_b, DRAM),
    "wtail20_bf8b": ((3, 70, 52), ttnn.bfloat8_b, DRAM),
    "wtail5_bf8b_l1": ((2, 45, 37), ttnn.bfloat8_b, L1),
    "tails_f32": ((3, 47, 100), ttnn.float32, DRAM),
    "rank2_i32": ((65, 12), ttnn.int32, DRAM),
}

# Knob sweep from the environment (perf work): G6_SWEEP="cap:depth,..." ("-" = default). The default list pins
# the shipped knobs plus forced multi-block pipelines (cap 3 -> ragged tail, depth 3; cap 1, depth 2) so the
# per-tile face walk (Perf 1 live-face transfers) crosses block boundaries. READ_INFLIGHT was removed in Perf 1.
_sweep = os.environ.get("G6_SWEEP")
_cases = os.environ.get("G6_CASES")
if _cases:
    CASES = {k: v for k, v in CASES.items() if k in _cases.split(",")}
KNOBS = (
    [tuple(None if v == "-" else int(v) for v in s.split(":")) for s in _sweep.split(",")]
    if _sweep
    else [(None, None), (3, 3), (1, 2)]
)


def _torch_ref(x, dst):
    if dst == ttnn.int32:
        return x.to(torch.int32)
    return x.float()


@pytest.mark.parametrize("knobs", KNOBS, ids=lambda k: f"cap{k[0]}_d{k[1]}")
@pytest.mark.parametrize("case", list(CASES))
def test_g6_typecast_tile_perf(device, monkeypatch, case, knobs):
    shape, dst, mem = CASES[case]
    cap, depth = knobs
    if cap is not None:
        monkeypatch.setattr(pd, "BLOCK_TILES_CAP", cap)
    if depth is not None:
        monkeypatch.setattr(pd, "BUFFER_DEPTH", depth)
    monkeypatch.setattr(op_mod, "_PROGRAM_CACHE", {})
    torch.manual_seed(4)
    x = (torch.randn(shape) * 50).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    out = ttnn.to_torch(g6_typecast(tt_x, dst))
    if os.environ.get("G6_NOCHECK"):  # ablation profiling only (stubbed payloads)
        return
    if dst == ttnn.bfloat8_b:
        check_bfp8_group(out, x.float())
    else:
        assert torch.equal(out, _torch_ref(x, dst)), "mismatch"
    print("PLAN", case, knobs, op_mod.last_plan)
