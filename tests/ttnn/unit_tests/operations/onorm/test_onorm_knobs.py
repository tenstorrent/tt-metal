# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""onorm Blocking-Model knob sweep — proves every knob is a LIVE tunable.

DO NOT DELETE.  Two jobs:

1. **Correctness under a knob turn.** Each knob in
   `onorm_program_descriptor` is set to a non-default value and the op must
   still be numerically correct.  This is what stops a knob from silently
   decaying into a decorative constant that only works at its phase-1 value:
   if someone hardcodes a block factor, or derives a CB page count / loop bound
   from a whole-op dimension instead of from the knob, one of these fails.

2. **Measurement vehicle.** Run under `scripts/run_safe_pytest.sh --profile`
   and read `DEVICE KERNEL DURATION [ns]` per row to compare knob settings.

The knobs are module-level constants read at descriptor-build time, so setting
them on the module is exactly how a future refinement would retune the op.
"""

import pytest
import torch

import ttnn
import ttnn.operations.onorm.onorm_program_descriptor as pd
from ttnn.operations.onorm import default_compute_kernel_config, onorm

from tests.ttnn.utils_for_testing import assert_with_pcc

HV = 32
V = 128
FLAT = HV * V
PCC = 0.995

# (name, value) pairs. Every one must divide/relate correctly per the host
# asserts in the descriptor; that is part of what is under test.
KNOB_SETTINGS = [
    # --- per-core normalize sub-block: 2 (default) -> 1 (floor), 4/8/16 (coarser) ---
    ("NORM_CHUNK_TOKENS", 1),
    ("NORM_CHUNK_TOKENS", 4),
    ("NORM_CHUNK_TOKENS", 8),
    ("NORM_CHUNK_TOKENS", 16),
    # --- gate-chain block factor: 16 (default) -> 4 (floor) and coarser ---
    ("GATE_CHUNK_TILES", 4),
    ("GATE_CHUNK_TILES", 32),
    ("GATE_CHUNK_TILES", 64),
    ("GATE_CHUNK_TILES", 128),  # the whole flat tile-row in one invocation
    # --- per-DEST-window tiles in the two gate phases (R1's knob) ---
    ("GATE_DEST_TILES", 1),
    ("GATE_DEST_TILES", 8),
    # --- data-format reconfig at every helper boundary (R3 lever 1) ---
    # Both settings must be numerically correct; `test_onorm_reconfig.py` is what
    # additionally proves they are BIT-identical to each other.
    ("RECONFIG_MODE", "on"),
    ("RECONFIG_MODE", "off"),
    # --- the `auto` dispatch policy's exchange term (R3) ---
    # 0.0 is exactly Refinement 2's objective; 0.5 is the top of the calibrated
    # window (above it a saturated shape falls back to group 1).  Both must be
    # correct, since both are legal policy calibrations.
    ("EXCHANGE_COST_PER_BLOCK", 0.0),
    ("EXCHANGE_COST_PER_BLOCK", 0.5),
    # --- NoC group size (reader AND writer, one knob for both halves) ---
    ("DM_BLOCK_TILES", 1),  # the documented latency-bound trap
    ("DM_BLOCK_TILES", 2),
    ("DM_BLOCK_TILES", 8),
    # --- streaming depths ---
    ("DM_DEPTH", 4),
    ("O_DEPTH", 3),
    # --- cross-core re-tile group size (Refinement 2): 1 (no exchange) and the
    #     whole legal power-of-two range up to one token / 4 columns per core ---
    ("RETILE_GROUP_CORES", 1),
    ("RETILE_GROUP_CORES", 2),
    ("RETILE_GROUP_CORES", 8),
    ("RETILE_GROUP_CORES", 32),
    ("MAX_RETILE_GROUP_CORES", 4),  # caps the "auto" policy
    ("RM_LOCAL_DEPTH", 3),
]

# Multi-knob settings. TOKENS_PER_BLOCK is the design's headline knob-turn, but at
# 64 tokens the two re-tile buffers double to 256 pages each, so it only fits L1
# once NORM_CHUNK_TOKENS comes down — exactly the trade the budget assert's
# message prescribes. This proves the knob is genuinely reachable, not decorative.
COMBOS = [
    # TOKENS_PER_BLOCK=64 doubles both re-tile buffers to 256 pages each, so at
    # RETILE_GROUP_CORES=1 it only fits once NORM_CHUNK_TOKENS and GATE_CHUNK_TILES
    # come down — exactly the trade the budget assert's message prescribes.
    {"RETILE_GROUP_CORES": 1, "TOKENS_PER_BLOCK": 64, "NORM_CHUNK_TOKENS": 4, "GATE_CHUNK_TILES": 32},
    # Same unlock, paying for it on the DM side instead (narrower streaming CBs).
    {
        "RETILE_GROUP_CORES": 1,
        "TOKENS_PER_BLOCK": 64,
        "NORM_CHUNK_TOKENS": 8,
        "GATE_CHUNK_TILES": 32,
        "DM_BLOCK_TILES": 4,
        "DM_DEPTH": 2,
    },
    # Coarsest normalize pass + whole-row gate chain, funded by narrower DM buffers.
    {"RETILE_GROUP_CORES": 1, "NORM_CHUNK_TOKENS": 16, "GATE_CHUNK_TILES": 128, "DM_BLOCK_TILES": 4, "DM_DEPTH": 2},
    # ...and the third way to fund a coarse TOKENS_PER_BLOCK, which Refinement 2
    # added: splitting the block across cores divides BOTH re-tile buffers by the
    # group size, so the same setting that needs two knobs lowered at group 1 fits
    # untouched at group 4.
    {"RETILE_GROUP_CORES": 4, "TOKENS_PER_BLOCK": 64, "NORM_CHUNK_TOKENS": 16, "GATE_CHUNK_TILES": 32},
    # The whole pre-Refinement-3 block surface at once (what R2 shipped), which is
    # also the `r2` arm of test_onorm_r3_guard.py — it must stay CORRECT, not just
    # slower, so the guard set is comparing two working configurations.
    {
        "NORM_CHUNK_TOKENS": 8,
        "GATE_CHUNK_TILES": 64,
        "GATE_DEST_TILES": 4,
        "RECONFIG_MODE": "on",
        "EXCHANGE_COST_PER_BLOCK": 0.0,
    },
]

# Knob settings that legitimately do NOT fit L1. The host budget assert must
# reject them with its actionable message rather than letting the runtime throw a
# bare "beyond max L1 size".
#
# These pin RETILE_GROUP_CORES=1 because that is where the L1 frontier IS: the
# two re-tile buffers are the dominant term and the cross-core split divides them
# by the group size, so an "over budget" setting at group 1 is comfortably inside
# the budget at group 4. Refinement 2 therefore MOVED this frontier rather than
# invalidating it, and the assert has to be tested where it still binds.
OVER_BUDGET = [
    {"RETILE_GROUP_CORES": 1, "NORM_CHUNK_TOKENS": 32},
    {"RETILE_GROUP_CORES": 1, "TOKENS_PER_BLOCK": 64, "NORM_CHUNK_TOKENS": 16},
]


@pytest.fixture
def restore_knobs():
    saved = {
        k: getattr(pd, k)
        for k in (
            "TOKENS_PER_BLOCK",
            "NORM_CHUNK_TOKENS",
            "GATE_CHUNK_TILES",
            "GATE_DEST_TILES",
            "RECONFIG_MODE",
            "EXCHANGE_COST_PER_BLOCK",
            "DM_BLOCK_TILES",
            "DM_DEPTH",
            "O_DEPTH",
            "RETILE_GROUP_CORES",
            "MAX_RETILE_GROUP_CORES",
            "RM_LOCAL_DEPTH",
        )
    }
    yield
    for k, v in saved.items():
        setattr(pd, k, v)


def _run(device, batch, tokens):
    torch.manual_seed(42)
    t_o = torch.randn(batch, tokens, HV, V, dtype=torch.bfloat16)
    t_gate = torch.randn(batch, tokens, FLAT, dtype=torch.bfloat16)
    t_w = torch.randn(1, 1, 1, V, dtype=torch.bfloat16)

    o = ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gate = ttnn.from_torch(t_gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = onorm(o, gate, w, compute_kernel_config=default_compute_kernel_config())

    eps = 1e-5
    f = t_o.to(torch.float32)
    ref = f * torch.rsqrt(f.pow(2).mean(dim=-1, keepdim=True) + eps)
    ref = ref * t_w.to(torch.float32).reshape(1, 1, 1, V)
    ref = ref.reshape(batch, tokens, FLAT) * torch.sigmoid(t_gate.to(torch.float32))

    assert list(out.shape) == [batch, tokens, FLAT]
    assert_with_pcc(ref, ttnn.to_torch(out).to(torch.float32), PCC)


@pytest.mark.parametrize("knob, value", KNOB_SETTINGS, ids=lambda v: str(v))
def test_knob_turn(device, restore_knobs, knob, value):
    """The op must stay correct at a non-default value of every knob."""
    setattr(pd, knob, value)
    # T=640 exercises multi-core; TOKENS_PER_BLOCK=64 also needs T % 64 == 0.
    _run(device, 1, 640)


@pytest.mark.parametrize("combo", COMBOS, ids=lambda c: "-".join(f"{k}{v}" for k, v in c.items()))
def test_knob_combo(device, restore_knobs, combo):
    """Coarser block factors that need two knobs moved together."""
    for k, v in combo.items():
        setattr(pd, k, v)
    _run(device, 1, 640)


@pytest.mark.parametrize("combo", OVER_BUDGET, ids=lambda c: "-".join(f"{k}{v}" for k, v in c.items()))
def test_knob_over_budget_is_rejected_with_guidance(device, restore_knobs, combo):
    """An out-of-L1 knob setting fails the host guard, naming what to lower."""
    for k, v in combo.items():
        setattr(pd, k, v)
    try:
        _run(device, 1, 640)
    except AssertionError as exc:
        msg = str(exc)
        assert "exceeds the CB-available L1" in msg, msg
        # The message must name the knobs to lower, in the design's order.
        assert "GATE_CHUNK_TILES" in msg and "NORM_CHUNK_TOKENS" in msg, msg
        return
    pytest.fail("expected the host L1 budget assert to reject this knob setting")
