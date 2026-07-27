# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""onorm Refinement 3 non-regression guard set — device-ns, paired in one process.

DO NOT DELETE.  Same shape as `test_onorm_retile_guard.py` (Refinement 2's guard),
one refinement later: it measures the **whole R3 delta** — the retuned compute
block surface (`NORM_CHUNK_TOKENS`, `GATE_CHUNK_TILES`, `GATE_DEST_TILES`), the
`RECONFIG_MODE` knob, and the recalibrated dispatch policy
(`EXCHANGE_COST_PER_BLOCK`) — as one paired old-vs-new comparison over the
config-spanning guard set.

  * `r2`  — everything R3 moved, pinned back to what Refinement 2 shipped (see
            R2_BLOCK_SURFACE below, including the policy).  Every other knob is
            left at the module value, so this arm isolates R3 exactly.
  * `r3`  — every knob read from the module, i.e. exactly what ships.  The arm
            restates NOTHING, so a future retune cannot silently mislabel a column.

Trial-major interleaved inside one process, medians over >= 5 trials — the only
reproducible protocol for this op (see `test_onorm_trials.py`).

    scripts/run_safe_pytest.sh --profile tests/.../test_onorm_r3_guard.py
"""

import pytest
import torch

import ttnn
import ttnn.operations.onorm.onorm_program_descriptor as pd
from ttnn.operations.onorm import default_compute_kernel_config, onorm

from tests.ttnn.utils_for_testing import assert_with_pcc

HV, V = 32, 128
FLAT = HV * V
PCC = 0.995

# The knobs Refinement 3 moved, at their pre-R3 (Refinement 2) values.
#
# `EXCHANGE_COST_PER_BLOCK = 0` is exactly Refinement 2's dispatch objective: with
# no exchange term the cost collapses to `blocks_per_group / g`, R2's `work(g)`,
# and the strict-`<` scan resolves ties to the smaller group just as R2's explicit
# tie-break did.  So this arm reproduces R2's group PICK as well as its block
# factors (verified: it picks 32 / 16 / 32 / 2 on the four shapes, which is what
# R2's changelog recorded) — i.e. the comparison below is the WHOLE R3 delta, not
# just the part of it that shows at a fixed group size.
R2_BLOCK_SURFACE = {
    "NORM_CHUNK_TOKENS": 8,
    "GATE_CHUNK_TILES": 64,
    "GATE_DEST_TILES": 4,
    "RECONFIG_MODE": "on",
    "EXCHANGE_COST_PER_BLOCK": 0.0,
}

# The shipping values, captured at IMPORT time so a test that mutates the module
# cannot corrupt the reference (the `restore_knobs` fixture restores per test, but
# this makes the "what ships" arm independent of fixture ordering).
_SHIPPED = {k: getattr(pd, k) for k in R2_BLOCK_SURFACE}

_FP32_ON = dict(fp32_dest_acc_en=True)
_APPROX = dict(math_approx_mode=True)
_LOFI = dict(math_fidelity=ttnn.MathFidelity.LoFi)

GUARD_CELLS = [
    ("b1t32_default", 1, 32, None),
    ("b1t64_default", 1, 64, None),
    ("b1t128_default", 1, 128, None),
    ("b1t640_default", 1, 640, None),
    ("b8t640_default", 8, 640, None),
    ("b1t640_approx", 1, 640, _APPROX),
    ("b1t640_lofi", 1, 640, _LOFI),
    ("b1t640_fp32on", 1, 640, _FP32_ON),
]


def _config(overrides):
    if overrides is None:
        return default_compute_kernel_config()
    base = dict(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    base.update(overrides)
    return ttnn.WormholeComputeKernelConfig(**base)


@pytest.fixture
def restore_knobs():
    saved = {k: getattr(pd, k) for k in R2_BLOCK_SURFACE}
    yield
    for k, v in saved.items():
        setattr(pd, k, v)


def _run(device, batch, tokens, overrides):
    torch.manual_seed(42)
    t_o = torch.randn(batch, tokens, HV, V, dtype=torch.bfloat16)
    t_gate = torch.randn(batch, tokens, FLAT, dtype=torch.bfloat16)
    t_w = torch.randn(1, 1, 1, V, dtype=torch.bfloat16)

    o = ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gate = ttnn.from_torch(t_gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = onorm(o, gate, w, compute_kernel_config=_config(overrides))

    f = t_o.to(torch.float32)
    ref = f * torch.rsqrt(f.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    ref = ref * t_w.to(torch.float32).reshape(1, 1, 1, V)
    ref = ref.reshape(batch, tokens, FLAT) * torch.sigmoid(t_gate.to(torch.float32))
    got = ttnn.to_torch(out)
    assert_with_pcc(ref, got.to(torch.float32), PCC)
    return got


@pytest.mark.parametrize("name, batch, tokens", [(n, b, t) for n, b, t, o in GUARD_CELLS if o is None], ids=str)
def test_r3_surface_is_bit_identical_to_r2(device, restore_knobs, name, batch, tokens):
    """Refinement 3 must be a pure PERF change: not one output byte may move.

    Every lever in it is numerically inert by construction, and this asserts the
    conjunction of those arguments end-to-end rather than trusting them:
      * the two block factors only change how many tokens / output tiles one helper
        invocation covers, and NO reduction crosses a chunk boundary (P1 accumulates
        o^2 within one token, P2 reduces within one token), so the arithmetic per
        element is identical;
      * the dispatch-policy recalibration only changes WHICH core owns a token or an
        output column — `test_onorm_retile_group.py` already pins that to exact
        equality at every group size;
      * `RECONFIG_MODE` re-programs a register with the value it already holds —
        `test_onorm_reconfig.py` pins that to exact equality too.
    A difference here means one of those three claims is false.
    """
    for k, v in R2_BLOCK_SURFACE.items():
        setattr(pd, k, v)
    r2 = _run(device, batch, tokens, None)

    for k, v in _SHIPPED.items():  # what actually ships
        setattr(pd, k, v)
    r3 = _run(device, batch, tokens, None)

    assert torch.equal(r2, r3), (
        f"R3 perturbed the numerics at {name}: max |diff| = "
        f"{(r2.to(torch.float32) - r3.to(torch.float32)).abs().max().item()}"
    )


@pytest.mark.parametrize("trials", [5])
def test_r3_guard_set_trial(device, restore_knobs, trials):
    """Paired R2-vs-R3 device-ns over the whole config-spanning guard set."""
    for _ in range(trials):
        for _name, batch, tokens, overrides in GUARD_CELLS:
            for k, v in R2_BLOCK_SURFACE.items():  # the pre-R3 arm
                setattr(pd, k, v)
            _run(device, batch, tokens, overrides)
            for k, v in _SHIPPED.items():  # what actually ships
                setattr(pd, k, v)
            _run(device, batch, tokens, overrides)
