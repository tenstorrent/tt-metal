# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""onorm RECONFIG_MODE (Refinement 3, lever 1) — bit-neutrality + device-ns.

DO NOT DELETE.  Two jobs, in the order the refinement's "Done when" states them:

1. **Bit-neutrality.**  Turning the per-helper data-format reconfig off must not
   change a single output byte: every CB in this kernel carries one and the same
   format, so the reconfig calls are re-programming the registers with the value
   they already hold.  `test_reconfig_off_is_bit_identical_to_on` asserts
   `torch.equal` between the two settings — the same discipline Refinement 2 used
   for the cross-core exchange, and a strictly stronger claim than "PCC
   unchanged".  If reconfig-off ever perturbs a value, that is a BUG (a real
   format transition somewhere), not a precision trade.

2. **Measurement vehicle**, trial-major interleaved inside one process (single-shot
   numbers for onorm are not reproducible across processes — see
   `test_onorm_trials.py`'s module docstring).  Run under
   `scripts/run_safe_pytest.sh --profile --run-all` and take the MEDIAN of
   `DEVICE KERNEL DURATION [ns]` per arm; the row order is (trial, cell, arm).
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

# The config-spanning guard set from op_requirements.md (same cells as
# test_onorm_retile_guard.py, so the two files' numbers are comparable).
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

# Bit-identity is checked over the shape span AND at pinned group sizes, because the
# number of helper boundaries per block — every one of which the knob elides a
# reconfig at — is a function of the group size (both `norm_chunks` and
# `gate_chunks` scale with it).  At the shipped block factors one block crosses
# `norm_chunks*5 + 1 + 2*gate_chunks` boundaries: **113** at RETILE_GROUP_CORES=1
# (16 norm chunks, 16 gate chunks), 29 at G=8, and 8 at G=32 where both chunk
# counts bottom out at 1.  So these cells span a 14x range of elided calls.
IDENTITY_CELLS = [
    ("b1t32", 1, 32, "auto"),
    ("b1t128", 1, 128, "auto"),
    ("b1t640", 1, 640, "auto"),
    ("b1t640_g1", 1, 640, 1),
    ("b1t640_g8", 1, 640, 8),
    ("b2t64_g2", 2, 64, 2),
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
    saved = {k: getattr(pd, k) for k in ("RECONFIG_MODE", "RETILE_GROUP_CORES")}
    yield
    for k, v in saved.items():
        setattr(pd, k, v)


def _inputs(batch, tokens):
    torch.manual_seed(42)
    t_o = torch.randn(batch, tokens, HV, V, dtype=torch.bfloat16)
    t_gate = torch.randn(batch, tokens, FLAT, dtype=torch.bfloat16)
    t_w = torch.randn(1, 1, 1, V, dtype=torch.bfloat16)
    return t_o, t_gate, t_w


def _to_device(device, t_o, t_gate, t_w):
    return (
        ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(t_gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )


def _reference(t_o, t_gate, t_w, batch, tokens):
    f = t_o.to(torch.float32)
    ref = f * torch.rsqrt(f.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    ref = ref * t_w.to(torch.float32).reshape(1, 1, 1, V)
    return ref.reshape(batch, tokens, FLAT) * torch.sigmoid(t_gate.to(torch.float32))


def _run(device, batch, tokens, overrides=None):
    t_o, t_gate, t_w = _inputs(batch, tokens)
    o, gate, w = _to_device(device, t_o, t_gate, t_w)
    out = onorm(o, gate, w, compute_kernel_config=_config(overrides))
    got = ttnn.to_torch(out)
    assert_with_pcc(_reference(t_o, t_gate, t_w, batch, tokens), got.to(torch.float32), PCC)
    return got


@pytest.mark.parametrize("name, batch, tokens, group", IDENTITY_CELLS, ids=lambda v: str(v))
def test_reconfig_off_is_bit_identical_to_on(device, restore_knobs, name, batch, tokens, group):
    """RECONFIG_MODE='off' must be BIT-identical to 'on' — not merely close.

    Every CB is `o.dtype`, so each elided reconfig was re-writing the format
    register with the value it already held.  Any output difference at all means a
    boundary really does change format and the knob's precondition is violated.
    """
    pd.RETILE_GROUP_CORES = group

    pd.RECONFIG_MODE = "on"
    on = _run(device, batch, tokens)

    pd.RECONFIG_MODE = "off"
    off = _run(device, batch, tokens)

    assert torch.equal(on, off), (
        f"RECONFIG_MODE off != on at {name}: max |diff| = "
        f"{(on.to(torch.float32) - off.to(torch.float32)).abs().max().item()}"
    )


def test_reconfig_mode_rejects_unknown_value(device, restore_knobs, expect_error):
    """A typo in the knob must fail loudly at descriptor-build time."""
    pd.RECONFIG_MODE = "sometimes"
    with expect_error(AssertionError, "RECONFIG_MODE"):
        _run(device, 1, 32)


@pytest.mark.parametrize("trials", [5])
def test_reconfig_trial(device, restore_knobs, trials):
    """Paired on-vs-off device-ns over the whole config-spanning guard set."""
    for _ in range(trials):
        for _name, batch, tokens, overrides in GUARD_CELLS:
            pd.RECONFIG_MODE = "on"
            _run(device, batch, tokens, overrides)
            pd.RECONFIG_MODE = "off"
            _run(device, batch, tokens, overrides)
