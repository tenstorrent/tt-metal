# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""onorm Refinement 2 non-regression guard set — device-ns, paired in one process.

DO NOT DELETE.  This is the "no regression across the config-spanning guard set"
evidence for the cross-core re-tile, measured the only way onorm's numbers are
reproducible: **trial-major interleaved**, both arms in the SAME process, medians
over >= 5 trials (a 248 vs 102 us swing on identical config is on record for
candidate-major runs across processes).

Each cell is measured twice per trial:
  * `old` — `RETILE_GROUP_CORES = 1`, i.e. exactly what Refinement 1b shipped
    (one core per token-block, no exchange, byte-identical kernels);
  * `new` — `RETILE_GROUP_CORES` read from the module, i.e. what actually ships.
The `new` arm reads the module rather than restating a value, so this always
measures the shipping policy — a future retune cannot silently retitle a column.

Run under `scripts/run_safe_pytest.sh --profile` and read
`DEVICE KERNEL DURATION [ns]` + `CORE COUNT` per row; the row order is
(trial, cell, arm).
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

# The config-spanning guard set from op_requirements.md: the shape/occupancy span
# plus the config span.  `fp32on` keeps the public compute_kernel_config override
# (Refinement 1b's caller path) on the guard set as well.
_DEFAULT = None  # resolved per-call through default_compute_kernel_config()
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
    saved = {k: getattr(pd, k) for k in ("RETILE_GROUP_CORES",)}
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
    assert_with_pcc(ref, ttnn.to_torch(out).to(torch.float32), PCC)


@pytest.mark.parametrize("trials", [5])
def test_guard_set_trial(device, restore_knobs, trials):
    """Paired old-vs-new device-ns over the whole config-spanning guard set."""
    shipped = getattr(pd, "RETILE_GROUP_CORES")
    for _ in range(trials):
        for _name, batch, tokens, overrides in GUARD_CELLS:
            pd.RETILE_GROUP_CORES = 1  # the Refinement 1b baseline
            _run(device, batch, tokens, overrides)
            pd.RETILE_GROUP_CORES = shipped  # what actually ships
            _run(device, batch, tokens, overrides)
