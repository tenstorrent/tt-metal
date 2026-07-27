# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""onorm performance harness — on-device kernel time per configuration.

DO NOT DELETE.  This is the measurement vehicle for the blocking-model knobs:
run it under `scripts/run_safe_pytest.sh --profile` and read
`DEVICE KERNEL DURATION [ns]` per row of the emitted CSV.  Each test is also a
correctness check (PCC), so a knob turn that breaks the op fails here too.

The shape is the design's bringup profiling length (B=1, T=640 => 20 token
blocks => 20 cores).  `test_perf_grid_fill` uses B=8 so all 110 cores are busy.
"""

import pytest
import torch

import ttnn
from ttnn.operations.onorm import default_compute_kernel_config, onorm

from tests.ttnn.utils_for_testing import assert_with_pcc

HV = 32
V = 128
FLAT = HV * V
PCC = 0.995


def _run(device, batch, tokens, cfg):
    torch.manual_seed(42)
    t_o = torch.randn(batch, tokens, HV, V, dtype=torch.bfloat16)
    t_gate = torch.randn(batch, tokens, FLAT, dtype=torch.bfloat16)
    t_w = torch.randn(1, 1, 1, V, dtype=torch.bfloat16)

    o = ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gate = ttnn.from_torch(t_gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = onorm(o, gate, w, compute_kernel_config=cfg)

    eps = 1e-5
    f = t_o.to(torch.float32)
    ms = f.pow(2).mean(dim=-1, keepdim=True)
    ref = f * torch.rsqrt(ms + eps)
    ref = ref * t_w.to(torch.float32).reshape(1, 1, 1, V)
    ref = ref.reshape(batch, tokens, FLAT) * torch.sigmoid(t_gate.to(torch.float32))

    assert_with_pcc(ref, ttnn.to_torch(out).to(torch.float32), PCC)


def test_perf_default(device):
    """Baseline: the exported default compute config."""
    _run(device, 1, 640, default_compute_kernel_config())


def test_perf_math_approx(device):
    """Same shape with math_approx_mode=True (fast SFPU sigmoid / rsqrt)."""
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    _run(device, 1, 640, cfg)


def test_perf_lofi(device):
    """Same shape at LoFi — isolates FPU math-fidelity cost from SFPU cost."""
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    _run(device, 1, 640, cfg)


def test_perf_grid_fill(device):
    """B=8, T=640 => 160 token-blocks over the whole 110-core grid."""
    _run(device, 8, 640, default_compute_kernel_config())


def test_perf_no_fp32_dest_acc(device):
    """fp32_dest_acc_en=False.

    DIAGNOSTIC, not a candidate default: the design requires fp32 DEST
    accumulation for P1's sum-of-squares.  But `DST_ACCUM_MODE` is also threaded
    into every SFPU call (`calculate_sigmoid(fast_and_approx, DST_ACCUM_MODE,
    ITERATIONS)`), so this isolates how much of P7b's cost is the fp32-DEST SFPU
    penalty rather than sigmoid itself.
    """
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    _run(device, 1, 640, cfg)
