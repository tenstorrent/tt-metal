# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 1 focus bench: PERF FOCUS 1 at its exact golden config (Qwen3.8 prefill gate_up + SwiGLU,
1x2048x5120 bf16 @ 5120x8704 bf4b, LoFi, fp32 DEST, math_approx_mode=False, DRAM interleaved, bf16 out).

Skipped unless GENIFY_MATMUL_PERF=1. Profile with
    GENIFY_MATMUL_PERF=1 scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/genify_matmul/perf/test_genify_matmul_perf_focus.py
DEVICE KERNEL DURATION [ns] of the genify_matmul program is the number; per-stage zones land in
generated/profiler/.logs/profile_log_device.csv (MaybeDeviceZoneScope in the kernels).
`check=1` (GENIFY_MATMUL_CHECK=1) also checks PCC against torch (off for ablated runs).
"""
import math
import os
import pytest
import torch
import ttnn
from eval.golden_tests.genify_matmul.axes import observed as genify_matmul
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(os.environ.get("GENIFY_MATMUL_PERF") != "1", reason="perf bench: set GENIFY_MATMUL_PERF=1"),
]


def test_perf_focus1(device):
    m, k, n = 2048, 5120, 8704
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
    )
    out = genify_matmul(ta, tb, compute_kernel_config=cfg, activation="swiglu")
    ttnn.synchronize_device(device)
    if os.environ.get("GENIFY_MATMUL_CHECK") == "1":
        Bq = ttnn.to_torch(tb).float()
        Aq = ttnn.to_torch(ta).float()
        y = Aq @ Bq
        g = y.reshape(1, m, n // 64, 2, 32)
        ref = (torch.nn.functional.silu(g[..., 0, :]) * g[..., 1, :]).reshape(1, m, n // 2)
        got = ttnn.to_torch(out).float()
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print(f"FOCUS_PCC {pcc:.6f} rel_rms {rms:.5f}")
        assert pcc > 0.9995
