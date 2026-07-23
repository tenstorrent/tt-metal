# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-measurement harness for rms_norm (Refinement 3).

Runs the interleaved PREFILL perf shapes from
eval/golden_tests/rms_norm/feature_spec.py at their EXACT config
(bf16 / fp32_dest_acc_en=False / TILE input / TILE gamma / INTERLEAVED / HiFi2)
so the device-kernel duration can be read off the Tracy profiler CSV
(`scripts/run_safe_pytest.sh --profile ...`, column 19 = DEVICE KERNEL DURATION).

Each shape loops the op N_TRIALS times so the CSV carries N rows/shape; take the
median of all but the first (warm-up) row per /perf-measure.

Correctness: soft PCC gate 0.9995 (the perf cases' `pcc_threshold`). This file is
NOT the golden suite — it exists only to produce device-ns numbers and confirm the
soft PCC still holds.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ttnn.operations.rms_norm import rms_norm

N_TRIALS = 6

# (rows, W, achievable_ns) — the interleaved prefill perf cases (feature_spec _perf_case).
PREFILL_CASES = [
    (8192, 1024, 96744),
    (8192, 2304, 211345),
    (8192, 5120, 738307),
    (8192, 7168, 1032281),
]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 1.0
    return (torch.dot(a, b).item()) / denom


@pytest.mark.parametrize("rows,W,achievable_ns", PREFILL_CASES, ids=[f"{r}x{w}" for (r, w, _n) in PREFILL_CASES])
def test_prefill_perf(rows, W, achievable_ns, device):
    torch.manual_seed(0)
    shape = (1, 1, rows, W)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(W, dtype=torch.bfloat16)

    expected = torch_input.to(torch.float32)
    expected = expected * torch.rsqrt(expected.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = expected * torch_gamma.to(torch.float32).reshape(-1)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False

    out = None
    for _ in range(N_TRIALS):
        out = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=1e-6, compute_kernel_config=cfg)

    result = ttnn.to_torch(out)
    pcc = _pcc(result, expected)
    print(f"\nPERF_SHAPE ({rows},{W}) achievable_ns={achievable_ns} PCC={pcc:.6f}")
    assert pcc >= 0.9995, f"soft PCC gate: {pcc} < 0.9995 for shape {shape}"
