# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-measurement harness for rms_norm Refinement 6 (decode + sharded profiles).

Runs the DECODE interleaved perf shapes (`_perf_case(32, W, …)`) and the
WIDTH/BLOCK-sharded `_perf_case` geometries from
eval/golden_tests/rms_norm/feature_spec.py at their EXACT config
(bf16 / fp32_dest_acc_en=False / TILE input / TILE gamma / HiFi2) so the
device-kernel duration can be read off the Tracy profiler CSV
(`scripts/run_safe_pytest.sh --profile ...`, column = DEVICE KERNEL DURATION [ns]).

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

from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm

N_TRIALS = 8
_ML = ttnn.TensorMemoryLayout

# (rows, W, achievable_ns) — the DECODE interleaved perf cases (feature_spec _perf_case).
DECODE_CASES = [
    (32, 1024, 9149),
    (32, 2304, 17003),
    (32, 5120, 75825),
    (32, 7168, 104259),
]

# (rows, W, achievable_ns, memory_layout, (shard_h, shard_w), (grid_x, grid_y)) — the
# WIDTH/BLOCK-sharded perf geometries (feature_spec _perf_case, sharded block).
SHARDED_CASES = [
    (32, 1024, 4110, _ML.WIDTH_SHARDED, (32, 128), (8, 1)),
    (32, 2304, 4617, _ML.WIDTH_SHARDED, (32, 256), (9, 1)),
    (32, 5120, 5267, _ML.WIDTH_SHARDED, (32, 160), (8, 4)),
    (32, 7168, 5481, _ML.WIDTH_SHARDED, (32, 256), (7, 4)),
    (8192, 1024, 25640, _ML.BLOCK_SHARDED, (1024, 128), (8, 8)),
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


def _perf_cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _reference(torch_input, torch_gamma):
    expected = torch_input.to(torch.float32)
    expected = expected * torch.rsqrt(expected.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return expected * torch_gamma.to(torch.float32).reshape(-1)


@pytest.mark.parametrize("rows,W,achievable_ns", DECODE_CASES, ids=[f"decode_{r}x{w}" for (r, w, _n) in DECODE_CASES])
def test_decode_perf(rows, W, achievable_ns, device):
    torch.manual_seed(0)
    shape = (1, 1, rows, W)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(W, dtype=torch.bfloat16)
    expected = _reference(torch_input, torch_gamma)

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

    cfg = _perf_cfg()
    out = None
    for _ in range(N_TRIALS):
        out = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=1e-6, compute_kernel_config=cfg)

    result = ttnn.to_torch(out)
    pcc = _pcc(result, expected)
    print(f"\nPERF_DECODE ({rows},{W}) achievable_ns={achievable_ns} PCC={pcc:.6f}")
    assert pcc >= 0.9995, f"soft PCC gate: {pcc} < 0.9995 for shape {shape}"


@pytest.mark.parametrize(
    "rows,W,achievable_ns,ml,shard,grid",
    SHARDED_CASES,
    ids=[f"{ml.name.split('_')[0].lower()}_{r}x{w}" for (r, w, _n, ml, _s, _g) in SHARDED_CASES],
)
def test_sharded_perf(rows, W, achievable_ns, ml, shard, grid, device):
    torch.manual_seed(0)
    shape = (1, 1, rows, W)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(W, dtype=torch.bfloat16)
    expected = _reference(torch_input, torch_gamma)

    in_cfg = shard_config(
        list(shard),
        grid,
        ml,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_cfg,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    cfg = _perf_cfg()
    out = None
    for _ in range(N_TRIALS):
        out = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=1e-6, compute_kernel_config=cfg, memory_config=in_cfg)

    result = ttnn.to_torch(out)
    pcc = _pcc(result, expected)
    print(f"\nPERF_SHARDED ({rows},{W}) {ml.name} achievable_ns={achievable_ns} PCC={pcc:.6f}")
    assert pcc >= 0.9995, f"soft PCC gate: {pcc} < 0.9995 for shape {shape}"
