# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of rms_norm (Phase 0: bfloat16 input and weight, fp32 DEST, HiFi4).

Per shape: PCC, max / mean abs error, relative RMS (RMS error / std of the reference, as in
eval/metrics.py), and the got/true ratio spread (median and p5..p95 over finite elements whose reference
is not tiny). A tight ratio cluster around a constant other than 1.0 would be a scale bug, not rounding.

The gate is the bfloat16 `default` entry of the specification's NUMERIC: PCC >= 0.999 and relative
error (Frobenius) <= 1%. Results are written to precision_baseline.json beside this file (test output is not
shown under the simulator's workers) and recorded with record_property.
"""

import json
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.rms_norm import rms_norm

PCC_MIN = 0.999
REL_RMS_MAX = 0.01
RESULTS = Path(__file__).parent / "precision_baseline.json"

# (shape, epsilon, memory config): small, the Llama model call, a multi-NEO batch, a wide multi-row one,
# the widest row, and an L1 input.
CASES = [
    ((1, 1, 32, 128), 1e-6, "DRAM"),
    ((1, 1, 32, 2048), 1e-5, "DRAM"),
    ((2, 4, 128, 512), 1e-12, "DRAM"),
    ((1, 1, 128, 4096), 1e-5, "DRAM"),
    ((1, 1, 32, 8192), 1e-6, "DRAM"),
    ((1, 1, 8, 128), 1e-6, "L1"),
]


def reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xd = x.double()
    gamma = weight.double().reshape(-1)[: xd.shape[-1]]
    return xd * torch.rsqrt(xd.pow(2).mean(dim=-1, keepdim=True) + eps) * gamma


def _id(case):
    shape, eps, mem = case
    return "x".join(map(str, shape)) + f"_eps{eps:g}_{mem}"


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_precision_baseline(device, case, record_property):
    shape, eps, mem = case
    torch.manual_seed(1234)
    w = shape[-1]
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    g = torch.randn((1, 1, -(-w // 32), 32), dtype=torch.float32).to(torch.bfloat16)
    mem_cfg = ttnn.L1_MEMORY_CONFIG if mem == "L1" else ttnn.DRAM_MEMORY_CONFIG
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_cfg)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    got = ttnn.to_torch(rms_norm(tx, weight=tg, epsilon=eps)).double()
    expected = reference(x, g, eps)

    _, pcc = comp_pcc(expected, got, PCC_MIN)
    _, allclose_msg = comp_allclose(expected, got)
    diff = got - expected
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel_rms = diff.pow(2).mean().sqrt().item() / expected.std().item()

    # got/true ratio over elements whose reference is not tiny (|ref| above 1% of max|ref|).
    mask = torch.isfinite(got) & (expected.abs() > 0.01 * expected.abs().max())
    ratio = (got[mask] / expected[mask]).float()
    q = torch.quantile(ratio, torch.tensor([0.05, 0.5, 0.95]))
    ratio_p5, ratio_median, ratio_p95 = (float(v) for v in q)

    row = {
        "shape": list(shape),
        "epsilon": eps,
        "memory": mem,
        "pcc": float(pcc),
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "rel_rms": rel_rms,
        "ratio_median": ratio_median,
        "ratio_p5": ratio_p5,
        "ratio_p95": ratio_p95,
        "allclose": allclose_msg,
    }
    for key, value in row.items():
        record_property(key, value)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results[_id(case)] = row
    RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True))

    assert pcc >= PCC_MIN, row
    assert rel_rms <= REL_RMS_MAX, row
    # Scale-bug detector: the ratio must centre on 1 (a constant offset from 1 is structural, not rounding).
    assert abs(ratio_median - 1.0) < 0.01, row
