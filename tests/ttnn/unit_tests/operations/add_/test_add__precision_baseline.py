# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of add_ on its supported cells (Phase 0: bfloat16, TILE, height-sharded L1, RELU,
fp32_dest_acc_en=True through the default compute config).

Measures, per shape: PCC (comp_pcc), max abs error (comp_allclose), mean abs error, relative RMS error (RMS
error / std of the reference, as eval/metrics.py check_output), the ULP distance at bfloat16 granularity
(max and p99), and the got/true ratio spread over the finite, non-zero reference elements (the scale-bug
detector: a tight cluster around a constant != 1 is a structural bug, not rounding).

The reference is the specification's: float64 sum of the operands as stored (bfloat16), RELU applied. The
specification grades bfloat16 at 1 ulp; a single rounding of the exact sum is at most 0.5 ulp, so the
expectation here is max ulp <= 1.

Output under pytest-xdist / --forked is not shown, so each case records its figures with record_property and
appends one JSON line to $ADD__PRECISION_OUT (default generated/add_/precision_baseline.jsonl).
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch

import ttnn
from eval.sharding import auto_shard_config
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.add_ import add_

RELU = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]
OUT = Path(os.environ.get("ADD__PRECISION_OUT", "generated/add_/precision_baseline.jsonl"))

SHAPES = [
    (1, 1, 32, 64),  # small: one row of tiles, one core
    (2, 4, 47, 1024),  # medium: H non-aligned, leading dims
    (1, 1, 784, 512),  # ResNet-50 layer 2 (H non-aligned)
    (1, 1, 3136, 256),  # larger: ResNet-50 layer 1
]


def _bf16_ulp_distance(actual, expected):
    """Per-element ULP distance on the bfloat16 grid (both quantized to bfloat16 first)."""

    def ordered(t):
        bits = t.to(torch.bfloat16).view(torch.int16).to(torch.int64)
        sign = bits < 0
        mag = torch.where(sign, bits + (1 << 15), bits)
        return torch.where(sign, -mag, mag)

    return (ordered(actual) - ordered(expected)).abs().float()


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_precision_baseline(device, shape, record_property):
    torch.manual_seed(42)
    ta = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)
    tb = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)
    config = auto_shard_config(
        shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    put = lambda t: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config
    )
    a, b = put(ta), put(tb)

    result = add_(a, b, activations=RELU)
    assert result is a

    got = ttnn.to_torch(result).to(torch.float64)
    expected = torch.relu(ta.double() + tb.double())

    _, pcc = comp_pcc(expected, got)
    _, allclose = comp_allclose(expected, got)
    abs_err = (got - expected).abs()
    rel_rms = math.sqrt(((got - expected) ** 2).mean().item()) / expected.std().item()
    ulp = _bf16_ulp_distance(got, expected)
    mask = torch.isfinite(expected) & (expected != 0)
    ratio = (got[mask] / expected[mask]).float()
    figures = {
        "shape": list(shape),
        "pcc": float(pcc),
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "rel_rms": rel_rms,
        "ulp_max": ulp.max().item(),
        "ulp_p99": ulp.flatten().kthvalue(max(1, math.ceil(0.99 * ulp.numel()))).values.item(),
        "ratio_median": ratio.median().item(),
        "ratio_p5": ratio.kthvalue(max(1, math.ceil(0.05 * ratio.numel()))).values.item(),
        "ratio_p95": ratio.kthvalue(max(1, math.ceil(0.95 * ratio.numel()))).values.item(),
        "allclose": allclose,
    }
    for k, v in figures.items():
        record_property(k, v)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(figures) + "\n")

    # Gates: the shared check's pair (PCC and relative RMS together) and the specification's 1-ulp grade.
    assert pcc >= 0.9999, figures
    assert rel_rms <= 0.01, figures
    assert figures["ulp_max"] <= 1, figures
    # No scale bug: the ratio clusters at 1 within one bfloat16 rounding.
    assert abs(figures["ratio_median"] - 1.0) < 2**-8, figures
