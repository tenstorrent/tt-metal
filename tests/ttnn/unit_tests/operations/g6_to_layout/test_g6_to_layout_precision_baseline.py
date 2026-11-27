# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_to_layout (TILE -> ROW_MAJOR, bfloat16, interleaved DRAM).

Covers both regimes: staged_reorder (c309, c1993, c2020) and face_direct (c286).

A pure move: the gate is bit identity (helpers.TOLERANCES[bfloat16] == ("exact", None)). The
float metrics are recorded for the verification report; every one must be the trivial value.
Production shapes only (BB-22).
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_to_layout import g6_to_layout

# -0.0, subnormals, +inf, -inf, NaN with a payload.
SPECIAL_BITS = (0x8000, 0x0001, 0x8001, 0x007F, 0x7F80, 0xFF80, 0x7FC1)

SHAPES = [
    pytest.param((1, 185), id="c309-small-w-unaligned"),
    pytest.param((16, 1, 2560), id="c1993-h1-multi-image"),
    pytest.param((1, 1024, 1024, 3), id="c286-face-direct"),
    pytest.param((8, 1024, 2560), id="c2020-large"),
]


def _ordered_int(t: torch.Tensor) -> torch.Tensor:
    """bf16 bits mapped to a monotonic integer line, so |a - b| counts ULPs."""
    bits = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -0x8000 - bits, bits)


def _quantile(t: torch.Tensor, q: float) -> float:
    # torch.quantile caps its input size; kthvalue does not.
    k = min(t.numel(), max(1, round(q * t.numel())))
    return float(t.kthvalue(k).values)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    finite = torch.isfinite(expected)
    a = actual[finite].to(torch.float64)
    e = expected[finite].to(torch.float64)
    _, pcc = comp_pcc(expected[finite].float(), actual[finite].float(), 0.0)
    _, allclose_msg = comp_allclose(expected[finite].float(), actual[finite].float())
    err = a - e
    nonzero = e != 0
    ratio = a[nonzero] / e[nonzero]
    return {
        "pcc": float(pcc),
        "allclose": allclose_msg,
        "max_abs": float(err.abs().max()),
        "mean_abs": float(err.abs().mean()),
        "rel_rms": float(err.pow(2).mean().sqrt() / e.std()),
        "max_ulp": int((_ordered_int(actual) - _ordered_int(expected)).abs().max()),
        "bit_mismatches": int((actual.view(torch.int16) != expected.view(torch.int16)).sum()),
        "ratio_median": float(ratio.median()),
        "ratio_p5_p95": (_quantile(ratio, 0.05), _quantile(ratio, 0.95)),
    }


def _run(device, torch_input):
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=True,
    )
    return ttnn.to_torch(g6_to_layout(ttnn_input, ttnn.ROW_MAJOR_LAYOUT)).reshape(torch_input.shape)


@pytest.mark.parametrize("shape", SHAPES)
def test_precision_baseline_randn(device, shape):
    torch.manual_seed(1234)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    m = _metrics(_run(device, torch_input), torch_input)
    print(f"\nPRECISION {shape} randn: {m}")

    assert m["bit_mismatches"] == 0
    assert m["max_ulp"] == 0
    assert m["pcc"] == pytest.approx(1.0, abs=1e-12)
    assert m["rel_rms"] == 0.0
    assert m["ratio_p5_p95"] == (1.0, 1.0)


@pytest.mark.parametrize("shape", SHAPES)
def test_precision_baseline_special_bits(device, shape):
    torch.manual_seed(1234)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    flat = torch_input.view(-1).view(torch.int16)
    count = max(len(SPECIAL_BITS), flat.numel() // 100)
    positions = torch.randperm(flat.numel())[:count]
    table = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    flat[positions] = table[torch.arange(count) % len(SPECIAL_BITS)]

    actual = _run(device, torch_input)
    mismatches = int((actual.view(torch.int16) != torch_input.view(torch.int16)).sum())
    print(f"\nPRECISION {shape} specials: {count} special values, {mismatches} bit mismatches")

    assert mismatches == 0
