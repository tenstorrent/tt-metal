# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of zeros_like (verifier).

The specification grades zeros_like `exact` (NUMERIC_RESOLVED: full_like, movement class) and its reference is
constant, so the relative RMS is the absolute one (eval/metrics.py): only zeros meet it. This test measures, per
shape, PCC (comp_pcc), max / mean abs error (comp_allclose + torch), RMS error, the ULP distance and the bit
pattern of every element (+0.0 is required: tt-metal writes +0.0, spec DECISIONS), and the got/true ratio
spread (undefined here: the reference has no non-zero element, recorded as such).

A second input variant holds NaN and +-inf: the input's values are never read (rule R1), so the result is the
same zeros.

Measurements go to `record_property` and, when ZEROS_LIKE_BASELINE_OUT names a file, one JSON line per case
(a test's own output is not shown under the simulator's workers).
"""

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.zeros_like import zeros_like

# Small (the traced Qwen3-VL call, h_non_aligned), both dims non-aligned (rank 4), wide hidden, and the largest
# INPUTS shape (1024 pages, 32 per cluster).
SHAPES = [(70, 512), (2, 1, 100, 47), (1, 1, 32, 8192), (1024, 1024)]

PCC_MIN = 0.9999  # NUMERIC["legacy_default"]
RMS_MAX = 0.0  # NUMERIC["legacy_default"]: constant reference, absolute RMS, met by zeros only


def _input(shape, kind, seed=0):
    torch.manual_seed(seed)
    t = torch.randn(shape, dtype=torch.float32)
    if kind == "special":
        flat = t.view(-1)
        flat[0::7] = float("nan")
        flat[1::7] = float("inf")
        flat[2::7] = float("-inf")
    return t.to(torch.bfloat16)


def _ulp_bf16(got, expected):
    """ULP distance of two bfloat16 tensors (ordered-integer form of the bit patterns)."""

    def ordered(x):
        bits = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(got) - ordered(expected)).abs()


@pytest.mark.parametrize("kind", ["normal", "special"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_zeros_like_precision_baseline(device, shape, kind, record_property):
    t = _input(shape, kind)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = zeros_like(x)
    got = ttnn.to_torch(out)
    assert got.dtype == torch.bfloat16 and tuple(got.shape) == tuple(shape)
    expected = torch.zeros_like(t)

    _, pcc = comp_pcc(expected, got, PCC_MIN)
    _, allclose_msg = comp_allclose(expected, got)
    diff = (got.double() - expected.double()).abs()
    rms = float(torch.sqrt((diff**2).mean()))
    ulp = _ulp_bf16(got, expected)
    neg_zero = int(torch.signbit(got).sum())
    nonfinite = int((~torch.isfinite(got)).sum())

    # got/true ratio spread over finite, non-zero reference elements: the scale-bug detector. The reference is
    # all zeros, so the set is empty and no scale can be read; recorded as such.
    ref_nonzero = int(torch.count_nonzero(expected))
    ratio = "n/a (reference has no non-zero element)" if ref_nonzero == 0 else None

    m = {
        "shape": list(shape),
        "input": kind,
        "pcc": float(pcc),
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "rms_err": rms,
        "ulp_max": int(ulp.max()),
        "negative_zeros": neg_zero,
        "non_finite": nonfinite,
        "ratio_spread": ratio,
        "allclose": allclose_msg,
    }
    for k, v in m.items():
        record_property(k, v)
    path = os.environ.get("ZEROS_LIKE_BASELINE_OUT")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(m) + "\n")

    assert pcc >= PCC_MIN, m
    assert rms <= RMS_MAX, m
    assert int(ulp.max()) == 0 and neg_zero == 0 and nonfinite == 0, m  # exact, +0.0 everywhere
    # R1/R2: the input is untouched (its NaN/inf included).
    assert torch.equal(ttnn.to_torch(x).view(torch.int16), t.view(torch.int16))
