# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of typecast over the supported cells (verifier, Phase 0).

The specification grades every supported pair `exact` (NUMERIC_RESOLVED): the output must be the input value
rounded once to the output dtype, bit for bit (torch's `Tensor.to`, round-to-nearest-even). This test measures,
per (pair, shape): bit mismatches against torch, PCC (`comp_pcc`) and the abs errors (`comp_allclose`) against
the float64 input, relative RMS (RMS error / std of the reference, as eval/metrics.py), the error in units of
the output dtype's ULP, and the got/true ratio spread (the scale-bug detector).

Gate: zero bit mismatches (the `exact` requirement) and PCC / relative RMS within the legacy default (0.99, 0.13)
that a non-exact grader would apply. Each measurement is written to $TYPECAST_PRECISION_OUT (default
/tmp/typecast_precision_baseline) as one JSON file, because a test's output is not shown under the simulator's
forked workers.
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.typecast import typecast

TORCH = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
MANTISSA_BITS = {torch.bfloat16: 7, torch.float32: 23}
PAIRS = [(a, b) for a in (ttnn.bfloat16, ttnn.float32) for b in (ttnn.bfloat16, ttnn.float32)]
SHAPES = [(1, 4, 32, 64), (1, 1, 128, 128), (4, 8, 32, 256), (1, 1, 2048, 256)]
OUT_DIR = Path(os.environ.get("TYPECAST_PRECISION_OUT", "/tmp/typecast_precision_baseline"))


def _bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


def _ulp_error(got64, ref64, out_dtype):
    """|got - ref| in ULPs of the output dtype at |ref| (normal numbers; zero refs use the smallest normal)."""
    mag = ref64.abs().clamp(min=torch.finfo(out_dtype).tiny)
    ulp = torch.exp2(torch.floor(torch.log2(mag)) - MANTISSA_BITS[out_dtype])
    return ((got64 - ref64).abs() / ulp).max().item()


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("pair", PAIRS, ids=[f"{a.name}-{b.name}" for a, b in PAIRS])
def test_precision_baseline(device, record_property, pair, shape):
    in_dtype, out_dtype = pair
    torch.manual_seed(42)
    x = torch.randn(*shape, dtype=TORCH[in_dtype])  # the spec's input distribution: standard normal
    got = ttnn.to_torch(
        typecast(ttnn.from_torch(x, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device), dtype=out_dtype)
    )
    expected_rounded = x.to(TORCH[out_dtype])
    ref64 = x.double()  # the specification's reference: the input value itself, in float64
    got64 = got.double()

    mismatches = int((_bits(got) != _bits(expected_rounded)).sum().item())
    _, pcc = comp_pcc(ref64.float(), got64.float(), pcc=0.99)
    _, allclose_msg = comp_allclose(got64.float(), ref64.float())
    diff = got64 - ref64
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel_rms = math.sqrt(diff.pow(2).mean().item()) / max(ref64.std().item(), 1e-12)
    ulp_max = _ulp_error(got64, ref64, TORCH[out_dtype])
    nz = (ref64 != 0) & torch.isfinite(ref64) & torch.isfinite(got64)
    ratio = (got64[nz] / ref64[nz]).float()
    r_med, r_p5, r_p95 = (torch.quantile(ratio, q).item() for q in (0.5, 0.05, 0.95))

    result = {
        "pair": f"{in_dtype.name}->{out_dtype.name}",
        "shape": list(shape),
        "bit_mismatches": mismatches,
        "elements": x.numel(),
        "pcc": float(pcc),
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "rel_rms": rel_rms,
        "ulp_max": ulp_max,
        "ratio_median": r_med,
        "ratio_p5": r_p5,
        "ratio_p95": r_p95,
        "comp_allclose": allclose_msg,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{in_dtype.name}-{out_dtype.name}-{'x'.join(map(str, shape))}.json").write_text(json.dumps(result))
    for k, v in result.items():
        record_property(k, v)

    assert mismatches == 0, f"{mismatches} elements differ from torch's rounding: {result}"
    assert pcc >= 0.99 and rel_rms <= 0.13, result
    # One rounding to the output dtype: at most half an ULP of it (0 for a widening or same-dtype copy).
    assert ulp_max <= 0.5, result
