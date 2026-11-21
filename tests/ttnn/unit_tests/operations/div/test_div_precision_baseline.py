# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""div precision baseline over the Phase 0 surface (bfloat16 operands, fp32 DEST, TILE, DRAM).

Measures, against the float64 reference on the stored operands (the specification's `reference`):
PCC (comp_pcc), max abs error (comp_allclose), mean abs error, relative RMS (RMS error / reference std,
as eval/metrics.py), the ULP distance in the output dtype, and the got/true ratio spread (the scale-bug
detector: a tight cluster around a constant != 1 is a structural bug, not rounding).

Gates: NUMERIC of eval/specs/div/spec.py. A bfloat16 output is graded at 0 ulp (NUMERIC_RESOLVED); a
float32 output of bfloat16 operands by `legacy_default` (PCC >= 0.999, relative RMS <= 0.01). A floor or
trunc cell, on inputs whose quotient is kept away from every integer (as the spec's HOOKS do), must equal
the reference.

Test output is hidden under the simulator's workers: every case writes its numbers to
$DIV_PRECISION_DIR (default /tmp/div_precision_baseline/<case>.json) and records them with record_property.
"""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.div import div

OUT_DIR = Path(os.environ.get("DIV_PRECISION_DIR", "/tmp/div_precision_baseline"))
PCC_MIN, RMS_MAX = 0.999, 0.01  # NUMERIC["legacy_default"]

# (id, shape of a, shape of b or None for a number, the number)
CASES = [
    ("1x1x32x64", (1, 1, 32, 64), (1, 1, 32, 64), None),
    ("1x2x8400+number3", (1, 2, 8400), None, 3.0),  # the traced YOLO shape, a divisor that is not a power of 2
    ("4x8x32x256", (4, 8, 32, 256), (4, 8, 32, 256), None),
    ("2x512x1024", (2, 512, 1024), (2, 512, 1024), None),
]


def _ulp(got, ref, dtype):
    """|ULP distance| between got and ref rounded into `dtype` (ordered integer view of the bit patterns)."""
    if dtype == ttnn.bfloat16:
        g = got.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        r = ref.to(torch.float32).to(torch.bfloat16).view(torch.int16).to(torch.int32)
    else:
        g = got.to(torch.float32).view(torch.int32).to(torch.int64)
        r = ref.to(torch.float32).view(torch.int32).to(torch.int64)
    flip = lambda x, top: torch.where(x < 0, -(x & top), x)  # sign-magnitude -> ordered
    top = 0x7FFF if dtype == ttnn.bfloat16 else 0x7FFFFFFF
    return (flip(g, top) - flip(r, top)).abs()


def _metrics(got, ref, out_dtype):
    got64, ref32 = got.double(), ref.float()
    _, pcc = comp_pcc(ref32, got.float(), 0.0)
    _, allclose = comp_allclose(got.float(), ref32)
    max_abs = float(allclose.split("Max ATOL Delta:")[1].split(",")[0])
    diff = got64 - ref
    rms = diff.pow(2).mean().sqrt().item() / ref.std().item()
    ulp = _ulp(got, ref, out_dtype)
    mask = torch.isfinite(ref) & (ref != 0)
    ratio = (got64[mask] / ref[mask]).float()
    q = torch.quantile(ratio[: 1 << 24], torch.tensor([0.05, 0.5, 0.95]))
    return {
        "pcc": float(pcc),
        "max_abs_err": max_abs,
        "mean_abs_err": diff.abs().mean().item(),
        "rel_rms": rms,
        "ulp_max": int(ulp.max()),
        "ulp_mean": ulp.double().mean().item(),
        "frac_ulp_nonzero": (ulp != 0).double().mean().item(),
        "ratio_p5": q[0].item(),
        "ratio_median": q[1].item(),
        "ratio_p95": q[2].item(),
        "ratio_std": ratio.double().std().item(),
    }


def _record(record_property, name, result):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(result, indent=1))
    for k, v in result.items():
        record_property(k, v)


def _put(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.float32], ids=["out_bf16", "out_fp32"])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_precision_baseline(device, case, out_dtype, record_property):
    name, a_shape, b_shape, number = case
    torch.manual_seed(0)
    ta = torch.randn(a_shape).to(torch.bfloat16)
    if number is None:
        tb = torch.randn(b_shape).to(torch.bfloat16)
        tb = torch.where(tb == 0, torch.ones_like(tb), tb)
        b_dev, ref_b = _put(tb, device), tb.double()
    else:
        b_dev, ref_b = number, torch.tensor(number, dtype=torch.float64)
    ref = torch.div(ta.double(), ref_b)

    got = ttnn.to_torch(div(_put(ta, device), b_dev, dtype=out_dtype))
    result = {"case": name, "out_dtype": str(out_dtype), "rounding_mode": None, **_metrics(got, ref, out_dtype)}
    _record(record_property, f"{name}-{'bf16' if out_dtype == ttnn.bfloat16 else 'fp32'}", result)

    assert result["pcc"] >= PCC_MIN and result["rel_rms"] <= RMS_MAX, result
    if out_dtype == ttnn.bfloat16:
        assert result["ulp_max"] == 0, result  # NUMERIC_RESOLVED[('bfloat16', 'default')]: 0 ulp


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.float32], ids=["out_bf16", "out_fp32"])
def test_precision_rounding(device, rounding_mode, out_dtype, record_property):
    """Quotient drawn away from every integer (as the spec's build_inputs): the result must be exact."""
    shape = (4, 8, 32, 256)
    torch.manual_seed(1)
    quotient = torch.randint(-16, 16, shape).float() + 0.25 + 0.5 * torch.rand(shape)
    tb = torch.randn(shape).to(torch.bfloat16)
    tb = torch.where(tb == 0, torch.ones_like(tb), tb)
    ta = (quotient * tb.float()).to(torch.bfloat16)
    ref = torch.div(ta.double(), tb.double(), rounding_mode=rounding_mode)

    got = ttnn.to_torch(div(_put(ta, device), _put(tb, device), rounding_mode=rounding_mode, dtype=out_dtype))
    result = {"case": "4x8x32x256", "out_dtype": str(out_dtype), "rounding_mode": rounding_mode}
    result.update(_metrics(got, ref, out_dtype))
    result["mismatches"] = int((got.double() != ref).sum())
    _record(record_property, f"4x8x32x256-{rounding_mode}-{'bf16' if out_dtype == ttnn.bfloat16 else 'fp32'}", result)
    assert result["mismatches"] == 0, result
