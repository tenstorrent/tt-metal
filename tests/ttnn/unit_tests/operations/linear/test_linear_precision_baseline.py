# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of linear (Phase 0): PCC, max / mean absolute error, relative RMS, ULP (p99) and the
spread of the got/true ratio, on the models' shapes, with both DEST accumulator widths.

The inputs are drawn as the golden suite draws them (standard normal, stored in bfloat16); a bias holds the
specification's `bias_values(K, N)`; the reference is the specification's `reference` in float64 on the
stored inputs. Every case is gated on PCC and relative RMS together (as `eval.metrics.check_output`
does), at the specification's `legacy_default` (PCC >= 0.99, relative RMS <= 0.15), and LoFi at ResNet-50's
own tolerance.

Under the simulator's workers a test's output is not shown: every case writes its numbers to
`$LINEAR_PRECISION_DIR` (default /tmp/linear_precision_baseline/<case>.json) and records them with
`record_property`.
"""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from eval.metrics import compute_metrics_torch
from eval.specs.linear.spec import NUMERIC, bias_values, reference
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.linear import linear

OUT_DIR = Path(os.environ.get("LINEAR_PRECISION_DIR", "/tmp/linear_precision_baseline"))
DEFAULT_TOL = NUMERIC["legacy_default"]  # (PCC >=, relative RMS <=)
LOFI_TOL = (0.98, 0.2)  # the specification's loose case resnet50_fc_lofi

# (id, shape of a, shape of b, bias form)
SHAPES = [
    ("llama_32x64x128", (1, 1, 32, 64), (1, 1, 64, 128), None),
    ("padded_64x96x160", (1, 1, 64, 96), (96, 160), None),
    ("qwen_128x256x512_vector_bias", (1, 1, 128, 256), (256, 512), "vector"),
    ("resnet_32x2048x1024_row_bias", (1, 1, 32, 2048), (1, 1, 2048, 1024), "row"),
]


def _bias_tensor(form, k, n, b_rank, device):
    if form is None:
        return None, None
    shape = (n,) if form == "vector" else (1,) * (b_rank - 1) + (n,)
    values = bias_values(k, n).reshape(shape).to(torch.bfloat16)
    return values, ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _ratio_spread(got, ref):
    """got / true over the finite elements whose reference is not near zero: (median, p5, p95)."""
    mask = torch.isfinite(got) & (ref.abs() > 1e-3 * ref.abs().max())
    r = (got[mask] / ref[mask]).float()
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95]))
    return float(q[1]), float(q[0]), float(q[2])


def _measure(device, case, a_shape, b_shape, bias_form, config, tol, record_property):
    torch.manual_seed(42)
    ta = torch.randn(a_shape).to(torch.bfloat16)
    tb = torch.randn(b_shape).to(torch.bfloat16)
    k, n = a_shape[-1], b_shape[-1]
    tbias, bias = _bias_tensor(bias_form, k, n, len(b_shape), device)
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    got = ttnn.to_torch(linear(a, b, bias=bias, compute_kernel_config=config))
    ref = reference(ta.double(), tb.double(), {"bias": bias_form})

    _, pcc = comp_pcc(got.double(), ref)
    _, allclose = comp_allclose(got.double(), ref)
    max_abs = float(allclose.split("Max ATOL Delta:")[1].split(",")[0])
    mean_abs = float((got.double() - ref).abs().mean())
    shared = compute_metrics_torch(got, ref, readback_dtype=got.dtype)
    median_r, p5_r, p95_r = _ratio_spread(got.double(), ref)

    result = {
        "case": case,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "math_fidelity": str(config.math_fidelity),
        "pcc": float(pcc),
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "rel_rms": shared.rms,
        "ulp_p99": shared.ulp_p99,
        "ref_abs_max": float(ref.abs().max()),
        "ratio_median": median_r,
        "ratio_p5": p5_r,
        "ratio_p95": p95_r,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{case}.json").write_text(json.dumps(result, indent=1))
    for key, value in result.items():
        record_property(key, value)

    pcc_floor, rms_ceiling = tol
    assert pcc >= pcc_floor and shared.rms <= rms_ceiling, result
    # A uniform scale bug keeps PCC high and moves the ratio off 1 for every element.
    assert abs(median_r - 1.0) < 0.01, result


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32_dest", "bf16_dest"])
@pytest.mark.parametrize("case,a_shape,b_shape,bias_form", SHAPES, ids=[s[0] for s in SHAPES])
def test_precision_baseline(device, record_property, case, a_shape, b_shape, bias_form, fp32_dest_acc_en):
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest_acc_en, math_approx_mode=False
    )
    tag = "fp32_dest" if fp32_dest_acc_en else "bf16_dest"
    _measure(device, f"{case}-{tag}", a_shape, b_shape, bias_form, config, DEFAULT_TOL, record_property)


def test_precision_resnet_lofi(device, record_property):
    """ResNet-50's own compute config: LoFi, approximation on, 16-bit DEST."""
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, math_approx_mode=True
    )
    _, a_shape, b_shape, bias_form = SHAPES[-1]
    _measure(device, "resnet_32x2048x1024_row_bias-lofi", a_shape, b_shape, bias_form, config, LOFI_TOL,
             record_property)
