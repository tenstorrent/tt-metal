# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of layer_norm over its supported cells (bfloat16, TILE, rank 4, tile-aligned, DRAM), for
both accumulators, on four shapes from small to the model's widest.

Measured per case: PCC (comp_pcc), max abs error (comp_allclose), mean abs error, relative RMS (RMS error over
the reference's std, as eval/metrics.check_output does), the ULP p99 at bfloat16, and the got/true ratio
r = actual / expected over the finite elements whose reference is not near zero (median and p5..p95 spread).
A tight r cluster around a constant other than 1.0 is a scale/structural bug; a spread centred on 1.0 is
rounding noise.

Gates: the 16-bit accumulator at the specification's legacy tolerance for it, (PCC 0.999, rel-RMS 0.13); fp32
accumulation at the shared table's bfloat16 row, eval.metrics.default_tolerance(bfloat16, True).

Results are written as JSON lines to $LAYER_NORM_PRECISION_OUT (default /tmp/layer_norm_precision_baseline.jsonl)
and recorded with record_property: a test's own output is not shown under the simulator's workers.
"""

import json
import os
import re

import pytest
import torch

import ttnn
from eval.metrics import _ulp_p99, default_tolerance
from eval.specs.layer_norm.spec import NUMERIC, reference
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.layer_norm import layer_norm

SHAPES = [(1, 1, 32, 64), (2, 1, 512, 512), (1, 1, 128, 1024), (1, 1, 64, 4096)]
EPS = 1e-6
OUT = os.environ.get("LAYER_NORM_PRECISION_OUT", "/tmp/layer_norm_precision_baseline.jsonl")


def _tolerance(fp32):
    legacy = NUMERIC["legacy_tolerances"].get((ttnn.bfloat16, fp32))
    return legacy if legacy is not None else default_tolerance(ttnn.bfloat16, fp32)


@pytest.mark.parametrize("fp32", [True, False], ids=["fp32acc", "bf16acc"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_precision_baseline(device, shape, fp32, record_property):
    torch.manual_seed(1234)
    W = shape[-1]
    x = torch.randn(shape).bfloat16()
    w = torch.randn(1, 32, W).bfloat16()
    b = torch.randn(1, 32, W).bfloat16()
    put = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32
    )
    out = layer_norm(put(x), weight=put(w), bias=put(b), epsilon=EPS, compute_kernel_config=config)
    got = ttnn.to_torch(out)
    expected = reference(x, w, b, EPS)  # float64

    _, pcc = comp_pcc(expected.float(), got.float())
    _, allclose = comp_allclose(expected.float(), got.float())
    max_abs = float(re.search(r"Max ATOL Delta: ([0-9.eE+-]+|nan|inf)", allclose).group(1))
    diff = got.double() - expected
    mean_abs = diff.abs().mean().item()
    rel_rms = diff.pow(2).mean().sqrt().item() / expected.std().item()
    ulp99 = _ulp_p99(got.float(), expected.float(), torch.bfloat16)

    finite = torch.isfinite(got.double()) & (expected.abs() > 1e-3 * expected.abs().max())
    r = got.double()[finite] / expected[finite]
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()

    row = {
        "shape": list(shape), "fp32_dest_acc_en": fp32, "pcc": pcc, "max_abs": max_abs, "mean_abs": mean_abs,
        "rel_rms": rel_rms, "ulp_p99": ulp99, "ratio_p5": q[0], "ratio_median": q[1], "ratio_p95": q[2],
    }
    for k, v in row.items():
        record_property(k, v)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")

    pcc_min, rms_max = _tolerance(fp32)
    assert pcc >= pcc_min, row
    assert rel_rms <= rms_max, row
    # Scale-bug detector: the ratio's median must sit on 1.0, not on another constant.
    assert abs(q[1] - 1.0) < 0.01, row
