# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for distributed_rms_norm (verifier, Phase 0).

Runs at the production geometry (G = 4 on a ring, 2 links if usable: the 1x4 snake-ring mock on a QuietBox 2, or a
real torus axis), at 4 shapes x {bf16, fp32, bf8b} input x {fp32 DEST, 16-bit DEST}. Per device it records PCC
(asserted with assert_with_pcc), max / mean abs error, relative RMS error, ULP error in the output dtype, and the
got/true ratio spread (median, p5, p95) — the scale-bug detector: a tight cluster around a constant != 1.0 means a
structural (scale) bug, a broad spread around 1.0 is rounding. The worst device of each case is printed (-s) and
appended to PRECISION_BASELINE_OUT (JSON lines) when that env var is set.

Run (one mesh shape per pytest session):
    scripts/run_safe_pytest.sh --dev tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_precision_baseline.py -s
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import _fabric_mesh_cache, mesh_device  # noqa: F401 (fixtures)
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

# The acceptance module (immutable spec) holds the torch reference and the mesh helpers; load it by path.
_spec = importlib.util.spec_from_file_location(
    "drms_acceptance", Path(__file__).with_name("test_distributed_rms_norm.py")
)
_acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acc)

RING_MESHES = _acc.RING_MESHES
LINEAR_AXES = _acc.LINEAR_AXES
# Production geometry first (a 4-device ring); fall back to the system mesh's linear group.
if RING_MESHES:
    MESH_SHAPE, CLUSTER_AXIS = RING_MESHES[0]
    TOPOLOGY = ttnn.Topology.Ring
else:
    MESH_SHAPE, CLUSTER_AXIS = None, (LINEAR_AXES or [1])[0]
    TOPOLOGY = ttnn.Topology.Linear

SHAPES = [
    (1, 1, 32, 1792),  # decode (R3-i, 1 wave)
    (1, 1, 640, 1792),  # PERF FOCUS (R3-i, 2 waves)
    (1, 1, 2048, 1280),  # prefill (R1)
    (1, 1, 128, 8192),  # wide local hidden (TP2 of 16384)
]
DTYPES = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32, "bf8b": ttnn.bfloat8_b}
PCC_FLOOR = {ttnn.float32: 0.9999, ttnn.bfloat16: 0.999, ttnn.bfloat8_b: 0.99}
MANTISSA_BITS = {ttnn.float32: 23, ttnn.bfloat16: 7, ttnn.bfloat8_b: 7}


def _ulp_error(actual, expected, dtype):
    """|actual - expected| in units of the output dtype's ulp at |expected| (bf8b: its bf16 mantissa; the shared
    exponent can only make the true bf8b ulp larger)."""
    e = expected.to(torch.float64)
    a = actual.to(torch.float64)
    mag = e.abs().clamp_min(torch.finfo(torch.float32).tiny)
    ulp = torch.pow(2.0, torch.floor(torch.log2(mag)) - MANTISSA_BITS[dtype])
    return ((a - e).abs() / ulp).flatten()


def _metrics(actual, expected, dtype):
    a = actual.to(torch.float32).flatten()
    e = expected.to(torch.float32).flatten()
    diff = (a - e).abs()
    rel_rms = float(torch.sqrt(((a - e) ** 2).mean()) / torch.sqrt((e**2).mean()).clamp_min(1e-30))
    # got/true ratio over the finite, non-tiny reference elements (tiny ones only measure absolute rounding)
    keep = torch.isfinite(a) & torch.isfinite(e) & (e.abs() > 1e-2 * e.abs().max())
    r = (a[keep] / e[keep]).to(torch.float64)
    ulp = _ulp_error(a, e, dtype)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_rms": rel_rms,
        "ulp_max": float(ulp.max()),
        "ulp_mean": float(ulp.mean()),
        "ratio_median": float(r.median()),
        "ratio_p5": float(torch.quantile(r, 0.05)),
        "ratio_p95": float(torch.quantile(r, 0.95)),
    }


@pytest.mark.parametrize("mesh_shape", [MESH_SHAPE])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32acc", "bf16acc"])
@pytest.mark.parametrize("dtype_name", list(DTYPES))
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_precision_baseline(mesh_shape, fp32_dest_acc_en, dtype_name, shape, mesh_device):
    dtype = DTYPES[dtype_name]
    if dtype == ttnn.float32 and not fp32_dest_acc_en:
        pytest.skip("EXCLUSION: float32 input needs fp32 DEST")
    torch.manual_seed(1234)
    rows, cols = tuple(mesh_device.shape)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(_acc.TORCH_DTYPE[dtype])
    x = _acc.quantize_like_device(x, dtype)
    gamma = torch.randn((rows, cols, shape[-1]), dtype=torch.float32).to(torch.bfloat16)
    eps = 1e-6
    expected = _acc.reference(x, gamma, CLUSTER_AXIS, eps)

    tx = _acc.to_mesh(x, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tg = _acc.make_gamma(gamma, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out = distributed_rms_norm(
        tx,
        cluster_axis=CLUSTER_AXIS,
        gamma=tg,
        epsilon=eps,
        topology=TOPOLOGY,
        compute_kernel_config=_acc.config(fp32_dest_acc_en),
    )
    got = _acc.device_outputs(out, mesh_device)

    worst = None
    for r in range(rows):
        for c in range(cols):
            actual = got[r][c]
            ref = expected[r, c]
            _, pcc_val = assert_with_pcc(ref, actual.to(torch.float32), PCC_FLOOR[dtype])
            _, allclose_msg = comp_allclose(ref, actual.to(torch.float32))
            m = _metrics(actual, ref, dtype)
            m["pcc"] = float(pcc_val)
            m["device"] = (r, c)
            m["allclose"] = allclose_msg
            if worst is None or m["pcc"] < worst["pcc"]:
                worst = m

    record = {
        "shape": list(shape),
        "dtype": dtype_name,
        "fp32_dest_acc_en": fp32_dest_acc_en,
        "G": tuple(mesh_device.shape)[CLUSTER_AXIS],
        "topology": str(TOPOLOGY),
        **worst,
    }
    print(
        f"\nPRECISION {record['shape']} {dtype_name} fp32acc={fp32_dest_acc_en}: pcc={worst['pcc']:.6f} "
        f"max_abs={worst['max_abs']:.4g} mean_abs={worst['mean_abs']:.4g} rel_rms={worst['rel_rms']:.4g} "
        f"ulp_max={worst['ulp_max']:.3g} ulp_mean={worst['ulp_mean']:.3g} "
        f"ratio med={worst['ratio_median']:.5f} p5={worst['ratio_p5']:.5f} p95={worst['ratio_p95']:.5f}"
    )
    out_path = os.environ.get("PRECISION_BASELINE_OUT")
    if out_path:
        with open(out_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    # A high-PCC / high-RMS case with a tight ratio cluster off 1.0 is a scale bug, never precision.
    assert not (
        worst["rel_rms"] > 0.05 and abs(worst["ratio_median"] - 1.0) > 0.01
    ), f"scale-bug signature: ratio median {worst['ratio_median']}"
    assert math.isfinite(worst["max_abs"])
