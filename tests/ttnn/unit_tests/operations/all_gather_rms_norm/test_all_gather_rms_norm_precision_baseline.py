# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for all_gather_rms_norm (verifier, Phase 0).

Measures, over every device of the 4-device group (1x4 snake ring, Ring, 2 links, FABRIC_2D_TORUS_XY):
  PCC (comp_pcc), max / mean abs error (comp_allclose + direct), relative RMS error (RMS error / reference RMS, the
  golden helpers' definition, and RMS error / reference std, eval/metrics.py's), and the got/true ratio spread over
  the finite, non-tiny-reference elements (scale-bug detector: a tight cluster around a constant != 1 is a scale bug,
  a broad spread centred on 1 is rounding noise).

Gate: PCC and relative RMS together, at the golden suite's TOLERANCES for (dtype, fp32_dest_acc_en=True).
Results print with -s (one `PRECISION` line per case); verification_report.md records them.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc

from eval.golden_tests.all_gather_rms_norm.helpers import (
    LN_ROW_OFFSET_SCALE,
    TOLERANCES,
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    gathered_shape,
    pytorch_all_gather_rms_norm,
    quantize_like_device,
    random_gamma,
    random_stacked,
)
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

from .test_all_gather_rms_norm import DEVICE_PARAMS

EPS = 1e-6
_TORCH = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}

# (shape, norm_type, dtype): decode / perf-focus / LN perf-focus / wide prefill / R_U (W_local non-aligned)
CASES = [
    ((1, 1, 32, 384), "rms_norm", ttnn.bfloat16),
    ((1, 1, 640, 384), "rms_norm", ttnn.bfloat16),
    ((1, 1, 640, 384), "rms_norm", ttnn.float32),
    ((1, 1, 640, 384), "rms_norm", ttnn.bfloat8_b),
    ((1, 1, 640, 1792), "rms_norm", ttnn.bfloat16),
    ((1, 1, 17, 50), "rms_norm", ttnn.bfloat16),
    ((1, 1, 640, 32), "layer_norm", ttnn.bfloat16),
    ((1, 1, 640, 32), "layer_norm", ttnn.float32),
    ((1, 1, 640, 384), "layer_norm", ttnn.bfloat16),
    ((1, 1, 640, 1792), "layer_norm", ttnn.bfloat16),
]


def _id(c):
    shape, norm, dtype = c
    return f"{'x'.join(map(str, shape))}-{norm}-{str(dtype).split('.')[-1]}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_precision_baseline(mesh_device, case):
    shape, norm_type, dtype = case
    cluster_axis = 1
    G = tuple(mesh_device.shape)[cluster_axis]
    out_shape = gathered_shape(shape, G)
    ln = norm_type == "layer_norm"
    tdt = _TORCH[dtype]
    x = random_stacked(mesh_device, shape, tdt, seed=7, row_offset_scale=LN_ROW_OFFSET_SCALE if ln else 0.0)
    x = quantize_like_device(x, dtype)
    gdt = ttnn.bfloat16 if dtype == ttnn.bfloat8_b else dtype
    gamma = random_gamma(out_shape[-1], _TORCH[gdt], seed=8)
    beta = random_gamma(out_shape[-1], _TORCH[gdt], seed=9) if ln else None
    kwargs = dict(gamma=create_ttnn_gamma(gamma, mesh_device, dtype=gdt, layout=ttnn.ROW_MAJOR_LAYOUT))
    if ln:
        kwargs.update(
            norm_type="layer_norm", beta=create_ttnn_gamma(beta, mesh_device, dtype=gdt, layout=ttnn.ROW_MAJOR_LAYOUT)
        )
    expected = pytorch_all_gather_rms_norm(
        x.to(torch.float32), gamma, cluster_axis, EPS, norm_type=norm_type, beta=beta
    ).to(torch.float64)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    out = all_gather_rms_norm(
        tx, cluster_axis=cluster_axis, epsilon=EPS, topology=ttnn.Topology.Ring, num_links=2, **kwargs
    )
    devs = ttnn.get_device_tensors(out)
    rows, cols = tuple(mesh_device.shape)
    got = torch.stack([ttnn.to_torch(devs[r * cols + c]).to(torch.float64) for r in range(rows) for c in range(cols)])
    ref = expected.reshape(rows * cols, *out_shape)

    _, pcc = comp_pcc(ref, got)
    allclose_ok, allclose_msg = comp_allclose(ref, got)
    err = (got - ref).abs()
    max_abs = err.max().item()
    mean_abs = err.mean().item()
    rms_err = ((got - ref) ** 2).mean().sqrt().item()
    rel_rms = rms_err / max((ref**2).mean().sqrt().item(), 1e-30)
    rel_rms_std = rms_err / max(ref.std().item(), 1e-30)
    mask = torch.isfinite(got) & (ref.abs() > 1e-2 * ref.abs().max())
    r = (got[mask] / ref[mask]).float()
    r_med = r.median().item()
    q = torch.quantile(r[: min(r.numel(), 1 << 24)], torch.tensor([0.05, 0.95]))
    print(
        f"\nPRECISION {_id(case)}: pcc={pcc:.6f} max_abs={max_abs:.4g} mean_abs={mean_abs:.4g} "
        f"rel_rms={rel_rms:.5f} rel_rms_std={rel_rms_std:.5f} ratio_median={r_med:.5f} "
        f"ratio_p5={q[0].item():.5f} ratio_p95={q[1].item():.5f} | {allclose_msg}"
    )
    tol_pcc, tol_rms = TOLERANCES[(dtype, True)]
    assert torch.isfinite(got).all()
    assert pcc >= tol_pcc and rel_rms <= tol_rms, (pcc, rel_rms)
    assert abs(r_med - 1.0) < 0.01, f"got/true ratio median {r_med} != 1: scale bug"
