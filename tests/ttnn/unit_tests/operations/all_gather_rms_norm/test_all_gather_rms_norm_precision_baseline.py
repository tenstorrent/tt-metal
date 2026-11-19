# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for all_gather_rms_norm (verifier).

Measures, per device of the 1x4 production ring (cluster_axis 1, Ring, 2 links), PCC, max / mean abs error, relative
RMS error (RMS error / reference RMS, as helpers.check_output), and the got/true ratio spread (the scale-bug
detector: a tight cluster around a non-1.0 constant is a uniform scale error, a broad spread centred on 1.0 is
rounding noise), plus the per-row least-squares scale (mean / std over rows).

Gated on PCC and relative RMS together against helpers.TOLERANCES, except the one known precision gap
(bfloat16 layer_norm with fp32_dest_acc_en=False: op_requirements.md Refinement 2), which is measured and
reported but only gated on PCC.
"""

from __future__ import annotations

import pytest
import torch
import ttnn
from loguru import logger

from eval.golden_tests.all_gather_rms_norm.helpers import TOLERANCES, create_ttnn_gamma, create_ttnn_input_tensor
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

from .test_all_gather_rms_norm import (
    _DEVICE_PARAMS,
    EPS,
    ROW_OFFSET_SCALE,
    make_input,
    make_weight,
    torch_all_gather_norm,
)

SHAPES = [
    (1, 1, 32, 384),  # decode q_a latent
    (1, 1, 640, 32),  # indexer k (perf case 2 shape), one tile per slice
    (1, 1, 640, 384),  # PERF FOCUS shape
    (1, 1, 128, 1792),  # wide hidden
]

# (id, dtype, norm_type, fp32_dest_acc_en, known_gap)
CONFIGS = [
    ("bf16_rms", ttnn.bfloat16, "rms_norm", True, False),
    ("bf16_ln", ttnn.bfloat16, "layer_norm", True, False),
    ("fp32_ln", ttnn.float32, "layer_norm", True, False),
    ("bf8b_rms", ttnn.bfloat8_b, "rms_norm", True, False),
    ("bf16_ln_acc16", ttnn.bfloat16, "layer_norm", False, True),
]


def _metrics(actual, expected):
    a = actual.to(torch.float64)
    e = expected.to(torch.float64)
    _, pcc = comp_pcc(e, a)
    _, allclose_msg = comp_allclose(e, a)
    err = (a - e).abs()
    rel_rms = float(((a - e) ** 2).mean().sqrt() / max(float((e**2).mean().sqrt()), 1e-30))
    mask = e.abs() > 1e-2
    ratio = (a[mask] / e[mask]).flatten()
    A = a.reshape(-1, a.shape[-1])
    E = e.reshape(-1, e.shape[-1])
    valid = E.abs().sum(-1) > 0
    row_scale = (A * E).sum(-1)[valid] / (E * E).sum(-1)[valid]
    return dict(
        pcc=float(pcc),
        max_abs=float(err.max()),
        mean_abs=float(err.mean()),
        rel_rms=rel_rms,
        ratio_median=float(ratio.median()),
        ratio_p5=float(ratio.quantile(0.05)),
        ratio_p95=float(ratio.quantile(0.95)),
        row_scale_mean=float(row_scale.mean()),
        row_scale_std=float(row_scale.std()) if row_scale.numel() > 1 else 0.0,
        allclose=allclose_msg,
    )


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["mesh1x4"], indirect=True)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c[0])
def test_precision_baseline(mesh_device, shape, cfg):
    name, dtype, norm_type, fp32_dest_acc_en, known_gap = cfg
    cluster_axis, G = 1, tuple(mesh_device.shape)[1]
    layer_norm = norm_type == "layer_norm"
    W = G * shape[-1]
    g_dtype = ttnn.float32 if dtype == ttnn.float32 else ttnn.bfloat16
    x = make_input(
        tuple(mesh_device.shape), shape, dtype, seed=7, row_offset_scale=ROW_OFFSET_SCALE if layer_norm else 0.0
    )
    gamma = make_weight(W, g_dtype, 8)
    beta = make_weight(W, g_dtype, 9) if layer_norm else None
    expected = torch_all_gather_norm(x, cluster_axis, gamma, beta, norm_type, EPS)

    tx = create_ttnn_input_tensor(x, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tg = create_ttnn_gamma(gamma, mesh_device, dtype=g_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    kwargs = {}
    if layer_norm:
        kwargs = dict(
            norm_type="layer_norm",
            beta=create_ttnn_gamma(beta, mesh_device, dtype=g_dtype, layout=ttnn.ROW_MAJOR_LAYOUT),
        )
    ckc = ttnn.ComputeConfigDescriptor()
    ckc.math_fidelity = ttnn.MathFidelity.HiFi4
    ckc.fp32_dest_acc_en = fp32_dest_acc_en
    ckc.math_approx_mode = False
    out = all_gather_rms_norm(
        tx,
        cluster_axis=cluster_axis,
        gamma=tg,
        epsilon=EPS,
        topology=ttnn.Topology.Ring,
        num_links=2,
        compute_kernel_config=ckc,
        **kwargs,
    )
    pcc_t, rms_t = TOLERANCES[(dtype, fp32_dest_acc_en)]
    worst = None
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        m = _metrics(ttnn.to_torch(t), expected[0, idx])
        if worst is None or m["rel_rms"] > worst["rel_rms"]:
            worst = m
    logger.info(
        f"PRECISION {name} {'x'.join(map(str, shape))}: pcc={worst['pcc']:.6f} max_abs={worst['max_abs']:.4f} "
        f"mean_abs={worst['mean_abs']:.5f} rel_rms={worst['rel_rms']:.5f} | ratio median={worst['ratio_median']:.5f} "
        f"p5={worst['ratio_p5']:.4f} p95={worst['ratio_p95']:.4f} | row scale mean={worst['row_scale_mean']:.5f} "
        f"std={worst['row_scale_std']:.5f}"
    )
    assert worst["pcc"] >= pcc_t, f"pcc {worst['pcc']} < {pcc_t}"
    if not known_gap:
        assert worst["rel_rms"] <= rms_t, f"rel_rms {worst['rel_rms']} > {rms_t}"
