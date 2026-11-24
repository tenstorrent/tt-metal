# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for q_stem (Phase 0: bf16 everywhere, fp32 DEST, HiFi4, NoPE, TILE out).

Per shape and per device: PCC (comp_pcc), max / mean abs error (comp_allclose), relative RMS error (RMS error /
std of the reference, as eval.metrics.check_output), bf16 ULP error (|got - true| / ulp_bf16(true)), and the
got / true ratio spread (median, p5, p95 over finite elements with |true| above 1% of its std): a tight cluster
around a non-1.0 constant would be a scale / structural bug, a broad spread centred on 1.0 is rounding noise.
The nope (prefix) and rope (pass-through) parts are reported separately: the rope part is a bit-exact copy of
the bf16 q, so its error is only the bf16 rounding of q.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.metrics import default_tolerance
from models.common.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.unit_tests.operations.q_stem.test_q_stem import (
    _DEVICE_PARAMS,
    _MESH_PARAM,
    CLUSTER_AXIS,
    _make_inputs,
    _read_back,
    reference,
)
from ttnn.operations.q_stem import q_stem

SHAPES = [
    pytest.param((1, 128, 64, 2, 32, 32, 32), id="small_single_tile"),
    pytest.param((1, 256, 256, 4, 64, 32, 128), id="medium_multi_tile"),
    pytest.param((2, 128, 512, 8, 128, 64, 512), id="medium_multi_batch"),
    pytest.param((1, 640, 1536, 24, 128, 64, 512), id="large_kimi_k31_640"),
]


def _ulp_bf16(t):
    """Spacing of bf16 values at |t| (normal range): 2^(exponent - 7)."""
    a = t.abs().clamp(min=torch.finfo(torch.bfloat16).tiny)
    return torch.pow(2.0, torch.floor(torch.log2(a)) - 7)


def _metrics(got, exp):
    got, exp = got.float().flatten(), exp.float().flatten()
    _, pcc = comp_pcc(exp, got, pcc=0.0)
    _, allclose_msg = comp_allclose(exp, got)
    err = got - exp
    rel_rms = (err.pow(2).mean().sqrt() / exp.std()).item()
    ulp = (err.abs() / _ulp_bf16(exp)).float()
    mask = torch.isfinite(got) & (exp.abs() > 0.01 * exp.std())
    r = got[mask] / exp[mask]
    q = torch.quantile(r[: 1 << 20], torch.tensor([0.05, 0.5, 0.95]))
    return dict(
        pcc=float(pcc),
        max_abs=err.abs().max().item(),
        mean_abs=err.abs().mean().item(),
        rel_rms=rel_rms,
        ulp_med=ulp.median().item(),
        ulp_p99=torch.quantile(ulp[: 1 << 20], 0.99).item(),
        ratio_p5=q[0].item(),
        ratio_med=q[1].item(),
        ratio_p95=q[2].item(),
        allclose=allclose_msg,
    )


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", _MESH_PARAM, indirect=True)
@pytest.mark.parametrize("dims", SHAPES)
def test_q_stem_precision_baseline(mesh_device, dims):
    B, S, Q, Hl, Dn, Dr, N = dims
    G = tuple(mesh_device.shape)[CLUSTER_AXIS]
    x, q_b, wkv, tx, tq, tw = _make_inputs(mesh_device, B, S, Q, Hl, Dn, Dr, N)
    expected = reference(x, q_b, wkv, G)
    out = q_stem(tx, tq, tw, cluster_axis=CLUSTER_AXIS)
    pcc_min, rms_max = default_tolerance(ttnn.bfloat16, True)
    rows, cols = tuple(mesh_device.shape)
    for idx, dev in enumerate(_read_back(out, mesh_device)):
        r, c = divmod(idx, cols)
        got = ttnn.to_torch(dev).float()
        exp = expected[r, c]
        assert got.shape == exp.shape
        whole = _metrics(got, exp)
        nope = _metrics(got[..., :N], exp[..., :N])
        rope = _metrics(got[..., N:], exp[..., N:])
        for part, m in (("all", whole), ("nope", nope), ("rope", rope)):
            print(
                f"PRECISION {dims} dev{idx} {part:4s} pcc={m['pcc']:.6f} max_abs={m['max_abs']:.4g} "
                f"mean_abs={m['mean_abs']:.4g} rel_rms={m['rel_rms']:.4g} ulp_med={m['ulp_med']:.3g} "
                f"ulp_p99={m['ulp_p99']:.3g} ratio[p5,med,p95]=[{m['ratio_p5']:.4f},{m['ratio_med']:.4f},"
                f"{m['ratio_p95']:.4f}]"
            )
        assert whole["pcc"] >= pcc_min, f"dev{idx}: PCC {whole['pcc']} < {pcc_min}"
        assert whole["rel_rms"] <= rms_max, f"dev{idx}: relative RMS {whole['rel_rms']} > {rms_max}"
        # scale-bug detector: the ratio must centre on 1.0
        assert abs(whole["ratio_med"] - 1.0) < 0.01, f"dev{idx}: got/true median ratio {whole['ratio_med']}"
