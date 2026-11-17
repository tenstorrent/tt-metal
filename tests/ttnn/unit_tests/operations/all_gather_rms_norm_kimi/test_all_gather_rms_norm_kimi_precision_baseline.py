# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for all_gather_rms_norm_kimi (verifier, Phase 0).

Runs on the golden suite's 4-device ring (FABRIC_2D_TORUS_XY). For every device's full gathered output it measures
PCC (comp_pcc), max / mean abs error (comp_allclose), relative RMS error (RMS error / RMS of the reference, the
golden helpers' definition), bf16 ULP error against an fp32 (un-rounded) reference, and the got/true ratio
spread (the scale-bug detector: a tight cluster around a non-1.0 constant means a structural scale error, a broad
spread centred on 1.0 is rounding noise). Gates on PCC and relative RMS together at helpers.TOLERANCES.
Run with -s to see the table.
"""

from __future__ import annotations

import pytest
import torch
import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

from eval.golden_tests.all_gather_rms_norm_kimi.conftest import _FabricMeshCache, focus_mesh_or_skip
from eval.golden_tests.all_gather_rms_norm_kimi.helpers import (
    TOLERANCES,
    create_ttnn_gamma,
    create_ttnn_input_tensor,
)
from ttnn.operations.all_gather_rms_norm_kimi import all_gather_rms_norm_kimi, default_compute_kernel_config

CLUSTER_AXIS = 1
EPS = 1e-6


@pytest.fixture(scope="module")
def mesh_device():
    cache = _FabricMeshCache()
    try:
        yield cache.get(focus_mesh_or_skip())
    finally:
        cache.close()


def _reference_fp32(x, gamma):
    """(R, C, 1, 1, M, W_local) bf16 slices -> fp32 normalized full rows per device, NOT rounded to bf16."""
    rows, cols = x.shape[0], x.shape[1]
    out = []
    for r in range(rows):
        full = torch.cat([x[r, c].float() for c in range(cols)], dim=-1)
        normed = torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=gamma.float(), eps=EPS)
        out.append([normed] * cols)
    return out


def _bf16_ulp(e):
    """Size of one bf16 ULP at each reference value (7 explicit mantissa bits)."""
    a = e.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return torch.pow(2.0, torch.floor(torch.log2(a)) - 7)


SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 128, 256), id="medium"),
    pytest.param((1, 1, 640, 384), id="kimi_mla_q_a"),  # the perf-focus cell
    pytest.param((1, 1, 2048, 384), id="tall_block_h_3"),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_precision_baseline(mesh_device, shape):
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(1234)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(torch.bfloat16)
    gamma = (torch.rand((cols * shape[-1],), dtype=torch.float32) + 0.5).to(torch.bfloat16)

    tt_x = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_g = create_ttnn_gamma(gamma, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = all_gather_rms_norm_kimi(
        tt_x,
        gamma=tt_g,
        cluster_axis=CLUSTER_AXIS,
        epsilon=EPS,
        topology=ttnn.Topology.Ring,
        num_links=2,
        compute_kernel_config=default_compute_kernel_config(),
    )
    ref = _reference_fp32(x, gamma)
    pcc_target, rms_target = TOLERANCES[(ttnn.bfloat16, True)]

    worst = None
    for idx, dev in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        got = ttnn.to_torch(dev).float()
        exp32 = ref[r][c]
        exp16 = exp32.to(torch.bfloat16).float()
        _, pcc = comp_pcc(exp16, got)
        _, allclose_msg = comp_allclose(exp16, got)
        err = (got - exp32).abs()
        max_abs, mean_abs = err.max().item(), err.mean().item()
        rel_rms = ((got - exp16) ** 2).mean().sqrt().item() / max((exp16**2).mean().sqrt().item(), 1e-30)
        ulp = err / _bf16_ulp(exp32)
        mask = exp32.abs() > 1e-3
        ratio = got[mask] / exp32[mask]
        q = torch.quantile(ratio.double()[:1_000_000], torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
        row = dict(
            dev=(r, c),
            pcc=float(pcc),
            max_abs=max_abs,
            mean_abs=mean_abs,
            rel_rms=rel_rms,
            max_ulp=ulp.max().item(),
            mean_ulp=ulp.mean().item(),
            ratio_p5=q[0].item(),
            ratio_med=q[1].item(),
            ratio_p95=q[2].item(),
        )
        if worst is None or row["pcc"] < worst["pcc"]:
            worst = row
        assert row["pcc"] >= pcc_target and row["rel_rms"] <= rms_target, (row, allclose_msg)
        # scale-bug detector: the median got/true ratio must sit on 1.0
        assert abs(row["ratio_med"] - 1.0) < 1e-2, row
    print(
        f"\nPRECISION {shape}: worst device {worst['dev']} PCC={worst['pcc']:.6f} max_abs={worst['max_abs']:.4g} "
        f"mean_abs={worst['mean_abs']:.4g} rel_rms={worst['rel_rms']:.4g} max_ulp={worst['max_ulp']:.2f} "
        f"mean_ulp={worst['mean_ulp']:.3f} ratio p5/med/p95={worst['ratio_p5']:.4f}/{worst['ratio_med']:.5f}/"
        f"{worst['ratio_p95']:.4f}"
    )
