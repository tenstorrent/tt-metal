# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for all_to_all_matmul (Phase 0: bf16 x bf16, fp32 DEST, HiFi4, DRAM out).

Measures, over every device of the 4-device ring: PCC (comp_pcc), max / mean abs error (comp_allclose + torch),
relative RMS (RMS error / std of the reference, as eval.metrics.check_output), ULP p99 at bf16, and the got/true
ratio spread (median and p5..p95 of actual/expected over finite, non-tiny-reference elements): a tight cluster
around a constant != 1 would be a scale bug, a broad spread around 1.0 is rounding noise.

Gates on PCC and relative RMS together at DEFAULT_TOLERANCES[(bfloat16, True)] = (0.995, 0.04). Run with -s to see
the table rows (the verification report records them).
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.all_to_all_matmul.helpers import (
    create_ttnn_input_tensor,
    pytorch_all_to_all_matmul,
    read_back,
    replicate,
)
from eval.metrics import compute_metrics_torch, default_tolerance
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.all_to_all_matmul import all_to_all_matmul, default_compute_kernel_config

CLUSTER_AXIS = 1

SHAPES = [
    pytest.param((1, 4, 128, 64), (1, 16, 64, 128), id="small"),
    pytest.param((1, 8, 256, 128), (1, 32, 128, 256), id="medium"),
    pytest.param((1, 24, 640, 128), (1, 96, 128, 512), id="kimi_k31"),
    pytest.param((1, 4, 128, 4096), (1, 16, 4096, 128), id="deep_k"),
]


@pytest.mark.parametrize("x_shape,w_shape", SHAPES)
def test_precision_baseline(mesh_device, x_shape, w_shape):
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(1234)
    x = torch.randn((rows, cols, *x_shape), dtype=torch.float32).to(torch.bfloat16)
    w = (torch.randn(w_shape, dtype=torch.float32) / w_shape[-2] ** 0.5).to(torch.bfloat16)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tw = create_ttnn_input_tensor(replicate(w, mesh_device), mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out = all_to_all_matmul(tx, tw, cluster_axis=CLUSTER_AXIS, compute_kernel_config=default_compute_kernel_config())
    got = read_back(out, (rows, cols)).to(torch.float32)
    want = pytorch_all_to_all_matmul(x, w, CLUSTER_AXIS)
    assert got.shape == want.shape

    _, pcc = comp_pcc(want, got)
    allclose_msg = comp_allclose(want, got)[1]
    abs_err = (got - want).abs()
    m = compute_metrics_torch(got, want, readback_dtype=torch.bfloat16)

    # got/true ratio spread (scale-bug detector)
    mask = torch.isfinite(got) & (want.abs() > 1e-2 * want.std())
    r = got[mask] / want[mask]
    qs = torch.quantile(r[: 1 << 20].double(), torch.tensor([0.05, 0.5, 0.95], dtype=torch.double)).tolist()

    print(
        f"\nPRECISION {x_shape}x{w_shape}: pcc={float(pcc):.7f} max_abs={abs_err.max().item():.5f} "
        f"mean_abs={abs_err.mean().item():.6f} rel_rms={m.rms:.5f} ulp_p99={m.ulp_p99:.1f} "
        f"ratio p5/median/p95={qs[0]:.4f}/{qs[1]:.4f}/{qs[2]:.4f} ({allclose_msg})"
    )

    pcc_min, rms_max = default_tolerance(ttnn.bfloat16, True)
    assert float(pcc) >= pcc_min, f"PCC {pcc} < {pcc_min}"
    assert m.rms <= rms_max, f"relative RMS {m.rms} > {rms_max}"
    assert abs(qs[1] - 1.0) < 1e-2, f"median got/true ratio {qs[1]} is off 1.0: scale bug"
