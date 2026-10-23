# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 5 debug (DO NOT DELETE): fp32_dest_acc_en=False corner, deterministic inputs.

All-ones x, no gamma: every device's partial = W_local per row, the combine = W_global, inv = rsqrt(1 + eps) ~ 1, so
out == 1 everywhere. A per-row ratio map (random x) tells a wrong inv (constant per row) from a wrong normalize."""

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import (
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    pytorch_distributed_rms_norm,
)
from tests.ttnn.unit_tests.operations.distributed_rms_norm.test_distributed_rms_norm_precision_baseline import (  # noqa: F401
    LINEAR_AXES,
    mesh_device,
)
from ttnn.operations.distributed_rms_norm import distributed_rms_norm


def _cfg(fp32_acc):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_acc
    cfg.math_approx_mode = False
    return cfg


def _run(x, gamma, mesh_device, fp32_acc, dtype=ttnn.bfloat16):
    shape = tuple(x.shape[2:])
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tg = None if gamma is None else create_ttnn_gamma(gamma, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out = distributed_rms_norm(
        tx, cluster_axis=LINEAR_AXES[-1], gamma=tg, epsilon=1e-6, compute_kernel_config=_cfg(fp32_acc)
    )
    return torch.stack([ttnn.to_torch(t).float().reshape(shape) for t in ttnn.get_device_tensors(out)]).reshape(x.shape)


@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32acc", "bf16acc"])
@pytest.mark.parametrize("use_gamma", [False, True], ids=["nogamma", "gamma"])
@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 64, 128)], ids=["32x32", "64x128"])
def test_ones(shape, use_gamma, fp32_acc, mesh_device):
    rows, cols = tuple(mesh_device.shape)
    x = torch.ones((rows, cols, *shape), dtype=torch.bfloat16)
    gamma = torch.ones((rows, cols, 1, 1, 1, shape[-1]), dtype=torch.bfloat16) if use_gamma else None
    got = _run(x, gamma, mesh_device, fp32_acc)
    print("ONES", shape, use_gamma, fp32_acc, "got[0,0,0,0,:4,:4]", got[0, 0, 0, 0, :4, :4].tolist())
    print("ONES unique", torch.unique(got).tolist()[:16])
    assert torch.allclose(got, torch.ones_like(got), atol=0.02)


@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32acc", "bf16acc"])
def test_row_ratio_map(fp32_acc, mesh_device):
    torch.manual_seed(0)
    rows, cols = tuple(mesh_device.shape)
    shape = (1, 1, 64, 128)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(torch.bfloat16)
    got = _run(x, None, mesh_device, fp32_acc)
    exp = pytorch_distributed_rms_norm(x.float(), None, LINEAR_AXES[-1], 1e-6)
    ratio = got / exp
    per_row = ratio[0, 0, 0, 0]  # (64, 128)
    print("ROW median ratio", [round(v, 3) for v in per_row.median(dim=1).values.tolist()])
    print("ROW ratio spread", [round(v, 3) for v in (per_row.max(dim=1).values - per_row.min(dim=1).values).tolist()])
    assert torch.allclose(got, exp, rtol=0.05, atol=0.05)
