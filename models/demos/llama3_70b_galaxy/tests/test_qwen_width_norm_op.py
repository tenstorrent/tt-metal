# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""custom_ops.qwen_width_norm against an fp32 host RMSNorm and against the stock 2-slice form it replaces
(qwen_cp_prefill.local_norm with QWEN_CP_WIDTH_NORM=0), on the layout-B residual [1, 1, S/32, 5120] of each CP
bucket, bf16 and bf8 inputs.

Checks: every output finite; relative L2 error vs the fp32 host result no worse than the stock form's (+1%);
every device's output equal (replicated input) and a second call on a fresh output equal to the first (program
cache hit path). Not bit-identical to the stock form by design (fp32 partials, HiFi4).

Run on the Wormhole Galaxy (8x4 mesh):
    pytest models/demos/llama3_70b_galaxy/tests/test_qwen_width_norm_op.py -s
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama3_70b_galaxy.tt import qwen_cp_prefill as cp
from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_width_norm import qwen_width_norm

EPS = 1e-6


def _replicated(mesh_device, t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _devices(t):
    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "worker_l1_size": 1345000}], indirect=True
)  # production dispatch / L1
@pytest.mark.parametrize("x_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
@pytest.mark.parametrize("seq_len", [16384, 24576, 32768, 40960])
def test_qwen_width_norm(mesh_device, seq_len, x_dtype, monkeypatch):
    torch.manual_seed(0)
    rows = seq_len // 32
    xh = torch.randn(1, 1, rows, 5120)
    xh[..., torch.randperm(5120)[:16]] *= 30.0  # a few large channels, as in real residuals
    gh = (1 + 0.2 * torch.randn(5120)).to(torch.bfloat16)
    x = _replicated(mesh_device, xh.to(torch.bfloat16), x_dtype)
    g = _replicated(mesh_device, gh.reshape(1, 1, 160, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    xf = _devices(x)[0]
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * gh.float()

    dn = SimpleNamespace(norm=SimpleNamespace(weight=g, eps=EPS))
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    with monkeypatch.context() as mp:
        mp.setenv("QWEN_CP_WIDTH_NORM", "0")  # the stock 2-slice norm, restored to the caller's value afterwards
        stock = _devices(cp.local_norm(dn, x, ckc))[0]

    o1 = qwen_width_norm(x, g, eps=EPS, fidelity=ttnn.MathFidelity.HiFi4)
    o2 = qwen_width_norm(x, g, eps=EPS, fidelity=ttnn.MathFidelity.HiFi4)
    d1, d2 = _devices(o1), _devices(o2)
    y = d1[0]
    err_new = float((y - ref).norm() / ref.norm())
    err_stock = float((stock - ref).norm() / ref.norm())
    logger.info(f"S={seq_len} {x_dtype}: rel_l2 vs fp32 width_norm {err_new:.4e}, stock 2-slice {err_stock:.4e}")
    assert torch.isfinite(y).all()
    assert err_new <= err_stock * 1.01, (err_new, err_stock)
    assert all(torch.equal(a, d1[0]) for a in d1), "devices differ"
    assert all(torch.equal(a, b) for a, b in zip(d1, d2)), "second call differs"
