# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe 7: rms_norm kernel-config variants (time + error vs fp32 torch). Prints 'P7 ...'."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_attn_probe_matmul import chain_ms

T = 4


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_probe7(mesh_device):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    torch.manual_seed(0)
    mk = lambda f, a, fp: ttnn.init_device_compute_kernel_config(
        md.arch(), math_fidelity=f, math_approx_mode=a, fp32_dest_acc_en=fp, packer_l1_acc=False
    )
    cfgs = {
        "default": None,
        "HiFi4+fp32": mk(ttnn.MathFidelity.HiFi4, False, True),
        "HiFi2+fp32": mk(ttnn.MathFidelity.HiFi2, False, True),
        "HiFi4": mk(ttnn.MathFidelity.HiFi4, False, False),
        "HiFi2": mk(ttnn.MathFidelity.HiFi2, False, False),
        "LoFi": mk(ttnn.MathFidelity.LoFi, True, False),
    }
    for D in (1280, 512):
        x = (torch.randn(1, 1, T, D) * 3.0 + torch.randn(1, 1, 1, D)).to(torch.bfloat16)
        g = (torch.rand(D) + 0.5).to(torch.bfloat16)
        xf = x.float()
        ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-20) * g.float()
        xd = ttnn.from_torch(x, device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
        gt = ttnn.from_torch(
            g.reshape(1, 1, 1, D), device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep
        )
        gr = ttnn.from_torch(
            g.reshape(1, 1, D // 32, 32), device=md, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
        )
        for gn, gamma in (("tile", gt), ("rm", gr)):
            for cn, ck in cfgs.items():
                kw = {} if ck is None else {"compute_kernel_config": ck}
                try:
                    o = ttnn.rms_norm(xd, weight=gamma, epsilon=1e-20, **kw)
                    got = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float()[0, 0, :T]
                    err = float((got - ref[0, 0]).norm() / ref.norm())
                    print(
                        f"P7 D={D} gamma={gn:4s} {cn:11s} rel_err {err:.5f}  {chain_ms(md, lambda: ttnn.rms_norm(xd, weight=gamma, epsilon=1e-20, **kw)) * 1e3:6.1f} us",
                        flush=True,
                    )
                except Exception as e:
                    print(f"P7 D={D} gamma={gn} {cn}: FAIL {str(e).splitlines()[0][:100]!r}", flush=True)
