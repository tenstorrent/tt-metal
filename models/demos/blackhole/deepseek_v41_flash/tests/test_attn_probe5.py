# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe 5: per-user rotation matrix rope as ONE batched matmul: x [1,T,32,512] @ R [1,T,512,512]. Prints 'P5 name us'."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_attn_probe_matmul import chain_ms
from models.demos.blackhole.deepseek_v41_flash.tt import attention as A

T, H, D = 4, 8, 512


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


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
def test_probe5(mesh_device):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    torch.manual_seed(0)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    ckc2 = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    freqs = torch.polar(torch.ones(256, 32), torch.rand(256, 32))
    cos, sin = A.rope_tables(freqs)
    pos = torch.tensor([3, 17, 40, 99])
    R = torch.zeros(T, 1, D, D)
    for t in range(T):
        R[t, 0] = torch.eye(D)
        for m in range(32):
            i = D - D - 64 + 2 * m
            c, s = cos[pos[t], 2 * m], sin[pos[t], 2 * m]
            R[t, 0, i, i], R[t, 0, i + 1, i + 1] = c, c
            R[t, 0, i + 1, i], R[t, 0, i, i + 1] = -s, s
    x_h = torch.randn(1, T, 8, D).to(torch.bfloat16)
    # torch reference: x * C + (x@P) * S
    P = A.full_pair_swap()
    c = torch.ones(T, D)
    s = torch.zeros(T, D)
    c[:, D - 64 :] = cos[pos]
    s[:, D - 64 :] = sin[pos]
    ref = x_h[0].float() * c[:, None] + (x_h[0].float() @ P) * s[:, None]
    up = lambda t_: ttnn.from_torch(
        t_.to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    x = up(x_h)
    Rt = up(R.reshape(1, T, D, D)[:, :, :, :] if False else R.reshape(T, 1, D, D).reshape(1, T, D, D))
    out = ttnn.matmul(x, Rt, compute_kernel_config=ckc)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, :, :8]
    print(f"P5 correctness pcc {pcc(got, ref):.6f} shape {tuple(out.shape)}", flush=True)

    def run(name, fn):
        try:
            print(f"P5 {name:48s} {chain_ms(md, fn) * 1e3:7.1f} us", flush=True)
        except Exception as e:
            print(f"P5 {name:48s} FAIL {str(e).splitlines()[0][:110]!r}", flush=True)

    run("batched matmul default HiFi4", lambda: ttnn.matmul(x, Rt, compute_kernel_config=ckc))
    run("batched matmul default HiFi2", lambda: ttnn.matmul(x, Rt, compute_kernel_config=ckc2))
    for y, xx in ((1, 4), (2, 8), (4, 8), (8, 8), (4, 4)):
        run(
            f"batched matmul core_grid y={y} x={xx}",
            lambda: ttnn.matmul(x, Rt, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=y, x=xx)),
        )
    Rb = ttnn.from_torch(
        R.reshape(1, T, D, D).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=rep,
    )
    run("bfp8 R default", lambda: ttnn.matmul(x, Rb, compute_kernel_config=ckc))
    # current approach for reference
    Ch = up(c.reshape(1, T, 1, D))
    Sh = up(s.reshape(1, T, 1, D))
    Pf = up(P.reshape(1, 1, D, D))
    run(
        "current: mul + P-mm + addcmul",
        lambda: ttnn.addcmul(
            ttnn.multiply(x, Ch), ttnn.linear(x, Pf, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=1, x=8)), Sh
        ),
    )
    Cf = up(c.reshape(1, T, 1, D).expand(1, T, 32, D).contiguous())
    Sf = up(s.reshape(1, T, 1, D).expand(1, T, 32, D).contiguous())
    run(
        "full tables: mul + P-mm + addcmul",
        lambda: ttnn.addcmul(
            ttnn.multiply(x, Cf), ttnn.linear(x, Pf, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=1, x=8)), Sf
        ),
    )
    run("only multiply bcast", lambda: ttnn.multiply(x, Ch))
    run("only multiply full", lambda: ttnn.multiply(x, Cf))
