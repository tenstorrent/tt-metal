# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe 3: variants of the o @ P (pair swap) matmul on the SDPA output [1,T,8(32),512]. Prints 'P3 name us'."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_attn_probe_matmul import chain_ms
from models.demos.blackhole.deepseek_v41_flash.tt import attention as A


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
def test_probe3(mesh_device):
    md = mesh_device
    T = 4
    rep = ttnn.ReplicateTensorToMesh(md)
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
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    P = ttnn.from_torch(
        A.full_pair_swap().reshape(1, 1, 512, 512).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=rep,
    )
    o = ttnn.from_torch(
        torch.randn(1, T, 8, 512).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=rep,
    )

    def run(name, fn):
        try:
            print(f"P3 {name:48s} {chain_ms(md, fn) * 1e3:7.1f} us", flush=True)
        except Exception as e:
            print(f"P3 {name:48s} FAIL {str(e).splitlines()[0][:110]!r}", flush=True)

    run("default", lambda: ttnn.linear(o, P, compute_kernel_config=ckc))
    for y, x in ((1, 8), (2, 8), (4, 8), (1, 4), (4, 4), (4, 2), (1, 16 if False else 2)):
        run(
            f"core_grid y={y} x={x}",
            lambda: ttnn.linear(o, P, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=y, x=x)),
        )
    run("default HiFi2", lambda: ttnn.linear(o, P, compute_kernel_config=ckc2))
    run("matmul (not linear) default", lambda: ttnn.matmul(o, P, compute_kernel_config=ckc))
    # 2D view [1,1,T*32,512] (padded reshape)
    try:
        o2 = ttnn.reshape(o, (1, 1, T * 32, 512), (1, 1, T * 32, 512))
        run("2D view default", lambda: ttnn.linear(o2, P, compute_kernel_config=ckc))
        for y, x in ((1, 8), (2, 8), (4, 8), (4, 4)):
            run(
                f"2D view core_grid y={y} x={x}",
                lambda: ttnn.linear(o2, P, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=y, x=x)),
            )
    except Exception as e:
        print("P3 2D view FAIL", str(e).splitlines()[0][:150])
    # explicit 1D program configs, M = 4 tiles
    for pcn, gx, gy in ((2, 8, 1), (1, 8, 2), (1, 16 if False else 8, 2), (4, 4, 1)):
        for pcm in (4,):
            try:
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                    in0_block_w=4,
                    out_subblock_h=1,
                    out_subblock_w=min(pcn, 2),
                    per_core_M=pcm,
                    per_core_N=pcn,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                run(
                    f"1D pcm={pcm} pcn={pcn} grid={gx}x{gy}",
                    lambda: ttnn.linear(o2, P, compute_kernel_config=ckc, program_config=pc),
                )
            except Exception as e:
                print("P3 1D cfg FAIL", str(e).splitlines()[0][:150])
    # alternative: heads-as-batch layout  [1,8,T,512] @ P  (1 tile-row per head)
    oh = ttnn.transpose(o, 1, 2)
    run("transposed [1,8,T,512] default", lambda: ttnn.linear(oh, P, compute_kernel_config=ckc))
    run(
        "transposed [1,8,T,512] cg 2x8",
        lambda: ttnn.linear(oh, P, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=2, x=8)),
    )
    # alternative without the matmul: shuffle the pair swap with elementwise ops on a [..,32,2] view? (reshape to RM)
    orm = ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT)
    run("to_layout RM", lambda: ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT))
