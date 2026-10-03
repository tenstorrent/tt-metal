# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe 6: DRAM-sharded matmuls for the o-projection chain (nlp_concat out -> wo_a -> wo_b). Prints 'P6 ...'."""

import math

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_attn_probe_matmul import chain_ms

T = 4


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
def test_probe6(mesh_device):
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
    nb = md.dram_grid_size().x
    print(f"P6 dram banks {nb}", flush=True)

    def run(name, fn):
        try:
            print(f"P6 {name:50s} {chain_ms(md, fn) * 1e3:7.1f} us", flush=True)
        except Exception as e:
            print(f"P6 {name:50s} FAIL {str(e).splitlines()[0][:150]!r}", flush=True)

    def dram_sharded_w(w):  # w [K, N] torch
        K, N = w.shape
        npad = math.ceil(N / (32 * nb)) * 32 * nb
        spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))}),
            (K, npad // nb),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)
        return ttnn.from_torch(
            w.reshape(1, 1, K, N).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mc,
            mesh_mapper=rep,
        )

    def in0_cfg(K, ncores):
        return ttnn.create_sharded_memory_config(
            shape=(32, K // ncores),
            core_grid=ttnn.CoreGrid(x=ncores, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def out_cfg(N, ncores):
        return ttnn.create_sharded_memory_config(
            shape=(32, N // ncores),
            core_grid=ttnn.CoreGrid(x=ncores, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def pc(K, N, ncores, ibw=None):
        ibw = ibw or max(d for d in range(1, K // 32 // ncores + 1) if (K // 32 // ncores) % d == 0 and d <= 16)
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=ibw, per_core_M=1, per_core_N=math.ceil(N / 32 / ncores), fused_activation=None
        )

    for K, N, nm in ((4096, 1024, "wo_a"), (1024, 5120, "wo_b"), (1280, 4096, "wq_b"), (5120, 1792, "wqkv")):
        W = torch.randn(K, N) * 0.02
        x = torch.randn(1, 1, T, K).to(torch.bfloat16)
        w_i = ttnn.from_torch(
            W.reshape(1, 1, K, N).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )
        x_i = ttnn.from_torch(x, device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
        ref = ttnn.to_torch(ttnn.get_device_tensors(ttnn.linear(x_i, w_i, compute_kernel_config=ckc))[0]).float()[
            0, 0, :T
        ]
        w_s = dram_sharded_w(W)
        for ncores in (8, 16, 32, 64):
            if K // 32 % ncores or (N // 32) % ncores and N // 32 // ncores == 0:
                continue
            try:
                xs = ttnn.to_memory_config(x_i, in0_cfg(K, ncores))
                oc = out_cfg(math.ceil(N / (32 * ncores)) * 32 * ncores, ncores)
                f = lambda: ttnn.linear(
                    xs, w_s, program_config=pc(K, N, ncores), memory_config=oc, compute_kernel_config=ckc
                )
                o = f()
                got = ttnn.to_torch(
                    ttnn.get_device_tensors(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG))[0]
                ).float()[0, 0, :T, :N]
                print(
                    f"P6 {nm} dram-sharded cores={ncores}: pcc vs interleaved {pcc(got, ref):.5f}  {chain_ms(md, f) * 1e3:.1f} us   (interleaved default {chain_ms(md, lambda: ttnn.linear(x_i, w_i, compute_kernel_config=ckc)) * 1e3:.1f} us)",
                    flush=True,
                )
            except Exception as e:
                print(f"P6 {nm} dram-sharded cores={ncores}: FAIL {str(e).splitlines()[0][:150]!r}", flush=True)
