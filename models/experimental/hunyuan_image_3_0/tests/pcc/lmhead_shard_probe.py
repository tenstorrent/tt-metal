# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Throwaway: find the fastest config for the LM-head matmul 32x4096x133120
(bandwidth-bound: 149 GB/s, 29%% DRAM, 3%% FLOP). A/B auto vs 1D-mcast vs
DRAM-width-sharded vs act-width-sharded, with PCC vs auto."""
from __future__ import annotations

import time

import pytest
import torch
import ttnn
from models.experimental.hunyuan_image_3_0.tt import matmul_utils as mu

M, K, N = 32, 4096, 133120


def _bench(dev, fn, it=30):
    for _ in range(3):
        o = fn()
        ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(it):
        o = fn()
        ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / it * 1e6


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_shard(mesh_device):
    dev = mesh_device
    dev.enable_program_cache()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    wt = torch.randn(K, N) * 0.02
    xt = torch.randn(1, M, K) * 0.05

    # --- interleaved weight (for auto / 1D / act-sharded) ---
    w_il = ttnn.from_torch(
        wt,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
    )
    x_l1 = ttnn.from_torch(
        xt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
    )

    results = {}

    # 1) auto (baseline)
    auto = lambda: ttnn.linear(
        x_l1, w_il, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ckc
    )
    oa = ttnn.to_torch(auto(), mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:1].float()
    results["auto"] = (_bench(dev, auto), 1.0)

    def pcc(fn):
        ot = ttnn.to_torch(fn(), mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:1].float()
        return ((oa * ot).sum() / (oa.norm() * ot.norm())).item()

    # 2) 1D-mcast (my QKV-style; in0_block_w budgeted)
    grid = dev.compute_with_storage_grid_size()
    Nt, Kt = N // 32, K // 32
    ncols = next(c for c in range(min(grid.x * grid.y, Nt), 0, -1) if Nt % c == 0)
    pcn = Nt // ncols
    ibw = next(b for b in (16, 8, 4, 2, 1) if Kt % b == 0 and b * pcn <= 384)
    osw = next(w2 for w2 in (4, 3, 2, 1) if pcn % w2 == 0)
    pc_1d = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=osw,
        per_core_M=1,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    f1d = lambda: ttnn.linear(
        x_l1,
        w_il,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=ckc,
        program_config=pc_1d,
    )
    try:
        results[f"1D mcast pcn={pcn} ibw={ibw}"] = (_bench(dev, f1d), pcc(f1d))
    except Exception as e:
        results[f"1D mcast pcn={pcn} ibw={ibw}"] = (None, str(e)[:70])

    # 3) DRAM-width-sharded weight + width-sharded L1 act (the bandwidth-optimal pattern)
    try:
        w_dram = ttnn.from_torch(
            wt,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mu.dram_width_sharded_weight_mem_config(dev, K, N),
            mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
        )
        act_mc, agrid, ncores = mu.width_sharded_act_mem_config(K)
        x_ws = ttnn.to_memory_config(x_l1, act_mc)
        pc_dram = mu.decode_width_sharded_matmul_program_config(M, K, N, ncores)
        out_ws_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)

        def fdram():
            o = ttnn.linear(
                x_ws,
                w_dram,
                dtype=ttnn.bfloat16,
                program_config=pc_dram,
                compute_kernel_config=ckc,
                memory_config=out_ws_mc,
            )
            return ttnn.sharded_to_interleaved(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        results[f"DRAM-sharded ncores={ncores}"] = (_bench(dev, fdram), pcc(fdram))
    except Exception as e:
        results["DRAM-sharded"] = (None, str(e)[:90])

    for name, (us, extra) in results.items():
        if us is None:
            print(f"[shard] {name:34} FAIL {extra}")
        else:
            base = results["auto"][0]
            print(
                f"[shard] {name:34} {us:9.1f} us ({base/us:.2f}x)  PCC={extra:.5f}"
                if isinstance(extra, float)
                else f"[shard] {name:34} {us:9.1f} us"
            )
