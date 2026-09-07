# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: correctness of all_gather_minimal_matmul_async at the GDN/attn OUT-projection shape
(x K-sharded [1,1,2048,1536] x weight [6144, 1280] col-sharded) vs torch, for several block/grid configs,
next to the known-good in-projection shape. Also times each variant."""
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.tt_transformers.tt.ccl import TT_CCL

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2, "fabric_config": ttnn.FabricConfig.FABRIC_1D}]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_agmm_outproj(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    tt_ccl = TT_CCL(mesh)
    topo = ttnn.Topology.Ring
    comp = ttnn.ConcatMeshToTensor(mesh, dim=3)
    S = 2048
    cases = {
        "in_proj_like": (5120, 4120 * 4),  # K_total, N_total (per-device N = 4120)
        "out_proj": (6144, 1280 * 4),  # per-device N = 1280
    }
    for name, (K, N) in cases.items():
        torch.manual_seed(1)
        x = torch.randn(1, 1, S, K, dtype=torch.bfloat16)
        w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
        ref = x.float() @ w.float()  # [1,1,S,N]
        x_sh = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )
        w_sh = ttnn.from_torch(
            w,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        n_local_tiles = (N // 4) // 32
        variants = []
        for grid, n_block, sub_w, transpose in [
            ((8, 9), 8, 4, True),
            ((8, 9), min(8, -(-n_local_tiles // 9)), 1, True),
            ((8, 8), min(8, n_local_tiles // 8), 1, True),
            ((10, 8), min(8, n_local_tiles // 8), 1, True),
            ((8, 9), 8, 4, False),
        ]:
            sub_w = max(d for d in (4, 2, 1) if n_block % d == 0)
            variants.append((grid, n_block, sub_w, transpose))
        for grid, n_block, sub_w, transpose in variants:
            label = f"{name} grid{grid[0]}x{grid[1]} nblk{n_block} subw{sub_w} T{int(transpose)}"
            try:
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=4,
                    K_block_size=tpc.agmm_k_block_size(K // 4),
                    N_block_size=n_block,
                    subblock_h=1,
                    subblock_w=sub_w,
                    compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
                )
                links = 2 if grid[0] % 2 == 0 else 1

                def run():
                    return ttnn.experimental.all_gather_minimal_matmul_async(
                        input_tensor=x_sh,
                        weight_tensor=w_sh,
                        config=cfg,
                        compute_kernel_config=tpc.COMPUTE_HIFI2,
                        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(1),
                        num_links=links,
                        topology=topo,
                        cluster_axis=1,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16,
                        force_transpose=transpose,
                        num_workers_per_link=grid[0] // links,
                        num_buffers_per_channel=8,
                    )[0]

                out = run()
                ttnn.synchronize_device(mesh)
                t0 = time.time()
                for _ in range(5):
                    o2 = run()
                    ttnn.deallocate(o2)
                ttnn.synchronize_device(mesh)
                us = (time.time() - t0) / 5 * 1e6
                ot = ttnn.to_torch(out, mesh_composer=comp).float()
                _, p = comp_pcc(ref, ot, 0.99)
                logger.info(
                    f"AGMM_OUT {label}: PCC={p} max|d|={float((ref - ot).abs().max()):.3f} max|out|={float(ot.abs().max()):.3g} {us:.0f} us"
                )
                ttnn.deallocate(out)
            except Exception as e:  # noqa: BLE001
                logger.info(f"AGMM_OUT {label}: ERR {str(e).splitlines()[0][:140]}")
        # plain AG + 2D matmul reference path (ag_mm)
        t0 = time.time()
        out = tpc.all_gather_then_matmul_prefill(x_sh, w_sh, tt_ccl, tpc.COMPUTE_HIFI2, topo)
        ttnn.synchronize_device(mesh)
        ot = ttnn.to_torch(out, mesh_composer=comp).float()
        _, p = comp_pcc(ref, ot, 0.99)
        t1 = time.time()
        for _ in range(5):
            o2 = tpc.all_gather_then_matmul_prefill(x_sh, w_sh, tt_ccl, tpc.COMPUTE_HIFI2, topo)
            ttnn.deallocate(o2)
        ttnn.synchronize_device(mesh)
        logger.info(
            f"AGMM_OUT {name} ag_mm (plain AG + 2D mm): PCC={p} max|d|={float((ref - ot).abs().max()):.3f} {(time.time() - t1) / 5 * 1e6:.0f} us"
        )
        ttnn.deallocate(x_sh)
        ttnn.deallocate(w_sh)
    print("AGMM_OUT_DONE")
