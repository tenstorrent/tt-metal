# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH microbenchmark: is the fused all_gather_minimal_matmul_async (AGMM) in-projection AG-bound or
matmul-bound at the Qwen3.6-27B TP=4 prefill shapes? Wall-clock over N iterations on the (1,4) mesh.

  MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_agmm_bench_scratch.py -s
"""
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.tt_transformers.tt.ccl import TT_CCL

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2, "fabric_config": ttnn.FabricConfig.FABRIC_1D}]
S, K_TOTAL = 2048, 5120
N_LOCALS = {"gdn_in": 4120, "attn_in": 3584, "mlp_gateup": 8704}


def _time(fn, iters=10):
    fn()
    ttnn.synchronize_device(fn.mesh)
    t0 = time.time()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(fn.mesh)
    return (time.time() - t0) / iters * 1e6


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_agmm_bench(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    tt_ccl = TT_CCL(mesh)
    topo = ttnn.Topology.Ring
    ckc = tpc.COMPUTE_HIFI2
    x = torch.randn(1, 1, S, K_TOTAL, dtype=torch.bfloat16)
    x_sh = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3)
    )  # K-sharded [1,1,S,1280]
    x_full = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )
    results = {}
    for name, n_local in N_LOCALS.items():
        w = torch.randn(K_TOTAL, n_local * 4, dtype=torch.bfloat16) * 0.02
        w_sh = ttnn.from_torch(
            w,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )  # [5120, n_local] per device

        def agmm(links, grid):
            def f():
                x4 = x_sh
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=4 if name != "mlp_gateup" else 8,
                    K_block_size=tpc.agmm_k_block_size(K_TOTAL // 4),
                    N_block_size=8 if name != "mlp_gateup" else 16,
                    subblock_h=1,
                    subblock_w=4,
                    compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
                )
                out = ttnn.experimental.all_gather_minimal_matmul_async(
                    input_tensor=x4,
                    weight_tensor=w_sh,
                    config=cfg,
                    compute_kernel_config=ckc,
                    multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(1),
                    num_links=links,
                    topology=topo,
                    cluster_axis=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                    force_transpose=True,
                    num_workers_per_link=grid[0] // links,
                    num_buffers_per_channel=8,
                )[0]
                ttnn.deallocate(out)

            f.mesh = mesh
            return f

        def plain_mm(grid_cols):
            pc = tpc.create_prefill_mlp_matmul_program_config(S, K_TOTAL, n_local, max_cols=grid_cols)

            def f():
                out = ttnn.linear(
                    x_full, w_sh, compute_kernel_config=ckc, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ttnn.deallocate(out)

            f.mesh = mesh
            return f

        def plain_ag(links):
            def f():
                out = ttnn.experimental.all_gather_async(
                    x_sh,
                    dim=3,
                    multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(1),
                    num_links=links,
                    topology=topo,
                    cluster_axis=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(out)

            f.mesh = mesh
            return f

        r = {}
        for links, grid in [(2, (8, 9)), (1, (8, 9)), (2, (8, 8)), (2, (10, 8)), (1, (11, 8))]:
            try:
                r[f"agmm_l{links}_{grid[0]}x{grid[1]}"] = _time(agmm(links, grid))
            except Exception as e:  # noqa: BLE001
                r[f"agmm_l{links}_{grid[0]}x{grid[1]}"] = f"ERR {str(e).splitlines()[0][:80]}"
        for cols in (8, 11):
            try:
                r[f"plain_mm_{cols}cols"] = _time(plain_mm(cols))
            except Exception as e:  # noqa: BLE001
                r[f"plain_mm_{cols}cols"] = f"ERR {str(e).splitlines()[0][:80]}"
        for links in (1, 2):
            try:
                r[f"plain_ag_l{links}"] = _time(plain_ag(links))
            except Exception as e:  # noqa: BLE001
                r[f"plain_ag_l{links}"] = f"ERR {str(e).splitlines()[0][:80]}"
        results[name] = r
        ttnn.deallocate(w_sh)
        for k, v in r.items():
            logger.info(f"AGMM_BENCH {name:12s} {k:22s} {v if isinstance(v, str) else f'{v:8.1f} us'}")
        gflop = 2 * S * K_TOTAL * n_local / 1e9
        for k in ("agmm_l2_8x9", "plain_mm_8cols", "plain_mm_11cols"):
            v = r.get(k)
            if isinstance(v, float):
                logger.info(f"AGMM_BENCH {name:12s} {k:22s} -> {gflop / (v / 1e6) / 1e3:6.1f} TFLOPS/device")
    print("AGMM_BENCH_DONE")
