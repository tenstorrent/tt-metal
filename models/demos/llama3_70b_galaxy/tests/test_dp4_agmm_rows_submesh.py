# Row all-gather fused with the matmul (ttnn.experimental.all_gather_matmul_async, dim=2) on ONE mesh column as an 8x1
# submesh (the op has no cluster_axis; every DP4 column is an independent 8-chip ring). FF1|FF3-like shape per chip:
# rows [R, 8192] bf8 local -> gathered [8R, 8192] @ [8192, 7168] bf4 LoFi. vs all_gather_async + matmul on the submesh.
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_dp4_agmm_rows_submesh.py -s     env: AGR_M (4096,16384)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H, N, TP = 8192, 7168, 8
MS = [int(m) for m in (os.environ.get("AGR_M") or "4096,16384").split(",")]
DRAM = ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_dp4_agmm_rows_submesh(mesh_device, device_params):
    sub = mesh_device.create_submeshes(ttnn.MeshShape(8, 1))[0]
    logger.info(f"AGR submesh shape {tuple(sub.shape)}")
    grid = sub.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    sems = [[ttnn.create_global_semaphore(sub, crs, 0) for _ in range(2)] for _ in range(2)]
    bars = [ttnn.create_global_semaphore(sub, crs, 0) for _ in range(2)]
    k = [0]
    lofi = ttnn.init_device_compute_kernel_config(
        sub.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    torch.manual_seed(0)
    w = torch.randn(1, 1, H, N) * 0.02
    tw = ttnn.from_torch(
        w,
        device=sub,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat4_b,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(sub),
    )

    def nxt():
        k[0] ^= 1
        return sems[k[0]], bars[k[0]]

    def timed(name, fn, M):
        try:
            o = fn()
            ttnn.synchronize_device(sub)
            for z in o if isinstance(o, (list, tuple)) else [o]:
                ttnn.deallocate(z)
            t0 = time.perf_counter()
            for _ in range(10):
                o = fn()
                for z in o if isinstance(o, (list, tuple)) else [o]:
                    ttnn.deallocate(z)
            ttnn.synchronize_device(sub)
            ms = (time.perf_counter() - t0) / 10 * 1e3
            logger.info(f"AGR M={M} {name}: {ms:.3f} ms")
            return ms
        except Exception as e:
            logger.warning(f"AGR M={M} {name} ERROR {str(e)[:500]}")
            return float("nan")

    for M in MS:
        R = M // TP
        x = torch.randn(1, 1, M, H)
        tx = ttnn.from_torch(
            x,
            device=sub,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensorToMesh(sub, dim=2),
        )
        Mt, Kt, Nt = M // 32, H // 32, N // 32

        def serial():
            s, b = nxt()
            g = ttnn.experimental.all_gather_async(
                input_tensor=tx,
                persistent_output_buffer=None,
                dim=2,
                multi_device_global_semaphore=s,
                num_links=2,
                barrier_semaphore=b,
                memory_config=DRAM,
                topology=ttnn.Topology.Ring,
            )
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=1,
                subblock_w=8,
                compute_with_storage_grid_size=grid,
            )
            o = ttnn.experimental.minimal_matmul(
                input_tensor=g,
                weight_tensor=tw,
                config=cfg,
                compute_kernel_config=lofi,
                memory_config=DRAM,
                dtype=ttnn.bfloat8_b,
            )
            return [g, o]

        timed("serial all_gather_async + minimal_matmul", serial, M)
        for gy, off_y, in0w, sbh, sbw in ((8, 8, 8, 2, 4), (8, 8, 4, 1, 4), (8, 8, 2, 1, 7)):
            per_core_M, per_core_N = Mt // gy, Nt // 8
            if Mt % gy or Nt % 8 or per_core_N % sbw or per_core_M % sbh:
                continue
            pcfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, gy),
                in0_block_w=in0w,
                out_subblock_h=sbh,
                out_subblock_w=sbw,
                per_core_M=per_core_M,
                per_core_N=per_core_N,
                transpose_mcast=False,
                fused_activation=None,
            )

            def fused():
                s, b = nxt()
                return list(
                    ttnn.experimental.all_gather_matmul_async(
                        tx,
                        tw,
                        persistent_output_buffer=None,
                        dim=2,
                        multi_device_global_semaphore=s,
                        all_gather_core_grid_offset=ttnn.CoreCoord(0, off_y),
                        barrier_semaphore=b,
                        num_links=2,
                        memory_config_ag=DRAM,
                        memory_config_mm=DRAM,
                        topology=ttnn.Topology.Ring,
                        program_config=pcfg,
                        compute_kernel_config=lofi,
                        dtype=ttnn.bfloat8_b,
                    )
                )

            ms = timed(f"fused all_gather_matmul_async grid 8x{gy} in0w{in0w} sb{sbh}x{sbw}", fused, M)
            if ms == ms:
                outs = fused()
                got = ttnn.to_torch(ttnn.get_device_tensors(outs[1])[3]).float().reshape(M, N)
                gold = x[0, 0].float() @ w[0, 0].float()
                pcc = torch.corrcoef(torch.stack([gold.flatten()[:4_000_000], got.flatten()[:4_000_000]]))[0, 1].item()
                logger.info(f"AGR M={M} fused pcc {pcc:.4f} out {tuple(outs[1].shape)}")
                for z in outs:
                    ttnn.deallocate(z)
        ttnn.deallocate(tx)
