# matmul_reduce_scatter_prefill (WO / FF2, column axis, 8 chips) vs its parts: the matmul alone (minimal_matmul, full
# grid) and the reduce-scatter alone (reduce_scatter_minimal_async of the [M, 8192] partial, bf8). Prints the fused op's
# plan. Overlap ceiling = max(matmul, RS).
#   TT_MRS_BF8_TRANSPORT=1 MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_mrs_breakdown.py -s
# env: MB_M (4096,16384)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H, TP = 8192, 8
DRAM = ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_mrs_breakdown(mesh_device, device_params):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    sems = [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(3)] for _ in range(2)]
    bars = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    k = [0]
    rep = ttnn.ReplicateTensorToMesh(mesh)
    lofi = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def timed(fn, n=8):
        for z in fn():
            ttnn.deallocate(z)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(n):
            for z in fn():
                ttnn.deallocate(z)
        ttnn.synchronize_device(mesh)
        return (time.perf_counter() - t0) / n * 1e3

    for M in [int(m) for m in (os.environ.get("MB_M") or "4096,16384").split(",")]:
        for name, K in (("wo", 1024), ("ff2", 3584)):
            ta = ttnn.from_torch(
                torch.randn(1, 1, M, K),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=DRAM,
                mesh_mapper=rep,
            )
            w = torch.randn(K, H) * 0.02
            tw = ttnn.from_torch(
                w, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=DRAM, mesh_mapper=rep
            )  # (K, N) for the fused op
            tw4 = ttnn.from_torch(
                w.reshape(1, 1, K, H),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=DRAM,
                mesh_mapper=rep,
            )
            tp = ttnn.from_torch(
                torch.randn(1, 1, M, H),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=DRAM,
                mesh_mapper=rep,
            )
            res = {}
            for hr in (None, 0, 256 * 1024):
                kw = {} if hr is None else {"l1_headroom": hr}
                res[f"mrs hr={hr}"] = timed(
                    lambda: [
                        ttnn.experimental.matmul_reduce_scatter_prefill(
                            ta,
                            tw,
                            cluster_axis=0,
                            scatter_dim=-2,
                            topology=ttnn.Topology.Ring,
                            num_links=2,
                            compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
                            memory_config=DRAM,
                            **kw,
                        )
                    ]
                )
                try:
                    logger.info(
                        f"MRSB M={M} {name} hr={hr} plan {dict(ttnn._ttnn.operations.experimental.ccl_experimental._matmul_reduce_scatter_prefill_last_plan())}"
                    )
                except Exception as e:
                    try:
                        logger.info(f"MRSB plan {dict(ttnn.experimental._matmul_reduce_scatter_prefill_last_plan())}")
                    except Exception as e2:
                        logger.info(f"MRSB plan unavailable {str(e2)[:80]}")
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=1,
                subblock_w=8,
                compute_with_storage_grid_size=grid,
            )
            res["matmul only"] = timed(
                lambda: [
                    ttnn.experimental.minimal_matmul(
                        input_tensor=ta,
                        weight_tensor=tw4,
                        config=cfg,
                        compute_kernel_config=lofi,
                        memory_config=DRAM,
                        dtype=ttnn.bfloat8_b,
                    )
                ]
            )

            def rs(wpl):
                def f():
                    i = k[0]
                    k[0] ^= 1
                    return [
                        ttnn.experimental.reduce_scatter_minimal_async(
                            tp,
                            dim=2,
                            multi_device_global_semaphore=sems[i],
                            barrier_semaphore=bars[i],
                            num_links=2,
                            memory_config=DRAM,
                            topology=ttnn.Topology.Ring,
                            cluster_axis=0,
                            **({"num_workers_per_link": wpl} if wpl else {}),
                        )
                    ]

                return f

            for wpl in (None, 4, 8):
                try:
                    res[f"RS only wpl={wpl}"] = timed(rs(wpl))
                except Exception as e:
                    logger.warning(f"MRSB RS wpl={wpl} ERROR {str(e)[:200]}")
            logger.info(f"MRSB M={M} {name} " + " ".join(f"[{a}]={b:.3f}" for a, b in res.items()))
            for z in (ta, tw, tw4, tp):
                ttnn.deallocate(z)
