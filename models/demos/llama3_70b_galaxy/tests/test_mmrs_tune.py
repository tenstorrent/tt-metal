# Fused matmul + reduce-scatter (ttnn.experimental.minimal_matmul_strided_reduce_scatter_async, configured the way
# the tt_dit Blackhole callers do) vs the unfused minimal_matmul + ttnn.reduce_scatter the TP32 BH prefill runs
# today, at the per-chip FF1/FF3 prefill shape: [M, 2048] @ [2048, 3584] (bf4, LoFi), reduce-scatter over the
# 4-chip cluster_axis=1 ring -> [M, 896].
#   MESH_DEVICE=TG TT_FABRIC_OVERRIDE=FABRIC_1D_RING pytest models/demos/llama3_70b_galaxy/tests/test_mmrs_tune.py -s
# env: MMRS_M (default 4096), MMRS_LINKS (default "2")
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

N_PER_TRACE, REPLAYS = 10, 5
M = int(os.environ.get("MMRS_M", "4096"))
SHAPES = {  # name: (K, N, weight dtype, fidelity)
    "ff13": (2048, 3584, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
}


def _timed(mesh, fn):
    out = fn()
    ttnn.synchronize_device(mesh)
    first = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = [fn() for _ in range(N_PER_TRACE)]
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t0) / (REPLAYS * N_PER_TRACE) * 1e6
    ttnn.release_trace(mesh, tid)
    for o in outs:
        ttnn.deallocate(o)
    return us, first


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("shape", os.environ.get("MMRS_SHAPE", "ff13").split(","))
def test_mmrs_tune(mesh_device, device_params, shape):
    from models.tt_dit.utils.matmul import get_fused_mmrs_config

    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    K, N, wdt, fid = SHAPES[shape]
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    tx, tw = torch.randn(1, 1, M, K), torch.randn(1, 1, K, N) / K**0.5
    # Same x and w on every chip: the ring sum is 4 * (x @ w), and column c keeps N slice c.
    golden = (4 * (tx @ tw))[..., : N // 4].float()
    rep = ttnn.ReplicateTensorToMesh(mesh)
    x = ttnn.from_torch(tx, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=rep)
    w = ttnn.from_torch(tw, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=wdt, mesh_mapper=rep)
    full_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    logger.info(f"MMRS {shape} M={M} K={K} N={N} device grid {grid.x}x{grid.y}")

    base_cfg = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=1,
        subblock_w=8,
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
    )

    def baseline():
        o = ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=w,
            config=base_cfg,
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        r = ttnn.reduce_scatter(
            o,
            3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_workers_per_link=4,
        )
        ttnn.deallocate(o)
        return r

    try:
        us, first = _timed(mesh, baseline)
        _, pcc = comp_pcc(golden, first.reshape(golden.shape), 0.97)
        logger.info(f"MMRS {shape} M={M} baseline minimal_matmul+reduce_scatter: {us:.1f} us pcc={pcc:.4f}")
    except Exception as e:
        logger.warning(f"MMRS {shape} baseline ERROR {str(e)[:200]}")

    rs_sems = [ttnn.create_global_semaphore(mesh, full_crs, 0) for _ in range(3)]
    barrier = ttnn.create_global_semaphore(mesh, full_crs, 0)
    slots = grid.x * grid.y
    progress = ttnn.allocate_tensor_on_device(
        ttnn.Shape([slots, slots]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(full_crs, [1, slots], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    for nl in [int(v) for v in os.environ.get("MMRS_LINKS", "2").split(",")]:
        params = get_fused_mmrs_config(M, K, N, grid, nl)
        params = dict(params)
        use_l1 = params.get("mm_window_blocks") is not None
        credits = None
        if use_l1:
            credits = ttnn.allocate_tensor_on_device(
                ttnn.Shape([slots, slots]),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                mesh,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(full_crs, [1, slots], ttnn.ShardOrientation.ROW_MAJOR),
                ),
            )
        logger.info(f"MMRS {shape} params links={nl}: { {k: v for k, v in params.items() if k != 'config'} }")

        def fused():
            _, out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
                input_tensor=x,
                weight_tensor=w,
                dim=3,
                multi_device_global_semaphore=rs_sems,
                **params,
                memory_config_mm=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
                if use_l1
                else ttnn.DRAM_MEMORY_CONFIG,
                rs_intermediate_mem_config=ttnn.DRAM_MEMORY_CONFIG,
                rs_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                cluster_axis=1,
                compute_kernel_config=ckc,
                barrier_semaphore=barrier,
                using_persistent_buffers=False,
                dtype=ttnn.bfloat8_b,
                mm_progress_counters=progress,
                mm_credit_counters=credits,
            )
            return out

        try:
            us, first = _timed(mesh, fused)
            _, pcc = comp_pcc(golden, first.reshape(golden.shape), 0.97)
            logger.info(f"MMRS {shape} M={M} fused links={nl}: {us:.1f} us pcc={pcc:.4f}")
        except Exception as e:
            logger.warning(f"MMRS {shape} fused links={nl} ERROR {str(e)[:300]}")
