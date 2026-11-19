# Fused all-gather + minimal_matmul (ttnn.experimental.all_gather_minimal_matmul_async) vs the unfused
# all_gather + minimal_matmul the TP32 BH prefill runs today, at the per-chip prefill shapes (cluster_axis=1, 4-chip ring):
#   ff2:  gather [M, 896] -> [M, 3584], @ w2 [3584, 2048] (bf8, HiFi2)
#   ff13: gather [M, 2048] -> [M, 8192], @ [w1|w3] [8192, 2x896] (bf4, LoFi) with fused SwiGLU -> [M, 896]
#         (replaces FF1 + FF3 matmuls, their two reduce-scatters and the silu*mul)
#   MESH_DEVICE=TG TT_FABRIC_OVERRIDE=FABRIC_1D_RING pytest models/demos/llama3_70b_galaxy/tests/test_agmm_tune.py -s
# env: AGMM_M (default 4096), AGMM_TOPO ("ring"), AGMM_LINKS ("2")
import itertools
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

N_PER_TRACE, REPLAYS = 10, 5
M = int(os.environ.get("AGMM_M", "4096"))


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


def _mm_cfg(gx, gy, mb, kb, nb, sh, sw):
    return ttnn.MinimalMatmulConfig(
        M_block_size=mb,
        K_block_size=kb,
        N_block_size=nb,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
    )


def _interleave_gate_up(wg, wu):
    # [K, N] gate and up -> [K, 2N] with tile pairs [gate_t0, up_t0, gate_t1, up_t1, ...]
    K, N = wg.shape
    g = wg.reshape(K, N // 32, 1, 32)
    u = wu.reshape(K, N // 32, 1, 32)
    return torch.cat([g, u], dim=2).reshape(K, 2 * N)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("shape", ["ff2", "ff13"])
def test_agmm_tune(mesh_device, device_params, shape):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    if shape == "ff2":
        K_local, N, wdt, fid, swiglu = 896, 2048, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, False
    else:
        K_local, N, wdt, fid, swiglu = 2048, 2 * 896, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True
    K = 4 * K_local
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    tx = torch.randn(1, 1, M, K)
    if swiglu:
        wg, wu = torch.randn(K, N // 2) / K**0.5, torch.randn(K, N // 2) / K**0.5
        tw = _interleave_gate_up(wg, wu)
        golden = (torch.nn.functional.silu(tx[0, 0] @ wg) * (tx[0, 0] @ wu)).float()
    else:
        tw = torch.randn(K, N) / K**0.5
        golden = (tx[0, 0] @ tw).float()
    x = ttnn.from_torch(
        tx,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=tuple(mesh.shape)),
    )
    w = ttnn.from_torch(
        tw.reshape(1, 1, K, N),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=wdt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    full_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh, full_crs, 0) for _ in range(2)]

    # Baseline: unfused all_gather + minimal_matmul (+ silu*mul for ff13 on the two halves).
    base_cfg = _mm_cfg(11, 10, 8, 8, 4, 2, 4) if shape == "ff2" else _mm_cfg(11, 9, 8, 8, 8, 1, 8)

    def baseline():
        g = ttnn.all_gather(
            x, 3, cluster_axis=1, num_links=2, topology=ttnn.Topology.Linear, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        o = ttnn.experimental.minimal_matmul(
            input_tensor=g,
            weight_tensor=w,
            config=base_cfg,
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(g)
        return o

    try:
        us, _ = _timed(mesh, baseline)
        logger.info(f"AGMM {shape} M={M} baseline all_gather+minimal_matmul (no swiglu): {us:.1f} us")
    except Exception as e:
        logger.warning(f"AGMM {shape} baseline ERROR {str(e)[:200]}")

    # Config as the tt_dit Blackhole callers derive it (models/tt_dit/utils/matmul.py get_agmm_config): the
    # worker grid leaves the last core row free for the op's fabric muxes (transposed grid), the in0 sender axis
    # splits into exactly num_links groups, K_block divides the per-device K tiles, and a barrier semaphore is
    # passed because no persistent gather buffer is used.
    from models.tt_dit.utils.matmul import get_agmm_config

    topos = {"ring": ttnn.Topology.Ring, "linear": ttnn.Topology.Linear}
    topo_names = os.environ.get("AGMM_TOPO", "ring").split(",")
    links = [int(v) for v in os.environ.get("AGMM_LINKS", "2").split(",")]
    barrier = ttnn.create_global_semaphore(mesh, full_crs, 0)
    res = []
    for tname, nl in itertools.product(topo_names, links):
        core_grid, cfg, wpl = get_agmm_config(
            M, K, N, full_grid=grid, cluster_size=4, num_links=nl, fuse_swiglu=swiglu, force_transpose=True
        )
        gx, gy = core_grid.x, core_grid.y
        blk = (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size, cfg.subblock_h, cfg.subblock_w)

        def fused():
            out = ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=x,
                weight_tensor=w,
                config=cfg,
                compute_kernel_config=ckc,
                multi_device_global_semaphore=sems,
                barrier_semaphore=barrier,
                num_links=nl,
                topology=topos[tname],
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                force_transpose=True,
                num_workers_per_link=wpl,
                num_buffers_per_channel=24,
                fuse_swiglu=swiglu,
            )
            return out[0] if isinstance(out, (list, tuple)) else out

        tag = f"grid {gx}x{gy} {tname} links={nl} w/link={wpl} blk M{blk[0]} K{blk[1]} N{blk[2]} sb{blk[3]}x{blk[4]}"
        try:
            us, first = _timed(mesh, fused)
            _, pcc = comp_pcc(golden, first.reshape(golden.shape), 0.97)
            res.append((us, tag, pcc))
            logger.info(f"AGMM {shape} M={M} fused {tag}: {us:.1f} us pcc={pcc:.4f}")
        except Exception as e:
            logger.warning(f"AGMM {shape} fused {tag}: ERROR {str(e)[:200]}")
    for r in sorted(res)[:6]:
        logger.info(f"AGMM_BEST {shape} M={M} {r[0]:.1f} us {r[1]} pcc={r[2]:.4f}")
