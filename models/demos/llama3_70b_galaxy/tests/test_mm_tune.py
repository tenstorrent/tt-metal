# Sweep minimal_matmul grid/blocks for the TP32 prefill FF1/FF3 and FF2 per-chip shapes (seq 4096) on one BH chip.
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_mm_tune.py -s --timeout=1200
import itertools
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

N_PER_TRACE, REPLAYS = 10, 5
MM_M = int(os.environ.get("MM_M", "4096"))
SHAPES = {  # name: (M, K, N, weight dtype, fidelity)
    "ff13": (MM_M, 2048, 3584, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
    "ff2": (MM_M, 3584, 2048, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
}


def _cfg(gx, gy, mb, kb, nb, sh, sw):
    return ttnn.MinimalMatmulConfig(
        M_block_size=mb,
        K_block_size=kb,
        N_block_size=nb,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 100_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("shape", list(SHAPES))
def test_mm_tune(mesh_device, device_params, shape):
    M, K, N, wdt, fid = SHAPES[shape]
    dev = mesh_device
    grid = dev.compute_with_storage_grid_size()
    logger.info(f"MMTUNE {shape} device grid {grid.x}x{grid.y}")
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    ta, tb = torch.randn(1, 1, M, K), torch.randn(1, 1, K, N)
    a = ttnn.from_torch(
        ta, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b = ttnn.from_torch(tb, device=dev, layout=ttnn.TILE_LAYOUT, dtype=wdt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    golden = (ta @ tb).float()
    grids = [
        tuple(map(int, g.split("x")))
        for g in os.environ.get("MM_GRIDS", "7x9,6x8,8x9,8x10,10x9,10x10,11x9,11x10").split(",")
    ]
    grids = [g for g in grids if g[0] <= grid.x and g[1] <= grid.y]
    blocks = [(8, 8, 8, 4, 2), (8, 8, 8, 1, 8), (8, 7, 8, 1, 4), (8, 8, 4, 2, 4), (4, 8, 8, 4, 2), (16, 8, 8, 4, 2)]
    res = []
    for (gx, gy), (mb, kb, nb, sh, sw) in itertools.product(grids, blocks):
        cfg = _cfg(gx, gy, mb, kb, nb, sh, sw)
        fn = lambda: ttnn.experimental.minimal_matmul(
            input_tensor=a,
            weight_tensor=b,
            config=cfg,
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        try:
            out = fn()
            ttnn.synchronize_device(dev)
            ok, pcc = comp_pcc(golden, ttnn.to_torch(out).float(), 0.97)
            ttnn.deallocate(out)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            outs = [fn() for _ in range(N_PER_TRACE)]
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            us = (time.perf_counter() - t0) / (REPLAYS * N_PER_TRACE) * 1e6
            ttnn.release_trace(dev, tid)
            for o in outs:
                ttnn.deallocate(o)
            tflops = 2 * M * K * N / us / 1e6
            res.append((us, gx, gy, mb, kb, nb, sh, sw, pcc))
            logger.info(
                f"MMTUNE {shape} grid {gx}x{gy} blk M{mb} K{kb} N{nb} sb{sh}x{sw}: {us:.1f} us {tflops:.0f} TFLOP/s pcc={pcc:.4f}"
            )
        except Exception as e:
            logger.warning(f"MMTUNE {shape} grid {gx}x{gy} blk M{mb} K{kb} N{nb} sb{sh}x{sw}: ERROR {str(e)[:160]}")
    for r in sorted(res)[:8]:
        logger.info(
            f"MMTUNE_BEST {shape} {r[0]:.1f} us grid {r[1]}x{r[2]} blk M{r[3]} K{r[4]} N{r[5]} sb{r[6]}x{r[7]} pcc={r[8]:.4f}"
        )
