# Sustained matmul rate at the chip power cap, all 32 Blackhole chips busy (the DP4 x TP8 FF13 shape per chip:
# [M, 8192] x [8192, 7168], LoFi). Compares weight / activation formats: if a smaller format costs fewer joules per FLOP,
# the throttled rate rises, and with it the prefill floor. Short loops run at full clock; long loops hit the TDP limit.
#   pytest models/demos/llama3_70b_galaxy/tests/test_energy_floor.py -s
#   env: EF_M (16384), EF_ITERS (10,2000), EF_COMBOS (w8a8,w4a8,w8a16,w4a16), EF_KB (32)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H, N = 8192, 7168
DRAM = ttnn.DRAM_MEMORY_CONFIG
DT = {"8": ttnn.bfloat8_b, "4": ttnn.bfloat4_b, "16": ttnn.bfloat16}


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_energy_floor(mesh_device, device_params):
    mesh = mesh_device
    M = int(os.environ.get("EF_M") or 16384)
    iters = [int(x) for x in (os.environ.get("EF_ITERS") or "10,2000").split(",")]
    combos = (os.environ.get("EF_COMBOS") or "w8a8,w4a8,w8a16,w4a16").split(",")
    kb = int(os.environ.get("EF_KB") or 32)
    grid = mesh.compute_with_storage_grid_size()
    ck = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=8, K_block_size=kb, N_block_size=8, subblock_h=1, subblock_w=8, compute_with_storage_grid_size=grid
    )
    flop = 2.0 * M * H * N
    res = {}
    for c in combos:
        wd, ad = DT[c[1 : c.index("a")]], DT[c[c.index("a") + 1 :]]
        a = ttnn.from_torch(torch.randn(1, 1, M, H), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ad, memory_config=DRAM)
        w = ttnn.from_torch(
            torch.randn(1, 1, H, N) * 0.02, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=wd, memory_config=DRAM
        )

        def run():
            o = ttnn.experimental.minimal_matmul(
                input_tensor=a,
                weight_tensor=w,
                config=cfg,
                compute_kernel_config=ck,
                memory_config=DRAM,
                dtype=ttnn.bfloat8_b,
            )
            ttnn.deallocate(o)

        try:
            run()
            ttnn.synchronize_device(mesh)
            for n in iters:
                time.sleep(5)  # let the chips cool / recover clock between loops
                t0 = time.perf_counter()
                for _ in range(n):
                    run()
                ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - t0) / n * 1e3
                res[(c, n)] = ms
                logger.info(f"EFLOOR M={M} {c} iters={n}: {ms:.3f} ms/matmul, {flop / ms / 1e9:.0f} TF/s per chip")
        except Exception as e:
            logger.warning(f"EFLOOR M={M} {c} ERROR {str(e)[:300]}")
        ttnn.deallocate(a), ttnn.deallocate(w)
    logger.info(
        "EFLOOR SUMMARY " + " ".join(f"[{c} x{n}]={v:.3f}ms/{flop / v / 1e9:.0f}TF" for (c, n), v in res.items())
    )
