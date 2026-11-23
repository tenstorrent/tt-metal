# Energy of fp32 vs bf16 DEST accumulation for the FF2-shaped matmul ([M, 3584] x [3584, 8192], LoFi, bf8 in0) on all 32
# chips at the power cap. matmul_reduce_scatter_prefill always computes in fp32 DEST (full sync); the plain matmuls use
# bf16 DEST. If fp32 DEST costs much more energy per MAC, that explains part of MMRS's excess over a plain matmul.
#   pytest models/demos/llama3_70b_galaxy/tests/test_dest_energy.py -s      env: DE_M (16384,4096), DE_SECS (6)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama3_70b_galaxy.tests.test_op_energy import Poller

DRAM = ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_dest_energy(mesh_device, device_params):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    secs = float(os.environ.get("DE_SECS") or 6)
    poll = Poller()
    time.sleep(3)
    t0 = time.time()
    time.sleep(8)
    idle, _, _, _ = poll.window(t0 + 1, time.time())
    logger.info(f"DEST idle {idle:.1f} W")
    for M in [int(m) for m in (os.environ.get("DE_M") or "16384,4096").split(",")]:
        a = ttnn.from_torch(
            torch.randn(1, 1, M, 3584), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=DRAM
        )
        for wname, wdt in (("w8", ttnn.bfloat8_b), ("w4", ttnn.bfloat4_b)):
            w = ttnn.from_torch(
                torch.randn(1, 1, 3584, 8192) * 0.02,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=wdt,
                memory_config=DRAM,
            )
            for fp32 in (False, True):
                ck = ttnn.init_device_compute_kernel_config(
                    mesh.arch(),
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=fp32,
                    packer_l1_acc=True,
                )
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=16,
                    N_block_size=8,
                    subblock_h=1,
                    subblock_w=4 if fp32 else 8,
                    compute_with_storage_grid_size=grid,
                )

                def fn():
                    return ttnn.experimental.minimal_matmul(
                        input_tensor=a,
                        weight_tensor=w,
                        config=cfg,
                        compute_kernel_config=ck,
                        memory_config=DRAM,
                        dtype=ttnn.bfloat8_b,
                    )

                ttnn.deallocate(fn())
                ttnn.synchronize_device(mesh)
                time.sleep(4)
                t1 = time.perf_counter()
                for _ in range(10):
                    ttnn.deallocate(fn())
                ttnn.synchronize_device(mesh)
                ms_cool = (time.perf_counter() - t1) / 10 * 1e3
                n = max(20, int(secs * 1e3 / ms_cool))
                w0 = time.time()
                t1 = time.perf_counter()
                for _ in range(n):
                    ttnn.deallocate(fn())
                ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - t1) / n * 1e3
                p, amin, _, ns = poll.window(w0 + 2.0, time.time() - 0.2)
                logger.info(
                    f"DEST M={M} {wname} fp32={fp32}: {ms_cool:.3f} ms cool, {ms:.3f} ms sustained, {p:.1f} W, "
                    f"aiclk min {amin:.0f}, dyn {(p - idle) * ms:.1f} mJ ({ns} samples)"
                )
            ttnn.deallocate(w)
        ttnn.deallocate(a)
    poll.stop = True
