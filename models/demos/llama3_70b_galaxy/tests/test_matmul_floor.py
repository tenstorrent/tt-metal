# Matmul-only lower bound for the batch-32 DP4 x TP8 prefill, measured. Runs just the model's per-chip matmuls (QKV,
# WO, FF1|FF3 + SwiGLU, FF2 with the model's weight formats, fidelity and block configs) for a whole 32-prompt batch:
# STEPS steps x 80 layers back to back on all 32 chips, so the chips sit at their power cap as in a real burst. Leaves out
# all-gathers, reduce-scatters, attention, norms, rope and the LM head, so a real forward can only be slower.
#   pytest models/demos/llama3_70b_galaxy/tests/test_matmul_floor.py -s     env: MF_M (4096,16384), MF_STEPS (8)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H, D, NQ, FF, LAYERS = 8192, 128, 8, 28672 // 8, 80
DRAM = ttnn.DRAM_MEMORY_CONFIG
START_S = {4096: 0.115, 16384: 0.3}  # measured request-to-first-step offsets (dp4_ship)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_matmul_floor(mesh_device, device_params):
    mesh = mesh_device
    steps = int(os.environ.get("MF_STEPS") or 8)
    grid = mesh.compute_with_storage_grid_size()
    ck = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def cfg(kb):
        return ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=kb,
            N_block_size=8,
            subblock_h=1,
            subblock_w=8,
            compute_with_storage_grid_size=grid,
        )

    def dev(shape, dt, scale=1.0):
        return ttnn.from_torch(
            torch.randn(shape) * scale, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=dt, memory_config=DRAM
        )

    wqkv = dev((1, 1, H, (NQ + 2) * D), ttnn.bfloat8_b, 0.02)
    wo = dev((1, 1, NQ * D, H), ttnn.bfloat8_b, 0.02)
    w13 = dev((1, 1, H, 2 * FF), ttnn.bfloat4_b, 0.02)
    w2 = dev((1, 1, FF, H), ttnn.bfloat8_b, 0.02)
    for M in [int(m) for m in (os.environ.get("MF_M") or "4096,16384").split(",")]:
        hg = dev((1, 1, M, H), ttnn.bfloat8_b)
        a8 = dev((1, 1, M, NQ * D), ttnn.bfloat8_b)
        f8 = dev((1, 1, M, FF), ttnn.bfloat8_b)
        mm = ttnn.experimental.minimal_matmul

        def layer():
            outs = [
                mm(
                    input_tensor=hg,
                    weight_tensor=wqkv,
                    config=cfg(16),
                    compute_kernel_config=ck,
                    memory_config=DRAM,
                    dtype=ttnn.bfloat16,
                ),
                mm(
                    input_tensor=a8,
                    weight_tensor=wo,
                    config=cfg(16),
                    compute_kernel_config=ck,
                    memory_config=DRAM,
                    dtype=ttnn.bfloat16,
                ),
                mm(
                    input_tensor=hg,
                    weight_tensor=w13,
                    config=cfg(32),
                    compute_kernel_config=ck,
                    memory_config=DRAM,
                    dtype=ttnn.bfloat8_b,
                    fuse_swiglu=True,
                ),
                mm(
                    input_tensor=f8,
                    weight_tensor=w2,
                    config=cfg(16),
                    compute_kernel_config=ck,
                    memory_config=DRAM,
                    dtype=ttnn.bfloat16,
                ),
            ]
            for o in outs:
                ttnn.deallocate(o)

        for _ in range(LAYERS):
            layer()
        ttnn.synchronize_device(mesh)
        time.sleep(10)
        t0 = time.perf_counter()
        marks = []
        for s in range(steps):
            for _ in range(LAYERS):
                layer()
            ttnn.synchronize_device(mesh)
            marks.append(time.perf_counter() - t0)
        step_s = [b - a for a, b in zip([0.0] + marks[:-1], marks)]
        start = START_S.get(M, 0.115)
        mean_dp4 = start + sum(marks) / steps  # 4 prompts finish at the end of each step
        total = marks[-1]
        mean_ideal = start + total * (4 * steps + 1) / (2 * 4 * steps)  # one prompt at a time at the same rate
        gf = 2.0 * M * (H * (NQ + 2) * D + NQ * D * H + H * 2 * FF + FF * H) / 1e9
        logger.info(f"MFLOOR M={M} steps " + " ".join(f"{x * 1e3:.0f}" for x in step_s) + " ms")
        logger.info(
            f"MFLOOR M={M}: {total / steps / LAYERS * 1e3:.3f} ms per layer ({gf:.0f} GFLOP, "
            f"{gf / (total / steps / LAYERS) / 1e3:.0f} TF/s per chip); 32-prompt TTFT mean, matmuls only: "
            f"{mean_dp4:.3f} s with 4 prompts per step, {mean_ideal:.3f} s with an ideal one-at-a-time schedule"
        )
        for t in (hg, a8, f8):
            ttnn.deallocate(t)
