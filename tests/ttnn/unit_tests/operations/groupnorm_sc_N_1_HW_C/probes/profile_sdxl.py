"""Profile groupnorm_sc_N_1_HW_C on SDXL shapes with Tracy device profiler.

Run with:
    TT_METAL_DEVICE_PROFILER=1 \
    python3 -m tracy -v -r -p -n sdxl_groupnorm -m \
        'pytest tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C/probes/profile_sdxl.py'

or as a plain script (just calls ttnn under the profiler):
    TT_METAL_DEVICE_PROFILER=1 \
    python3 -m tracy -v -r -p -n sdxl_groupnorm -- \
        python3 tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C/probes/profile_sdxl.py
"""
import time

import torch
import ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C


# (shape, num_groups, label). Cover the shapes that the golden suite flagged
# as numerical-precision failures plus a small reference for comparison.
CASES = [
    ((1, 1, 64, 64), 2, "ref_small"),
    ((1, 1, 64, 320), 32, "ref_sdxl_C320_G32"),
    ((1, 1, 1024, 256), 8, "ref_larger_HW"),
    ((1, 1, 4096, 320), 32, "sdxl_4096x320"),
    ((1, 1, 4096, 640), 32, "sdxl_4096x640"),
    ((1, 1, 16384, 320), 32, "sdxl_16384x320"),
    ((1, 1, 1024, 960), 32, "sdxl_1024x960"),
    ((1, 1, 1024, 1280), 32, "sdxl_1024x1280"),
    ((1, 1, 1024, 1920), 32, "sdxl_1024x1920"),
]

# 1 warmup + 3 timed iterations. Warmup amortises JIT compile + program-cache
# population so timed runs measure pure dispatch + device kernel time.
WARMUP = 1
ITERS = 3


def run_case(device, shape, G, label):
    N, _, HW, C = shape
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    g = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)

    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    g_tt = ttnn.from_torch(
        g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b_tt = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )

    # Mark the zone label in Tracy so per-shape ops are easy to find.
    ttnn.tracy_message(f"BEGIN {label} shape={shape} G={G}")

    for i in range(WARMUP + ITERS):
        ttnn.tracy_message(f"{label} iter={i}")
        t0 = time.perf_counter()
        y = groupnorm_sc_N_1_HW_C(x_tt, G, gamma=g_tt, beta=b_tt, eps=1e-5, compute_kernel_config=cfg)
        ttnn.synchronize_device(device)
        dt = (time.perf_counter() - t0) * 1000.0
        kind = "warmup" if i < WARMUP else "timed "
        print(f"  {kind} {label:25s} iter={i} host_ms={dt:8.2f}")
        del y

    ttnn.tracy_message(f"END   {label}")
    del x_tt, g_tt, b_tt


def main():
    device = ttnn.open_device(device_id=0)
    try:
        for shape, G, label in CASES:
            run_case(device, shape, G, label)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
