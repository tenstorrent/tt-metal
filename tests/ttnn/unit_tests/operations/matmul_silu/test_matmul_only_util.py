"""Matmul-only math utilization probe at (4096, 2048, 2048).

Same compute config as the matmul_silu benchmark (HiFi2 + fp32_dest_acc_en),
but WITHOUT activation — measures the cost of just the matmul, so we can see
how much of the headline-fused number is matmul vs the SiLU pass.

Util formula (same as SDPA perf test):
  mm_flops      = 2 * M * K * N
  cycles        = duration_ns * 1.0 GHz
  peak_flops    = active_cores * cycles * 2048   (HiFi2: 2048 FLOPs/cycle/core)
  util          = mm_flops / peak_flops * 100

Run:
  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/matmul_silu/test_matmul_only_util.py -s
"""

import os
import statistics

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "Warning")

import torch  # noqa: E402
import ttnn  # noqa: E402


def _measure(device):
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    perf = ttnn.get_latest_programs_perf_data()
    chip = next(iter(perf))
    return sum(int(p.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration) for p in perf[chip])


def test_matmul_only_util(device):
    M, K, N = 4096, 2048, 2048
    torch.manual_seed(0)
    a_t = (-1.0 + 2.0 * torch.rand(M, K, dtype=torch.float32)).to(torch.bfloat16)
    b_t = (-1.0 + 2.0 * torch.rand(K, N, dtype=torch.float32)).to(torch.bfloat16)
    a = ttnn.from_torch(a_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(b_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)

    # Warmup + drain
    for _ in range(2):
        ttnn.deallocate(
            ttnn.matmul(
                a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG, core_grid=device.core_grid, compute_kernel_config=cfg
            )
        )
    _measure(device)

    durs = []
    for _ in range(5):
        out = ttnn.matmul(
            a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG, core_grid=device.core_grid, compute_kernel_config=cfg
        )
        durs.append(_measure(device))
        ttnn.deallocate(out)

    duration_ns = statistics.median(durs)
    mm_flops = 2 * M * K * N
    active_cores = 64  # full 8x8 grid forced via core_grid
    peak_flops = active_cores * duration_ns * 2048  # 1 cycle ≈ 1 ns on Wormhole
    util_pct = mm_flops / peak_flops * 100

    print(
        f"\n=== matmul-only (no SiLU) at ({M},{K},{N}) ===\n"
        f"  duration (median of 5): {duration_ns} ns\n"
        f"  matmul FLOPs:           {mm_flops:,}\n"
        f"  active cores:           {active_cores}\n"
        f"  math utilization:       {util_pct:.2f}%"
    )
