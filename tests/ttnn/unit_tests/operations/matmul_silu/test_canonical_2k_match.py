"""Reproduce GEMM_FLOPS row at (2048, 2048, 2048) bf16 HiFi2 DRAM-DRAM no-trace.

Report claims device util = 66.74% at this shape. Probe runs the matmul with
the same configuration the report's `test_matmul_2d_host_perf_out_of_box`
uses (line 646 of test_benchmark.py):

  output_t = in0_t @ in1_t

i.e. plain `ttnn.matmul` with NO kwargs (no activation, no core_grid, no
compute_kernel_config — all defaults). Same measurement methodology as
verify_makora.py: profiler DEVICE KERNEL DURATION [ns], sum across all
programs in latest read, median of 5 iters.

Util formula matches the report's:
  Ideal cycles = (M × K × N) / 32^3 × cycle_per_tile / num_cores
  Util         = ideal cycles / actual cycles
With HiFi2 cycle_per_tile=32 and num_cores=64.
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


def test_canonical_2k(device):
    M, K, N = 2048, 2048, 2048
    torch.manual_seed(0)
    # GEMM_FLOPS OOB code uses `torch.ones` for in0 and `torch.randn` for in1.
    in0 = torch.ones((1, 1, M, K), dtype=torch.bfloat16)
    in1 = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
    in0_t = ttnn.from_torch(
        in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    in1_t = ttnn.from_torch(
        in1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Warmup + drain (report uses 5 warmup, 100 measurement; using same warmup count).
    for _ in range(5):
        out = in0_t @ in1_t
        ttnn.deallocate(out)
    _measure(device)

    durs = []
    for _ in range(5):
        out = in0_t @ in1_t
        durs.append(_measure(device))
        ttnn.deallocate(out)

    duration_ns = statistics.median(durs)

    # Util via the same formula the GEMM_FLOPS report documents.
    HiFi2_cycle_per_tile = 32
    num_cores = 64
    ideal_cycles = (M * K * N) / (32 * 32 * 32) * HiFi2_cycle_per_tile / num_cores
    util_pct = ideal_cycles / duration_ns * 100  # 1 cycle = 1 ns on Wormhole

    print(
        f"\n=== Probe at ({M},{K},{N}) bf16 HiFi2 DRAM-DRAM no-trace ===\n"
        f"  GEMM_FLOPS OOB report claim:    device util = 66.74%, inference_time_avg = 302200 ns\n"
        f"  This probe (5 iters, median):   duration = {duration_ns} ns,  util = {util_pct:.2f}%\n"
        f"  All durations (ns):             {durs}"
    )
