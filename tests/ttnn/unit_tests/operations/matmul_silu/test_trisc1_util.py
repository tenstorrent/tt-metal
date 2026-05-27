"""Reproduce GEMM_FLOPS device-util convention: use TRISC1 kernel duration,
not DEVICE KERNEL DURATION (which is full program wall-clock).
"""

import os
import statistics

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "Warning")

import torch  # noqa: E402
import ttnn  # noqa: E402

TRISC1_KEY = "DEVICE TRISC1 KERNEL DURATION [ns]"
WALL_KEY = "DEVICE KERNEL DURATION [ns]"


def _measure_both(device):
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    perf = ttnn.get_latest_programs_perf_data()
    chip = next(iter(perf))
    trisc1 = sum(int(p.program_analyses_results[TRISC1_KEY].duration) for p in perf[chip])
    wall = sum(int(p.program_analyses_results[WALL_KEY].duration) for p in perf[chip])
    return trisc1, wall


def test_match_gemm_flops(device):
    M, K, N = 2048, 2048, 2048
    in0 = torch.ones((1, 1, M, K), dtype=torch.bfloat16)
    in1 = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
    a = ttnn.from_torch(
        in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b = ttnn.from_torch(
        in1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    for _ in range(5):
        out = a @ b
        ttnn.deallocate(out)
    _measure_both(device)

    trisc1_durs, wall_durs = [], []
    for _ in range(5):
        out = a @ b
        t, w = _measure_both(device)
        trisc1_durs.append(t)
        wall_durs.append(w)
        ttnn.deallocate(out)

    trisc1 = statistics.median(trisc1_durs)
    wall = statistics.median(wall_durs)

    ideal_cycles_hifi2 = (M * K * N) / (32 * 32 * 32) * 32 / 64  # 131,072
    util_trisc1 = ideal_cycles_hifi2 / trisc1 * 100
    util_wall = ideal_cycles_hifi2 / wall * 100

    print(
        f"\n=== ({M},{K},{N}) bf16 HiFi2 DRAM default config ===\n"
        f"  GEMM_FLOPS OOB CSV (this hardware):  device util = 66.58%\n"
        f"  TRISC1 duration (median):            {trisc1} ns,  util = {util_trisc1:.2f}%\n"
        f"  Wall-clock duration (median):        {wall} ns,  util = {util_wall:.2f}%"
    )
