"""(4096, 2048, 2048) matmul at canonical GEMM_FLOPS-style config.

Same as test_matmul_2d_host_perf_out_of_box invocation: `a @ b`, no kwargs.
Reports both util metrics: wall-clock (SDPA convention) and TRISC1 (GEMM_FLOPS
convention).
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
    t1 = sum(int(p.program_analyses_results["DEVICE TRISC1 KERNEL DURATION [ns]"].duration) for p in perf[chip])
    w = sum(int(p.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration) for p in perf[chip])
    return t1, w


def test_canonical_4k(device):
    M, K, N = 4096, 2048, 2048
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
    _measure(device)

    t1s, ws = [], []
    for _ in range(5):
        out = a @ b
        t1, w = _measure(device)
        t1s.append(t1)
        ws.append(w)
        ttnn.deallocate(out)

    t1 = statistics.median(t1s)
    w = statistics.median(ws)
    ideal = (M * K * N) / (32 * 32 * 32) * 32 / 64

    print(
        f"\n=== ({M},{K},{N}) bf16 HiFi2 DRAM canonical (a @ b, no kwargs) ===\n"
        f"  Wall-clock:  {w} ns,  util = {ideal/w*100:.2f}%\n"
        f"  TRISC1:      {t1} ns,  util = {ideal/t1*100:.2f}%"
    )
