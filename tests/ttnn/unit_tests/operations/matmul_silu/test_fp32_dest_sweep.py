"""Sweep fp32_dest_acc_en on/off, matmul-only and matmul+silu, at the biggest shape."""

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


def _run(a, b, device, cfg, activation):
    kwargs = dict(memory_config=ttnn.DRAM_MEMORY_CONFIG, core_grid=device.core_grid, compute_kernel_config=cfg)
    if activation is not None:
        kwargs["activation"] = activation
    # Warmup + drain
    for _ in range(2):
        ttnn.deallocate(ttnn.matmul(a, b, **kwargs))
    _measure(device)
    durs = []
    for _ in range(5):
        out = ttnn.matmul(a, b, **kwargs)
        durs.append(_measure(device))
        ttnn.deallocate(out)
    return statistics.median(durs)


def test_fp32_dest_sweep(device):
    M, K, N = 4096, 2048, 2048
    torch.manual_seed(0)
    a_t = (-1.0 + 2.0 * torch.rand(M, K, dtype=torch.float32)).to(torch.bfloat16)
    b_t = (-1.0 + 2.0 * torch.rand(K, N, dtype=torch.float32)).to(torch.bfloat16)
    a = ttnn.from_torch(a_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(b_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    mm_flops = 2 * M * K * N

    print(f"\n=== fp32_dest_acc_en sweep at ({M},{K},{N}), 64 active cores ===")
    for fp32_acc in (True, False):
        cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_acc)
        for act_label, activation in (("matmul-only", None), ("matmul+silu", "silu")):
            duration_ns = _run(a, b, device, cfg, activation)
            util = mm_flops / (64 * duration_ns * 2048) * 100
            print(
                f"  fp32_dest_acc_en={fp32_acc!s:<5}  {act_label:<12}"
                f"  duration={duration_ns:>9d} ns  util={util:>5.2f}%"
            )
