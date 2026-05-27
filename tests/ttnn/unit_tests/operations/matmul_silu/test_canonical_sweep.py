"""Sweep configurations at (2048, 2048, 2048) bf16 DRAM to find what
matches the GEMM_FLOPS report's claimed 66.74% device util.

Configs tried:
  - No compute_kernel_config (defaults — what test_matmul_2d_host_perf_out_of_box uses)
  - Explicit HiFi2 + fp32_dest_acc=False (what the report's CSV labels claim)
  - Explicit HiFi2 + fp32_dest_acc=True (what verify_makora uses to match Makora)
  - Explicit HiFi4 + fp32_dest_acc=False (default for bf16 in some paths)

All use plain a @ b (no activation, no core_grid).
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


def _run(a, b, device, cfg, iters=5):
    kwargs = {} if cfg is None else {"compute_kernel_config": cfg}
    for _ in range(5):
        out = ttnn.matmul(a, b, **kwargs)
        ttnn.deallocate(out)
    _measure(device)
    durs = []
    for _ in range(iters):
        out = ttnn.matmul(a, b, **kwargs)
        durs.append(_measure(device))
        ttnn.deallocate(out)
    return statistics.median(durs)


def _util(duration_ns, cycle_per_tile, M=2048, K=2048, N=2048, num_cores=64):
    ideal = (M * K * N) / (32 * 32 * 32) * cycle_per_tile / num_cores
    return ideal / duration_ns * 100


def test_canonical_sweep(device):
    M, K, N = 2048, 2048, 2048
    in0 = torch.ones((1, 1, M, K), dtype=torch.bfloat16)
    in1 = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
    a = ttnn.from_torch(
        in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b = ttnn.from_torch(
        in1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    configs = [
        ("default (no compute_kernel_config)", None, None),
        (
            "HiFi2 + fp32_dest_acc=False",
            ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False),
            32,
        ),
        (
            "HiFi2 + fp32_dest_acc=True",
            ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True),
            32,
        ),
        (
            "HiFi4 + fp32_dest_acc=False",
            ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False),
            64,
        ),
        (
            "HiFi4 + fp32_dest_acc=True",
            ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
            64,
        ),
    ]

    print(f"\n=== Sweep at ({M},{K},{N}) bf16 DRAM, no activation, no core_grid ===")
    print(f"  GEMM_FLOPS OOB claims: 66.74% util at HiFi2 (label)")
    for label, cfg, cycle_per_tile in configs:
        dur = _run(a, b, device, cfg)
        utils = []
        if cycle_per_tile is not None:
            utils.append(f"util@HiFi2={_util(dur, 32):.2f}%")
            utils.append(f"util@HiFi4={_util(dur, 64):.2f}%")
        else:
            # Default — try both fidelity assumptions so we can see which it matches.
            utils.append(f"util_if_HiFi2={_util(dur, 32):.2f}%")
            utils.append(f"util_if_HiFi4={_util(dur, 64):.2f}%")
        print(f"  {label:<40}  dur={dur:>7d} ns   {'   '.join(utils)}")
