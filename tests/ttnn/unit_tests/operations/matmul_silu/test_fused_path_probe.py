"""Probe whether ttnn.matmul(core_grid=..., activation='silu') uses the fused
in-kernel SiLU path (single device program) or the 2-program fallback.

Run via:
  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/matmul_silu/test_fused_path_probe.py -s
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "Warning")

import torch
import ttnn


def _drain_profiler(device):
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    _ = ttnn.get_latest_programs_perf_data()


def _measure(device):
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    perf = ttnn.get_latest_programs_perf_data()
    chip = next(iter(perf))
    progs = perf[chip]
    return [int(p.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration) for p in progs]


def test_fused_path_probe(device):
    torch.manual_seed(0)
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)

    for M, K, N in [(32, 1024, 1024), (64, 2048, 2048)]:
        a_t = (-1.0 + 2.0 * torch.rand(M, K, dtype=torch.float32)).to(torch.bfloat16)
        b_t = (-1.0 + 2.0 * torch.rand(K, N, dtype=torch.float32)).to(torch.bfloat16)
        a = ttnn.from_torch(a_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b = ttnn.from_torch(b_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Warmup both invocation flavors.
        for _ in range(2):
            ttnn.deallocate(
                ttnn.matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG, activation="silu", compute_kernel_config=cfg)
            )
            ttnn.deallocate(
                ttnn.matmul(
                    a,
                    b,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    activation="silu",
                    core_grid=device.core_grid,
                    compute_kernel_config=cfg,
                )
            )
        _drain_profiler(device)

        # Kwarg-only (no core_grid) — expect 2 programs (matmul + silu).
        out = ttnn.matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG, activation="silu", compute_kernel_config=cfg)
        durs_no_grid = _measure(device)
        out_no_grid = ttnn.to_torch(out).to(torch.float32)
        ttnn.deallocate(out)

        # With core_grid set — does it fuse into 1 program?
        out = ttnn.matmul(
            a,
            b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            activation="silu",
            core_grid=device.core_grid,
            compute_kernel_config=cfg,
        )
        durs_grid = _measure(device)
        out_grid = ttnn.to_torch(out).to(torch.float32)
        ttnn.deallocate(out)

        ref = torch.nn.functional.silu(a_t.to(torch.float32) @ b_t.to(torch.float32))
        mad_no_grid = (out_no_grid - ref).abs().max().item()
        mad_grid = (out_grid - ref).abs().max().item()

        print(
            f"\n=== shape ({M:4d},{K:5d},{N:5d}) ===\n"
            f"  no core_grid: programs={len(durs_no_grid)} durs={durs_no_grid} total={sum(durs_no_grid):>7d}ns  vs_ref_max_abs={mad_no_grid:.3e}\n"
            f"  with core_grid={device.core_grid!s}: programs={len(durs_grid)} durs={durs_grid} total={sum(durs_grid):>7d}ns  vs_ref_max_abs={mad_grid:.3e}"
        )
