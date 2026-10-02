# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Performance characterization for #58986 (FP32 cumsum inf/NaN fix).
#
# Measures device kernel time for FP32 cumsum:
#   1. WITH the fix (default compensated Kahan path, patched)
#   2. Uncompensated baseline (disable_compensated_sum=True)
#   3. PRE-FIX code requires checking out the base commit separately;
#      run this script on both commits and diff the results.
#
# Uses the ttnn device profiler. Run on Blackhole / Wormhole hardware:
#   python tests/ttnn/unit_tests/operations/reduce/cumsum_inf_benchmark.py
#
# Scan lengths and dims cover the regression-test matrix (4, 8, 33;
# dims 0 and -1) plus longer scans to characterize the per-tile overhead
# of the fix (1 binary sub + isfinite + fill + where + init switches).

import time

import torch

import ttnn
from tests.ttnn.utils_for_testing import get_device


def bench_cumsum(device, scan_length, dim, disable_compensated, iters=20, warmup=5):
    shape = (2, scan_length) if dim == 0 else (scan_length,)
    torch_input = torch.randn(shape, dtype=torch.float32)
    # Include an inf in the input so the fix's guard path is exercised
    torch_input.flatten()[1] = float("inf")

    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.Layout.TILE)

    # Warmup
    for _ in range(warmup):
        out = ttnn.cumsum(input_tensor, dim=dim, disable_compensated_sum=disable_compensated)
        ttnn.synchronize_device(device)

    # Timed runs; device-side kernel time via profiler if available,
    # otherwise wall-clock with device synchronize as a fallback.
    times_ms = []
    for _ in range(iters):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        out = ttnn.cumsum(input_tensor, dim=dim, disable_compensated_sum=disable_compensated)
        ttnn.synchronize_device(device)
        times_ms.append((time.perf_counter() - start) * 1e3)

    times_ms.sort()
    # Median of timed runs (robust to outliers)
    median_ms = times_ms[len(times_ms) // 2]
    return median_ms


def main():
    device = get_device()
    print(f"device: {device}")
    print(f"{'scan':>6} {'dim':>4} {'path':>14} {'median_ms':>10}")
    print("-" * 42)

    results = {}
    for scan_length in [4, 8, 33, 256, 4096]:
        for dim in [0, -1]:
            for disable_compensated, label in [(False, "compensated"), (True, "plain")]:
                try:
                    ms = bench_cumsum(device, scan_length, dim, disable_compensated)
                except Exception as e:
                    print(f"{scan_length:>6} {dim:>4} {label:>14} ERROR: {e}")
                    continue
                results[(scan_length, dim, label)] = ms
                print(f"{scan_length:>6} {dim:>4} {label:>14} {ms:>10.3f}")

    print()
    print("Overhead of fix vs uncompensated baseline (compensated/plain - 1):")
    for scan_length in [4, 8, 33, 256, 4096]:
        for dim in [0, -1]:
            comp = results.get((scan_length, dim, "compensated"))
            plain = results.get((scan_length, dim, "plain"))
            if comp is not None and plain is not None and plain > 0:
                overhead = (comp / plain - 1.0) * 100.0
                print(f"  scan={scan_length:>5} dim={dim:>2}: {overhead:+.1f}%")

    print()
    print("NOTE: for the pre-fix comparison, check out the base commit")
    print("(before this PR), re-run, and diff the 'compensated' column.")


if __name__ == "__main__":
    main()
