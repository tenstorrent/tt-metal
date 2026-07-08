# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Deterministic op workload for the real-time-profiler perf-report tests
# (test_rt_perf_report.py).

import torch
import ttnn

ITERATIONS = 12


def main():
    device = ttnn.open_device(device_id=0)
    try:
        a = ttnn.from_torch(torch.rand(64, 64), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        b = ttnn.from_torch(torch.rand(64, 64), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        for _ in range(ITERATIONS):
            c = ttnn.add(a, b)
            ttnn.multiply(c, b)
            ttnn.matmul(a, b)
        ttnn.synchronize_device(device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
