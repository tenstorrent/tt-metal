# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Back-to-back matmul load for the Tracy zone alignment test.

It keeps one device busy for ``SECONDS`` so the device clock's rate changes with load and the profiler re-anchors its
host/device clock mapping many times while zones are being placed. Run as a subprocess under Tracy capture.

Environment variables:
  SECONDS   how long to run (default 8)

Exit codes:
  0  success
  2  no device available
  5  real-time profiler not active on this dispatch configuration
"""

import os
import sys
import time

import torch

import ttnn


def main():
    seconds = float(os.environ.get("SECONDS", "8"))
    if ttnn.GetNumAvailableDevices() < 1:
        print("SKIP: no devices available", file=sys.stderr)
        sys.exit(2)

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        print("SKIP: Real-time profiler is not active on this configuration", file=sys.stderr)
        ttnn.close_mesh_device(mesh_device)
        sys.exit(5)

    a = ttnn.from_torch(torch.randn(2048, 2048, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=mesh_device)
    b = ttnn.from_torch(torch.randn(2048, 2048, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=mesh_device)

    matmuls = 0
    start = time.time()
    while time.time() - start < seconds:
        outputs = [ttnn.matmul(a, b) for _ in range(20)]
        matmuls += len(outputs)
        ttnn.synchronize_device(mesh_device)
        for output in outputs:
            output.deallocate()
    print(f"matmuls={matmuls} in {time.time() - start:.1f}s", flush=True)

    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
