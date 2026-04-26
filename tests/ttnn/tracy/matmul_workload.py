# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Small matmul workload for the quick real-time profiler tests.

Run as a subprocess from the pytest-level test so that the parent pytest
process never opens a device (the UMD ``CHIP_IN_USE_*_PCIe`` lock is
held for the lifetime of the process that first touches it).

Environment variables:
  MODE            "simple" (10 matmuls) or "trace" (trace capture + replay)
  RT_RECORDS_PATH output JSON file for real-time profiler records

Exit codes:
  0  success
  1  bad invocation
  2  insufficient devices
"""

import json
import os
import sys
import threading
import time

import torch

import ttnn


def _make_tensors(device, single_core: bool):
    torch.manual_seed(0)
    if single_core:
        shape = (1024, 1024)
    else:
        shape = (1024, 1024)
    a = ttnn.to_layout(
        ttnn.to_device(
            ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16)),
            device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        ),
        ttnn.TILE_LAYOUT,
    )
    b = ttnn.to_layout(
        ttnn.to_device(
            ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16)),
            device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        ),
        ttnn.TILE_LAYOUT,
    )
    return a, b


def main():
    mode = os.environ.get("MODE")
    rt_path = os.environ.get("RT_RECORDS_PATH")
    if mode not in ("simple", "trace") or not rt_path:
        print(
            f"ERROR: MODE must be 'simple'/'trace' and RT_RECORDS_PATH must be set (got MODE={mode})", file=sys.stderr
        )
        sys.exit(1)

    if ttnn.GetNumAvailableDevices() < 1:
        print("ERROR: no devices available", file=sys.stderr)
        sys.exit(2)

    device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        l1_small_size=24576,
        trace_region_size=1_996_800,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )

    records = []
    lock = threading.Lock()

    def collect(record):
        with lock:
            records.append(
                {
                    "program_id": record.program_id,
                    "chip_id": record.chip_id,
                    "start_timestamp": record.start_timestamp,
                    "end_timestamp": record.end_timestamp,
                    "frequency_ghz": record.frequency,
                }
            )

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect)

    try:
        if mode == "simple":
            a, b = _make_tensors(device, single_core=False)
            for _ in range(10):
                ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))
            ttnn.synchronize_device(device)
        else:  # trace
            a, b = _make_tensors(device, single_core=True)
            # Warm-up so binaries are loaded before trace capture.
            ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=1, x=1))
            ttnn.synchronize_device(device)

            tid = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(50):
                ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=1, x=1))
            ttnn.end_trace_capture(device, tid, cq_id=0)

            for _ in range(10):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.release_trace(device, tid)
            ttnn.synchronize_device(device)
    finally:
        time.sleep(1.0)
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
        ttnn.close_mesh_device(device)

    with lock:
        snapshot = list(records)
    with open(rt_path, "w") as f:
        json.dump(snapshot, f)
    print(f"Saved {len(snapshot)} RT records -> {rt_path}")


if __name__ == "__main__":
    main()
