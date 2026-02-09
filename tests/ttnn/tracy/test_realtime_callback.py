# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Example: register a real-time profiler callback that collects every program
execution record and writes them to a JSON file on disk.
"""

import json
import threading
import time
from pathlib import Path

import pytest
import torch

import ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 1996800, "dispatch_core_type": ttnn.DispatchCoreType.WORKER}],
    indirect=True,
)
def test_realtime_callback_json(device, tmp_path):
    """Run a matmul workload and dump every real-time profiler record to a JSON file."""

    # -- 1. Set up the callback ------------------------------------------------
    records = []
    lock = threading.Lock()

    def collect_record(record):
        entry = {
            "program_id": record.program_id,
            "chip_id": record.chip_id,
            "start_timestamp": record.start_timestamp,
            "end_timestamp": record.end_timestamp,
            "frequency_ghz": record.frequency,
            "kernel_sources": record.kernel_sources,
        }
        with lock:
            records.append(entry)

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect_record)

    # -- 2. Run a small matmul workload ----------------------------------------
    torch.manual_seed(0)
    m, k, n = 1024, 1024, 1024
    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)

    a = ttnn.to_device(a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    b = ttnn.to_device(b, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

    for _ in range(10):
        ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))
    ttnn.synchronize_device(device)

    # Give the real-time profiler receiver thread a moment to deliver remaining records
    time.sleep(0.5)

    # -- 3. Unregister and dump to JSON ----------------------------------------
    ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)

    out_file = tmp_path / "realtime_profiler.json"
    with lock:
        snapshot = list(records)

    with open(out_file, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\nCollected {len(snapshot)} real-time profiler records -> {out_file}")
    for i, rec in enumerate(snapshot[:5]):
        duration_ticks = rec["end_timestamp"] - rec["start_timestamp"]
        print(f"  [{i}] program={rec['program_id']} chip={rec['chip_id']} duration={duration_ticks} ticks")
    if len(snapshot) > 5:
        print(f"  ... and {len(snapshot) - 5} more")

    # Basic sanity: we should have received at least one record
    assert len(snapshot) > 0, "Expected at least one real-time profiler record"
    for rec in snapshot:
        assert rec["end_timestamp"] >= rec["start_timestamp"], "end should be >= start"
        assert rec["frequency_ghz"] > 0, "frequency should be positive"
