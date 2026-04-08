# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import random
from loguru import logger

import ttnn


@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 1996800, "dispatch_core_type": ttnn.DispatchCoreType.WORKER}], indirect=True
)
def test_with_ops(device):
    torch.manual_seed(0)
    m = 1024
    k = 1024
    n = 1024
    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)

    a = ttnn.to_device(a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    b = ttnn.to_device(b, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

    ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))
    # Ensure all binaries are compiled/loaded before starting trace capture.
    # Trace capture does not allow device writes (e.g. binary loading) on fast dispatch paths.
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(100):
        ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))
    ttnn.end_trace_capture(device, tid, cq_id=0)

    for i in range(5):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(device, tid)


@pytest.mark.parametrize(
    "device_params,capture_count,replay_count",
    [
        ({"trace_region_size": 1996800, "dispatch_core_type": ttnn.DispatchCoreType.WORKER}, 100, 5),
        ({"trace_region_size": 1996800, "dispatch_core_type": ttnn.DispatchCoreType.WORKER}, 5, 600),
    ],
    indirect=["device_params"],
    ids=["100-5", "5-600"],
)
def test_with_ops_single_core(device, capture_count, replay_count):
    torch.manual_seed(0)
    m = 1024
    k = 1024
    n = 1024
    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)

    a = ttnn.to_device(a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    b = ttnn.to_device(b, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

    ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=1, x=1))
    # Ensure all binaries are compiled/loaded before starting trace capture.
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(capture_count):
        ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=1, x=1))
    ttnn.end_trace_capture(device, tid, cq_id=0)

    for i in range(replay_count):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(device, tid)


@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 1996800, "dispatch_core_type": ttnn.DispatchCoreType.WORKER}], indirect=True
)
def test_with_ops_multiple_trace_ids(device):
    torch.manual_seed(0)
    m = 1024
    k = 1024
    n = 1024
    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)

    a = ttnn.to_device(a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    b = ttnn.to_device(b, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

    ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))
    # Warm-up the add op as well so its binaries are loaded outside trace capture.
    ttnn.add(a, b)
    # Ensure all binaries are compiled/loaded before starting trace capture.
    ttnn.synchronize_device(device)

    trace_ids = []
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(3):
        ttnn.add(a, b)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    trace_ids.append(tid)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(2):
        ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))
    ttnn.end_trace_capture(device, tid, cq_id=0)
    trace_ids.append(tid)

    for i in range(3):
        random.seed(i)
        shuffled_trace_ids = random.sample(trace_ids, len(trace_ids))
        for tid in shuffled_trace_ids:
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

        ttnn.ReadDeviceProfiler(device)

    for tid in trace_ids:
        ttnn.release_trace(device, tid)


@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 1996800, "dispatch_core_type": ttnn.DispatchCoreType.WORKER}], indirect=True
)
def test_with_ops_trace_with_non_trace(device):
    torch.manual_seed(0)
    m = 1024
    k = 1024
    n = 1024
    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((k, n), dtype=torch.bfloat16)

    a = ttnn.from_torch(torch_a)
    b = ttnn.from_torch(torch_b)

    a = ttnn.to_device(a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    b = ttnn.to_device(b, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

    for _ in range(5):
        ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))

    trace_ids = []
    # Ensure all binaries are compiled/loaded before starting trace capture.
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(10):
        ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))
    ttnn.end_trace_capture(device, tid, cq_id=0)
    trace_ids.append(tid)

    for _ in range(2):
        for tid in trace_ids:
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

    for _ in range(5):
        ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))

    for tid in trace_ids:
        ttnn.release_trace(device, tid)
