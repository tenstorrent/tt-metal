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
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(100):
        ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))
    ttnn.end_trace_capture(device, tid, cq_id=0)

    for i in range(5):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(device, tid)


@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 1996800, "dispatch_core_type": ttnn.DispatchCoreType.WORKER}], indirect=True
)
def test_with_ops_single_core(device):
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
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(100):
        ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=1, x=1))
    ttnn.end_trace_capture(device, tid, cq_id=0)

    for i in range(5):
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

    trace_ids = []
    for _ in range(3):
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))
        ttnn.end_trace_capture(device, tid, cq_id=0)
        trace_ids.append(tid)

    for _ in range(2):
        tid = ttnn.begin_trace_capture(device, cq_id=0)
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
    for _ in range(10):
        tid = ttnn.begin_trace_capture(device, cq_id=0)
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
