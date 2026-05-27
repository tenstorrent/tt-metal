# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch

import ttnn


@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
def test_read_write_full_synchronize(device):
    zeros = torch.zeros([1, 1, 65536, 2048]).bfloat16()
    input = torch.randn([1, 1, 65536, 2048]).bfloat16()
    host_tensor = ttnn.from_torch(input, ttnn.bfloat16)
    dev_tensor0 = ttnn.from_torch(zeros, ttnn.bfloat16, device=device)
    dev_tensor1 = ttnn.from_torch(zeros, ttnn.bfloat16, device=device)
    ttnn.synchronize_device(device)
    output0 = dev_tensor0.cpu(blocking=False)
    output1 = dev_tensor1.cpu(blocking=False)
    ttnn.synchronize_device(device)
    output0 = ttnn.to_torch(output0)
    output1 = ttnn.to_torch(output1)
    eq = torch.equal(zeros, output0)
    assert eq
    eq = torch.equal(zeros, output1)
    assert eq

    ttnn.copy_host_to_device_tensor(host_tensor, dev_tensor0, 0)
    ttnn.copy_host_to_device_tensor(host_tensor, dev_tensor1, 1)
    ttnn.synchronize_device(device)
    output0 = dev_tensor0.cpu(blocking=False)
    output1 = dev_tensor1.cpu(blocking=False)
    ttnn.synchronize_device(device)
    output0 = ttnn.to_torch(output0)
    output1 = ttnn.to_torch(output1)
    eq = torch.equal(input, output0)
    assert eq
    eq = torch.equal(input, output1)
    assert eq


@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
def test_read_write_cq_synchronize(device):
    zeros = torch.zeros([1, 1, 65536, 2048]).bfloat16()
    input = torch.randn([1, 1, 65536, 2048]).bfloat16()
    host_tensor = ttnn.from_torch(input, ttnn.bfloat16)
    dev_tensor = ttnn.from_torch(zeros, ttnn.bfloat16, device=device)
    ttnn.copy_host_to_device_tensor(host_tensor, dev_tensor, 1)
    ttnn.synchronize_device(device, 1)
    output = dev_tensor.cpu(blocking=False)
    ttnn.synchronize_device(device, 0)
    output = ttnn.to_torch(output)
    eq = torch.equal(input, output)
    assert eq


def test_cpu_blocking_false_discarded_return_no_uaf(device):
    """Regression test for issue #43638.

    Tensor.cpu(blocking=False) must keep the destination host buffer alive until
    the async D2H read completes, even if the caller immediately discards the
    returned tensor.  Without the fix, the DistributedHostBuffer is freed before
    the NumaAwareExecutor reader thread finishes its memcpy → SIGSEGV in
    libumd read_from_sysmem.

    Use ~16 MB tensors so the async read is still in flight when CPython would
    normally garbage-collect the unreferenced return value.
    """
    # 1024 x 8192 x bfloat16 = ~16 MB; large enough that the async D2H is
    # likely still in flight when the discarded tensor goes out of scope.
    host = torch.randn(1024, 8192, dtype=torch.bfloat16)
    dev_tensor = ttnn.from_torch(host, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    for _ in range(20):
        # Deliberately discard the return value — the host buffer must remain
        # pinned internally until synchronize_device() completes.
        dev_tensor.cpu(blocking=False)
        ttnn.synchronize_device(device)
    # If we reach here without crashing, the fix is working.
