# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch

import ttnn
from models.utility_functions import is_wormhole_b0


def is_eth_dispatch():
    return os.environ.get("WH_ARCH_YAML") == "wormhole_b0_80_arch_eth_dispatch.yaml"


@pytest.mark.skipif(is_wormhole_b0() and not is_eth_dispatch(), reason="Requires eth dispatch to run on WH")
@pytest.mark.parametrize("device_params", [{"num_hw_cqs": 2}], indirect=True)
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


@pytest.mark.skipif(is_wormhole_b0() and not is_eth_dispatch(), reason="Requires eth dispatch to run on WH")
@pytest.mark.parametrize("device_params", [{"num_hw_cqs": 2}], indirect=True)
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
