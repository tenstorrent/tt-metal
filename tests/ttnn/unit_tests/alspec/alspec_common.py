# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn
import torch


def get_buffer_address(tensor):
    """
    Get the buffer address of a multi-device tensor
    """
    addr = []
    for i, ten in enumerate(ttnn.get_device_tensors(tensor)):
        addr.append(ten.buffer_address())
        if len(addr) > 0:
            assert addr[i - 1] == addr[i], f"Expected {addr[i-1]} == {addr[i]}"
    return addr[0]


def create_multi_device_tensor(input_tensors, mesh_device, mem_config, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """
    Create a multi-device tensor from a list of input tensors
    """
    tt_tensors = []
    for i, t in enumerate(input_tensors):
        tt_tensors.append(ttnn.Tensor(t, dtype).to(layout).to(mesh_device.get_devices()[i], mem_config))
    tensor_mesh = ttnn.aggregate_as_tensor(tt_tensors)
    return tensor_mesh


def read_multi_device_tensor(tt_tensor):
    """
    Read a multi-device tensor and return a list of tensors
    """
    tensors = []
    for i, t in enumerate(ttnn.get_device_tensors(tt_tensor)):
        t = t.cpu().to_torch()
        tensors.append(t)
    return tensors
