# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn
import torch


def create_multi_device_tensors(input_tensors, mesh_device, mem_config, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    tt_tensors = []
    for i, t in enumerate(input_tensors):
        tt_tensors.append(ttnn.Tensor(t, dtype).to(layout).to(mesh_device.get_devices()[i], mem_config))
    tensor_mesh = ttnn.aggregate_as_tensor(tt_tensors)
    return tensor_mesh


def read_multi_device_tensor(tt_tensor):
    tensors = []
    for i, t in enumerate(ttnn.get_device_tensors(tt_tensor)):
        t = t.cpu().to_torch()
        tensors.append(t)
    return tensors


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


class ModelOps(torch.nn.Module):
    def __init__(self, mesh_device, num_devices=2):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_devices = num_devices

        M, K, N = (32, 2048, 172 * 1024)

        # Create the input tensors
        self.input_tensors = []
        self.weights_tensors = []
        for i in range(num_devices):
            self.input_tensors.append(torch.rand(1, 1, M, K))
            self.weights_tensors.append(torch.rand(1, 1, K, N))

        self.input_tensor_tt = create_multi_device_tensors(self.input_tensors, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
        self.weight_tensor_tt = create_multi_device_tensors(
            self.weights_tensors, mesh_device, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
        )

        ones = torch.ones([64, 1, 32, 32])
        ones = [ones for _i in range(num_devices)]

        tt_ones_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(32, 32),
            core_grid=ttnn.num_cores_to_corerangeset(64, mesh_device.compute_with_storage_grid_size(), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.tt_ones = create_multi_device_tensors(
            ones, mesh_device, tt_ones_mem_cfg, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32
        )

        self.out_add = create_multi_device_tensors(
            [torch.ones(1, 1, 32, 32)] * num_devices, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )

    def forward(self):
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        )

        mm_out = ttnn.matmul(
            self.input_tensor_tt,
            self.weight_tensor_tt,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Small tensor to that will be pushed to host to measure e2e latency
        tt_out = ttnn.add(self.out_add, self.out_add)

        return tt_out
