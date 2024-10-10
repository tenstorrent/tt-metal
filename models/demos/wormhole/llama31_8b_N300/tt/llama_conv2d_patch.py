# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch

import ttnn
from models.utility_functions import (
    nearest_32,
)
from models.common.lightweightmodule import LightweightModule

from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh


class TtLlamaConv2dPatch(LightweightModule):
    """Conv2D Patching layer.
    Column parallel over unfolded input.
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    Input: (bsz, in_channels, width, height)
    Output: (bsz, num_tokens, out_channels)
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        dtype,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.num_devices = len(self.mesh_device.get_devices())

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.bias = (
            ttnn.as_tensor(
                torch.reshape(state_dict[f"{state_dict_prefix}_linear.bias"], (1, -1)),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            if bias
            else None
        )

        self._unfold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

        weight = state_dict[f"{state_dict_prefix}_linear.weight"]
        pad_len = nearest_32(weight.shape[-1]) - weight.shape[-1]
        padding = torch.zeros(self.out_channels, pad_len, dtype=weight.dtype)
        padded_weight = torch.cat([weight, padding], dim=-1)
        padded_weight = padded_weight.permute(1, 0).reshape(1, 1, -1, self.out_channels)

        self._linear_weight = ttnn.as_tensor(
            padded_weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.program_config = None  # TODO: Update with actual program config

    def forward(self, x: torch.Tensor):
        x = self._unfold(x)
        x = x.permute(0, 2, 1)

        # Need to pad the last dimension of x to be a multiple of a tile
        pad_len = nearest_32(x.shape[-1]) - x.shape[-1]
        padding = torch.zeros((x.shape[0], x.shape[1], pad_len), dtype=x.dtype, device=x.device)
        x = torch.cat([x, padding], dim=-1)

        x = ttnn.as_tensor(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        out = ttnn.linear(
            x,
            self._linear_weight,
            bias=self.bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

        return out
