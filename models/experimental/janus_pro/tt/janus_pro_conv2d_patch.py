"""
Conv2d patch-embedding for the Janus-Pro-7B vision model.

The convolution weight is folded into a 2D matrix so the patch projection runs
as a single ttnn.linear over the unfolded input. A 4D conv weight is reshaped to
(out_channels, in_channels * kernel_size**2). For Janus-Pro this inner dimension
(3 * 16**2 = 768) is already a tile multiple, so no padding is applied.
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtJanusProConv2dPatch(LightweightModule):
    """Conv2D Patching layer.
    Column parallel over unfolded input.
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride: Stride for convolution.
        bias: Use bias in Conv2d.
    Input: (bsz, in_channels, height, width)
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
        self.num_devices = self.mesh_device.get_num_devices()

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
        if weight.ndim == 4:
            weight = weight.view(out_channels, -1)
        weight = weight.permute(1, 0).reshape(1, 1, -1, self.out_channels)

        self._linear_weight = ttnn.as_tensor(
            weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: torch.Tensor):
        x = self._unfold(x)
        x = x.permute(0, 2, 1)

        x = ttnn.as_tensor(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Bias applied outside ttnn.linear to avoid the FUSE_BIAS matmul kernel path.
        out = ttnn.linear(
            x,
            self._linear_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

        if self.bias is not None:
            out = ttnn.add(out, self.bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ttnn.deallocate(x)

        return out
