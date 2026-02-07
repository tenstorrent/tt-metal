# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32

_conv2d_collected = set()
if os.path.exists("llama_conv2d_patch_1d_performance.csv"):
    with open("llama_conv2d_patch_1d_performance.csv", "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                _conv2d_collected.add(",".join(row))


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
        model_name: str = "unknown",
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self._model_name = model_name

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

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: torch.Tensor):
        _file_exists = os.path.exists("llama_conv2d_patch_1d_performance.csv")
        with open("llama_conv2d_patch_1d_performance.csv", "a") as _f:
            if not _file_exists:
                _f.write(
                    "x_dtype,x_shape_0,x_shape_1,x_shape_2,x_shape_3,"
                    "in_channels,out_channels,kernel_size,stride,"
                    "linear_weight_shape_0,linear_weight_shape_1,linear_weight_shape_2,linear_weight_shape_3,"
                    "linear_weight_dtype,has_bias,device_shape_x,device_shape_y,model_name\n"
                )
            _dev_shape = list(self.mesh_device.shape) if hasattr(self.mesh_device, "shape") else [1, 1]
            _entry = (
                f"{x.dtype},{x.shape[0]},{x.shape[1]},{x.shape[2]},{x.shape[3]},"
                f"{self.in_channels},{self.out_channels},{self.kernel_size},{self.stride},"
                f"{self._linear_weight.shape[0]},{self._linear_weight.shape[1]},{self._linear_weight.shape[2]},{self._linear_weight.shape[3]},"
                f"{self._linear_weight.dtype},{self.bias is not None},{_dev_shape[0]},{_dev_shape[1]},{self._model_name}"
            )
            if _entry not in _conv2d_collected:
                _conv2d_collected.add(_entry)
                _f.write(f"{_entry}\n")

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
