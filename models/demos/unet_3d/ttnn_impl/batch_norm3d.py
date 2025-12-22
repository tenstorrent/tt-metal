# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import ttnn


class BatchNorm3D:
    def __init__(self, device, channels: int):
        self.channels = channels
        self.grid_size = device.compute_with_storage_grid_size()
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def init_params(self, device, params_dict: dict[str, torch.Tensor], module_prefix: Optional[str] = None):
        weight_tensor = params_dict[f"{module_prefix}.weight"] if module_prefix else params_dict["weight"]
        bias_tensor = params_dict[f"{module_prefix}.bias"] if module_prefix else params_dict["bias"]
        running_mean_tensor = (
            params_dict[f"{module_prefix}.running_mean"] if module_prefix else params_dict["running_mean"]
        )
        running_var_tensor = (
            params_dict[f"{module_prefix}.running_var"] if module_prefix else params_dict["running_var"]
        )

        self.weight = ttnn.from_torch(
            weight_tensor.reshape(1, self.channels, 1, 1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
            mesh_mapper=None,
            pad_value=0,
        )
        self.bias = ttnn.from_torch(
            bias_tensor.reshape(1, self.channels, 1, 1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
            mesh_mapper=None,
            pad_value=0,
        )
        self.running_mean = ttnn.from_torch(
            running_mean_tensor.reshape(1, self.channels, 1, 1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
            mesh_mapper=None,
            pad_value=0,
        )
        self.running_var = ttnn.from_torch(
            running_var_tensor.reshape(1, self.channels, 1, 1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
            mesh_mapper=None,
            pad_value=0,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        N, C, D, H, W = x.shape
        x = ttnn.reshape(x, (N, C, D * H, W))
        out = ttnn.batch_norm(
            x,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            eps=1e-05,
            momentum=0.1,
            training=False,
            output=x,
        )
        out = ttnn.reshape(out, (N, C, D, H, W))
        return out
