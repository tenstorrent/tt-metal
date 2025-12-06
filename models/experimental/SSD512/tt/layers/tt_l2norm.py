# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.common.utility_functions import torch_to_tt_tensor_rm


class TtL2Norm:
    def __init__(self, n_channels, scale=20, eps=1e-10, device=None):
        """Create an L2Norm module with learnable per-channel scale."""
        self.n_channels = n_channels
        self.eps = eps
        self.device = device
        self.dtype = ttnn.bfloat8_b
        self.weight = ttnn.full([1, n_channels, 1, 1], scale, device=device)

    def __call__(self, x, memory_config=None):
        """Apply L2 normalization and learned scale."""
        x_shape = x.shape
        if len(x_shape) == 4:
            dim1_val = x_shape[1]
            dim3_val = x_shape[3]
            if dim1_val == self.n_channels:
                x_nchw_ttnn = x
            elif dim3_val == self.n_channels:
                x_nchw_ttnn = ttnn.permute(x, (0, 3, 1, 2))
            else:
                x_nchw_ttnn = x
        else:
            x_nchw_ttnn = x

        batch_size, channels, height, width = x_nchw_ttnn.shape
        tensor_size_estimate = batch_size * height * width * channels

        if memory_config is None:
            layer_memory_config = ttnn.L1_MEMORY_CONFIG
        else:
            layer_memory_config = memory_config

        if x_nchw_ttnn.layout != ttnn.TILE_LAYOUT:
            x_nchw_ttnn = ttnn.to_layout(x_nchw_ttnn, ttnn.TILE_LAYOUT)

        if x_nchw_ttnn.memory_config() != layer_memory_config:
            x_nchw_ttnn = ttnn.to_memory_config(x_nchw_ttnn, layer_memory_config)

        squared = ttnn.mul(x_nchw_ttnn, x_nchw_ttnn, memory_config=layer_memory_config)
        squared = ttnn.to_layout(squared, layout=ttnn.TILE_LAYOUT)
        sum_result = ttnn.sum(squared, dim=1, keepdim=True, memory_config=layer_memory_config)
        # Add eps using ttnn operations
        eps_tensor = ttnn.full_like(sum_result, self.eps, memory_config=layer_memory_config)
        norm = ttnn.sqrt(
            ttnn.add(sum_result, eps_tensor, memory_config=layer_memory_config), memory_config=layer_memory_config
        )
        norm = ttnn.to_layout(norm, layout=ttnn.TILE_LAYOUT)
        x_norm = ttnn.div(x_nchw_ttnn, norm, memory_config=layer_memory_config)

        out = ttnn.mul(x_norm, self.weight, memory_config=layer_memory_config)

        return out


def l2norm(input_tensor, num_channels=512, scale=20.0, device=None):
    """Function wrapper for TtL2Norm for convenience."""
    l2norm_module = TtL2Norm(n_channels=num_channels, scale=scale, device=device)

    if isinstance(input_tensor, torch.Tensor):
        input_ttnn = torch_to_tt_tensor_rm(input_tensor, device=device)
        output = l2norm_module(input_ttnn)
        return output
    else:
        output = l2norm_module(input_tensor)
        return output
