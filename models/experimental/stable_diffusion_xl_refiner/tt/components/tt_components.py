# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_linear_params,
)
from models.experimental.stable_diffusion_xl_refiner.tt.components.convolution_layer import ConvolutionLayer


# ResNet components


class ResNetTimeEmbedding:
    def __init__(self, device, weights, bias, conv_w_dtype=ttnn.bfloat16):
        self.device = device

        self.temb_weights, self.temb_bias = prepare_linear_params(device, weights, bias, conv_w_dtype)

    def forward(self, hidden_states, temb):
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        temb = ttnn.silu(temb)
        temb = ttnn.linear(
            temb,
            self.temb_weights,
            bias=self.temb_bias,
        )
        return ttnn.add(hidden_states, temb, use_legacy=False)


class ResNetShortcutConnection:
    def __init__(self, device, use_conv_shortcut=False, conv_weights=None, conv_bias=None, conv_config=None):
        self.device = device
        self.use_conv_shortcut = use_conv_shortcut

        if self.use_conv_shortcut and conv_weights is not None:
            self.shortcut_conv = ConvolutionLayer(device, conv_weights, conv_bias, conv_config)
        else:
            self.shortcut_conv = None

    def forward(self, input_tensor, hidden_states, input_shape):
        B, C, H, W = input_shape

        shortcut = input_tensor

        if self.shortcut_conv is not None:
            shortcut, [C, H, W] = self.shortcut_conv.forward(shortcut, B, C, H, W)

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        shortcut = ttnn.to_layout(shortcut, ttnn.TILE_LAYOUT)
        result = ttnn.add(hidden_states, shortcut, use_legacy=False)
        return ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG), [C, H, W]


# Transformer components
class TransformerBlockLayerNorm:
    def __init__(self, device, weights, bias, eps=1e-5):
        self.device = device
        self.eps = eps

        self.norm_weights = ttnn.from_torch(weights, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self.norm_bias = (
            ttnn.from_torch(bias, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT) if bias is not None else None
        )

    def forward(self, hidden_states):
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.norm_weights,
            bias=self.norm_bias,
            epsilon=self.eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return hidden_states
