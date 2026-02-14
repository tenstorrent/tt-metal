# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Normalization layer implementations for TTNN."""

from torch import nn

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule


class TTNNLayerNorm(TTNNModule):
    """TTNN-accelerated LayerNorm."""

    def __init__(self):
        super().__init__()

    @classmethod
    def from_torch(cls, layer_norm: nn.LayerNorm):
        """Create TTNNLayerNorm from PyTorch LayerNorm."""
        new_layer_norm = TTNNLayerNorm()
        new_layer_norm._fallback_torch_layer = layer_norm
        return new_layer_norm

    def preprocess_weights_impl(self):
        """Preprocess LayerNorm weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = nn.LayerNorm(normalized_shape=1)
        self.tt_weight = ttnn.from_torch(self.torch_layer.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.tt_bias = ttnn.from_torch(self.torch_layer.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device."""
        self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
        if self.tt_bias is not None:
            self.tt_bias = ttnn.to_device(self.tt_bias, self.device)

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through LayerNorm."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.layer_norm(
            input_tensor,
            weight=self.tt_weight,
            bias=self.tt_bias,
        )
        return tt_output
