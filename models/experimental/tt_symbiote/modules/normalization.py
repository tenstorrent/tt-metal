# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Normalization layer implementations for TTNN."""

from torch import nn
import torch
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule


class TTNNLayerNormNoAffine(TTNNModule):
    """TTNN LayerNorm for layers with elementwise_affine=False (weight=None, bias=None)."""

    @classmethod
    def from_torch(cls, layer_norm: nn.LayerNorm):
        new_layer_norm = cls()
        new_layer_norm._fallback_torch_layer = layer_norm
        return new_layer_norm

    def preprocess_weights_impl(self):
        self.tt_weight = None
        self.tt_bias = None

    def move_weights_to_device_impl(self):
        pass

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        eps = getattr(self.torch_layer, "eps", 1e-5)
        return ttnn.layer_norm(
            input_tensor,
            weight=None,
            bias=None,
            epsilon=eps,
        )


class TTNNLayerNorm(TTNNModule):
    """TTNN-accelerated LayerNorm (with weight/bias)."""

    @classmethod
    def from_torch(cls, layer_norm: nn.LayerNorm):
        """Create TTNNLayerNorm from PyTorch LayerNorm."""
        if layer_norm.weight is None:
            return TTNNLayerNormNoAffine.from_torch(layer_norm)
        new_layer_norm = cls()
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


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TTNNRMSNorm(TTNNModule):
    @classmethod
    def from_torch(cls, rms_norm: DeepseekV2RMSNorm):
        """Create from PyTorch RMSNorm."""
        if rms_norm.weight is None:
            print(f"Warning: RMSNorm layer {rms_norm} has no weight. Using standard RMSNorm.")
            return rms_norm
        new_layer_norm = cls()
        new_layer_norm._fallback_torch_layer = rms_norm
        return new_layer_norm

    def preprocess_weights_impl(self):
        """Preprocess RMSNorm weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = DeepseekV2RMSNorm(hidden_size=1)
        self.tt_weight = ttnn.from_torch(
            self.torch_layer.weight.unsqueeze(0).expand(32, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device."""
        self.tt_weight = ttnn.to_device(self.tt_weight, self.device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.torch_layer.variance_epsilon)
        return x
