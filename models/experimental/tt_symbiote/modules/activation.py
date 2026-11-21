# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Activation function implementations for TTNN."""

import torch

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule


class TTNNSilu(TTNNModule):
    """TTNN-accelerated SiLU activation function."""

    def __init__(self):
        super().__init__()
        self._fallback_torch_layer = torch.nn.SiLU()

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through SiLU activation."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.silu(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tt_output


class TTNNReLU(TTNNModule):
    """TTNN-accelerated ReLU activation function."""

    def __init__(self):
        super().__init__()
        self._fallback_torch_layer = torch.nn.ReLU()

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through ReLU activation."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.relu(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tt_output


class TTNNGelu(TTNNModule):
    """TTNN-accelerated GELU activation function."""

    def __init__(self, approximate="none"):
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(f"GELU approximate must be 'none' or 'tanh', got '{approximate}'.")
        self.approximate = approximate
        self._fallback_torch_layer = torch.nn.GELU(approximate=approximate)

    @classmethod
    def from_torch(cls, gelu: torch.nn.GELU) -> "TTNNGelu":
        """Create TTNNGelu from PyTorch GELU layer."""
        return cls(approximate=gelu.approximate)

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through GELU activation."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        variant = ttnn.GeluVariant.Tanh if self.approximate == "tanh" else ttnn.GeluVariant.Accurate
        tt_output = ttnn.gelu(input_tensor, variant=variant, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tt_output
