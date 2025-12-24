"""Tensor function implementations for TTNN."""

import torch

import ttnn
from models.tt_symbiote.core.module import TTNNModule


class TorchPermute(torch.nn.Module):
    """Fallback PyTorch Permute activation function."""

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: torch.Tensor, perm) -> torch.Tensor:
        """Forward pass through Permute activation."""
        return input_tensor.permute(perm)


class TTNNPermute(TTNNModule):
    """TTNN-accelerated Permute activation function."""

    def __init__(self):
        super().__init__()
        self._fallback_torch_layer = TorchPermute()

    def forward(self, input_tensor: ttnn.Tensor, perm) -> ttnn.Tensor:
        """Forward pass through Permute activation."""
        tt_output = ttnn.permute(input_tensor, perm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tt_output


class TorchReshape(torch.nn.Module):
    """Fallback PyTorch Reshape activation function."""

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: torch.Tensor, shape) -> torch.Tensor:
        """Forward pass through Reshape activation."""
        return input_tensor.reshape(shape)


class TTNNReshape(TTNNModule):
    """TTNN-accelerated Reshape activation function."""

    def __init__(self):
        super().__init__()
        self._fallback_torch_layer = TorchReshape()

    def forward(self, input_tensor: ttnn.Tensor, shape) -> ttnn.Tensor:
        """Forward pass through Reshape activation."""
        tt_output = ttnn.reshape(input_tensor, shape, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tt_output
