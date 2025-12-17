"""Tensor function implementations for TTNN."""

import ttnn
from models.tt_symbiote.core.module import TTNNModule


class TTNNPermute(TTNNModule):
    """TTNN-accelerated Permute activation function."""

    def __init__(self, perm):
        super().__init__()
        self.perm = perm

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through Permute activation."""
        tt_output = ttnn.permute(input_tensor, self.perm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tt_output
