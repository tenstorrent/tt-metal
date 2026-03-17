# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Transformer block and stack implementations for TTNN."""

import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.attention import TTNNNoTPAttention
from models.experimental.tt_symbiote.modules.linear import TTNNNoTPFeedForward


class TTNNNoTPTransformerBlock(TTNNModule):
    """
    No Tensor Parallelism Transformer Block using TTNN operations.

    Implements pre-norm transformer block with attention and feedforward.
    """

    def __init__(
        self,
        n_heads,
        dim,
        head_dim,
        layer_id: int,
    ):
        """
        Initialize transformer block.

        Args:
            cfg: Configuration dict
            layer_id: Layer index
            weights: PyTorch weights dict (optional)
            device: TTNN device
        """
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = head_dim
        self.layer_id = layer_id
        self.layernorm_epsilon = 1e-5
        self.torch_layer_cp = None
        self.self_attn = None
        self.mlp = None

    @classmethod
    def from_torch(cls, NoTPTransformerBlock):
        """Create TTNN module from PyTorch equivalent."""
        new_NoTPTransformerBlock = cls(
            NoTPTransformerBlock.n_heads,
            NoTPTransformerBlock.dim,
            NoTPTransformerBlock.head_dim,
            NoTPTransformerBlock.layer_id,
        )
        new_NoTPTransformerBlock.self_attn = TTNNNoTPAttention.from_torch(NoTPTransformerBlock.self_attn)
        new_NoTPTransformerBlock.mlp = TTNNNoTPFeedForward.from_torch(NoTPTransformerBlock.mlp)

        new_NoTPTransformerBlock.torch_layer_cp = NoTPTransformerBlock
        new_NoTPTransformerBlock._fallback_torch_layer = NoTPTransformerBlock
        return new_NoTPTransformerBlock

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        ln1_weight = self.torch_layer_cp.layer_norm1.weight.data
        ln1_bias = self.torch_layer_cp.layer_norm1.bias.data
        ln2_weight = self.torch_layer_cp.layer_norm2.weight.data
        ln2_bias = self.torch_layer_cp.layer_norm2.bias.data

        self.layer_norm1_weight = ttnn.from_torch(
            ln1_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layer_norm1_bias = ttnn.from_torch(
            ln1_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layer_norm2_weight = ttnn.from_torch(
            ln2_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.layer_norm2_bias = ttnn.from_torch(
            ln2_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.self_attn.preprocess_weights_impl()
        self.mlp.preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        self.layer_norm1_weight = ttnn.to_device(self.layer_norm1_weight, self.device)
        self.layer_norm1_bias = ttnn.to_device(self.layer_norm1_bias, self.device)
        self.layer_norm2_weight = ttnn.to_device(self.layer_norm2_weight, self.device)
        self.layer_norm2_bias = ttnn.to_device(self.layer_norm2_bias, self.device)
        self.self_attn.move_weights_to_device_impl()
        self.mlp.move_weights_to_device_impl()

    def deallocate_weights_impl(self):
        """Deallocate device memory."""
        ttnn.deallocate(self.layer_norm1_weight)
        ttnn.deallocate(self.layer_norm1_bias)
        ttnn.deallocate(self.layer_norm2_weight)
        ttnn.deallocate(self.layer_norm2_bias)
        self.self_attn.deallocate_weights_impl()
        self.mlp.deallocate_weights_impl()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of transformer block.

        Args:
            x: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        if isinstance(x, TorchTTNNTensor):
            x = x.to_ttnn
            x = ttnn.to_device(x, device=self.device)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        if isinstance(x, torch.Tensor):
            x = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Pre-norm attention
        residual = ttnn.layer_norm(
            x,
            weight=self.layer_norm1_weight,
            bias=self.layer_norm1_bias,
            epsilon=self.layernorm_epsilon,
        )

        residual = self.self_attn.forward(residual)
        if isinstance(residual, torch.Tensor):
            residual = ttnn.from_torch(
                residual,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        h = ttnn.add(x, residual, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(residual)

        # Pre-norm feedforward
        out = ttnn.layer_norm(
            h,
            weight=self.layer_norm2_weight,
            bias=self.layer_norm2_bias,
            epsilon=self.layernorm_epsilon,
        )
        out = self.mlp.forward(out)
        if isinstance(out, torch.Tensor):
            out = ttnn.from_torch(
                out,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        out = ttnn.add(h, out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h)

        out = ttnn.to_torch(out)
        return out


class TTNNNoTPTransformer(TTNNModule):
    """
    No Tensor Parallelism Transformer using TTNN operations.

    Stack of transformer blocks.
    """

    def __init__(
        self,
        cfg,
        num_layers,
    ):
        """
        Initialize transformer.

        Args:
            cfg: Configuration dict
            weights: PyTorch weights dict (optional)
            device: TTNN device
        """
        super().__init__()

        self.cfg = cfg
        self.num_layers = num_layers
        self.torch_layer_cp = None
        self.layers = []

    @classmethod
    def from_torch(cls, NoTPTransformer):
        """Create TTNN module from PyTorch equivalent."""
        new_NoTPTransformer = cls(NoTPTransformer.cfg, NoTPTransformer.num_layers)

        for layer_id in range(new_NoTPTransformer.num_layers):
            layer = TTNNNoTPTransformerBlock.from_torch(NoTPTransformer.layers[layer_id])
            new_NoTPTransformer.layers.append(layer)

        new_NoTPTransformer.torch_layer_cp = NoTPTransformer
        new_NoTPTransformer._fallback_torch_layer = NoTPTransformer
        return new_NoTPTransformer

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        for layer in self.layers:
            layer.preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        for layer in self.layers:
            layer.move_weights_to_device_impl()

    def deallocate_weights_impl(self):
        """Deallocate device memory."""
        for layer in self.layers:
            layer.deallocate_weights_impl()

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of transformer.

        Args:
            hidden_states: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        if isinstance(hidden_states, TorchTTNNTensor):
            hidden_states = hidden_states.to_ttnn
            hidden_states = ttnn.to_device(hidden_states, device=self.device)
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        for layer in self.layers:
            hidden_states = layer.forward(hidden_states)

        return hidden_states
