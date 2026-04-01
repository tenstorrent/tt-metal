# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Transformer block and stack implementations for TTNN (DeepSeek-OCR ViT)."""

import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.attention import TTNNNoTPAttention
from models.experimental.tt_symbiote.modules.linear import TTNNNoTPFeedForward


class TTNNNoTPTransformerBlock(TTNNModule):
    """Pre-norm transformer block with attention and feedforward (no tensor parallelism)."""

    def __init__(self, n_heads, dim, head_dim, layer_id: int):
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
        obj = cls(
            NoTPTransformerBlock.n_heads,
            NoTPTransformerBlock.dim,
            NoTPTransformerBlock.head_dim,
            NoTPTransformerBlock.layer_id,
        )
        obj.self_attn = TTNNNoTPAttention.from_torch(NoTPTransformerBlock.self_attn)
        obj.mlp = TTNNNoTPFeedForward.from_torch(NoTPTransformerBlock.mlp)
        obj.torch_layer_cp = NoTPTransformerBlock
        obj._fallback_torch_layer = NoTPTransformerBlock
        return obj

    def preprocess_weights_impl(self):
        ln1_w = self.torch_layer_cp.layer_norm1.weight.data
        ln1_b = self.torch_layer_cp.layer_norm1.bias.data
        ln2_w = self.torch_layer_cp.layer_norm2.weight.data
        ln2_b = self.torch_layer_cp.layer_norm2.bias.data

        self.layer_norm1_weight = ttnn.from_torch(ln1_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.layer_norm1_bias = ttnn.from_torch(ln1_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.layer_norm2_weight = ttnn.from_torch(ln2_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.layer_norm2_bias = ttnn.from_torch(ln2_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.self_attn.preprocess_weights_impl()
        self.mlp.preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        self.layer_norm1_weight = ttnn.to_device(self.layer_norm1_weight, self.device)
        self.layer_norm1_bias = ttnn.to_device(self.layer_norm1_bias, self.device)
        self.layer_norm2_weight = ttnn.to_device(self.layer_norm2_weight, self.device)
        self.layer_norm2_bias = ttnn.to_device(self.layer_norm2_bias, self.device)
        self.self_attn.move_weights_to_device_impl()
        self.mlp.move_weights_to_device_impl()

    def deallocate_weights_impl(self):
        ttnn.deallocate(self.layer_norm1_weight)
        ttnn.deallocate(self.layer_norm1_bias)
        ttnn.deallocate(self.layer_norm2_weight)
        ttnn.deallocate(self.layer_norm2_bias)
        self.self_attn.deallocate_weights_impl()
        self.mlp.deallocate_weights_impl()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if isinstance(x, torch.Tensor):
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = ttnn.layer_norm(
            x, weight=self.layer_norm1_weight, bias=self.layer_norm1_bias, epsilon=self.layernorm_epsilon
        )
        residual = self.self_attn.forward(residual)
        if isinstance(residual, torch.Tensor):
            residual = ttnn.from_torch(residual, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        h = ttnn.add(x, residual, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(residual)

        out = ttnn.layer_norm(
            h, weight=self.layer_norm2_weight, bias=self.layer_norm2_bias, epsilon=self.layernorm_epsilon
        )
        out = self.mlp.forward(out)
        if isinstance(out, torch.Tensor):
            out = ttnn.from_torch(out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        out = ttnn.add(h, out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h)
        out = ttnn.to_torch(out)
        return out


class TTNNNoTPTransformer(TTNNModule):
    """Stack of NoTP transformer blocks."""

    def __init__(self, cfg, num_layers):
        super().__init__()
        self.cfg = cfg
        self.num_layers = num_layers
        self.torch_layer_cp = None
        self.layers = []

    @classmethod
    def from_torch(cls, NoTPTransformer):
        obj = cls(NoTPTransformer.cfg, NoTPTransformer.num_layers)
        for layer_id in range(obj.num_layers):
            layer = TTNNNoTPTransformerBlock.from_torch(NoTPTransformer.layers[layer_id])
            obj.layers.append(layer)
        obj.torch_layer_cp = NoTPTransformer
        obj._fallback_torch_layer = NoTPTransformer
        return obj

    def preprocess_weights_impl(self):
        for layer in self.layers:
            layer.preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        for layer in self.layers:
            layer.move_weights_to_device_impl()

    def deallocate_weights_impl(self):
        for layer in self.layers:
            layer.deallocate_weights_impl()

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states)
        return hidden_states
