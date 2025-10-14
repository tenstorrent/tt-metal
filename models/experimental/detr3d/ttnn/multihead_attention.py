# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.common.lightweightmodule import LightweightModule


class TtnnMultiheadAttention(LightweightModule):
    def __init__(self, d_model, nhead, device, parameters=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.device = device

        # These will be set from PyTorch weights
        self.q_weight = None
        self.k_weight = None
        self.v_weight = None
        self.q_bias = None
        self.k_bias = None
        self.v_bias = None
        self.out_weight = None
        self.out_bias = None

        # Load parameters if provided
        if parameters is not None:
            self.load_parameters(parameters)

    def load_parameters(self, parameters):
        """Load preprocessed attention parameters"""
        self.q_weight = parameters.get("q_weight")
        self.k_weight = parameters.get("k_weight")
        self.v_weight = parameters.get("v_weight")
        self.q_bias = parameters.get("q_bias")
        self.k_bias = parameters.get("k_bias")
        self.v_bias = parameters.get("v_bias")
        self.out_weight = parameters.get("out_weight")
        self.out_bias = parameters.get("out_bias")

    def forward(self, query, key, value, attn_mask=None):
        # Apply linear projections separately to avoid concat issues
        q = ttnn.linear(query, self.q_weight, bias=self.q_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.linear(key, self.k_weight, bias=self.k_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.linear(value, self.v_weight, bias=self.v_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Get dimensions for reshaping
        batch_size = q.shape[0]
        q_seq_len = q.shape[1]
        k_seq_len = k.shape[1]

        # Reshape each tensor separately for multi-head attention
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        q = ttnn.reshape(q, (batch_size, q_seq_len, self.nhead, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))

        k = ttnn.reshape(k, (batch_size, k_seq_len, self.nhead, self.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))

        v = ttnn.reshape(v, (batch_size, k_seq_len, self.nhead, self.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Use SDPA for attention computation
        context = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=1.0 / math.sqrt(self.head_dim),
            attn_mask=attn_mask,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Concatenate heads back to original format
        context = ttnn.transformer.concatenate_heads(context, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Output projection
        output = ttnn.linear(context, self.out_weight, bias=self.out_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return output
