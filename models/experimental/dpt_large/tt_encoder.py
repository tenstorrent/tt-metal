# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Optimized TTNN Encoder for DPT-Large.

Implements the 24-layer ViT encoder with bfloat8_b weight optimization
for efficient inference on Wormhole devices.

Key optimizations:
- bfloat8_b weights reduce memory bandwidth while maintaining PCC > 0.99
- Fused QKV projection reduces kernel launches
- Manual attention pattern (Q @ K^T) for better control
"""

import math
import torch
import ttnn


class DPTEncoderLayer:
    """Single transformer encoder layer for DPT-Large."""

    def __init__(self, state_dict, layer_idx: int, device):
        self.device = device
        self.num_heads = 16
        self.head_dim = 64
        self.scale = 1.0 / math.sqrt(self.head_dim)

        base = f"dpt.encoder.layer.{layer_idx}"

        # Fused QKV weights
        q_w = state_dict[f"{base}.attention.attention.query.weight"]
        q_b = state_dict[f"{base}.attention.attention.query.bias"]
        k_w = state_dict[f"{base}.attention.attention.key.weight"]
        k_b = state_dict[f"{base}.attention.attention.key.bias"]
        v_w = state_dict[f"{base}.attention.attention.value.weight"]
        v_b = state_dict[f"{base}.attention.attention.value.bias"]

        # Stack QKV weights [in, 3*out]
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0).T.contiguous()
        qkv_b = torch.cat([q_b, k_b, v_b], dim=0)

        # Use bfloat8_b for weights to reduce memory bandwidth
        self.qkv_weight = ttnn.from_torch(qkv_w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        self.qkv_bias = ttnn.from_torch(
            qkv_b.unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        # Output projection
        proj_w = state_dict[f"{base}.attention.output.dense.weight"].T.contiguous()
        proj_b = state_dict[f"{base}.attention.output.dense.bias"]
        self.proj_weight = ttnn.from_torch(proj_w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        self.proj_bias = ttnn.from_torch(
            proj_b.unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        # FFN weights
        ff1_w = state_dict[f"{base}.intermediate.dense.weight"].T.contiguous()
        ff1_b = state_dict[f"{base}.intermediate.dense.bias"]
        ff2_w = state_dict[f"{base}.output.dense.weight"].T.contiguous()
        ff2_b = state_dict[f"{base}.output.dense.bias"]

        self.ff1_weight = ttnn.from_torch(ff1_w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        self.ff1_bias = ttnn.from_torch(
            ff1_b.unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.ff2_weight = ttnn.from_torch(ff2_w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        self.ff2_bias = ttnn.from_torch(
            ff2_b.unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )

        # LayerNorm weights (keep bfloat16 for precision)
        ln1_w = state_dict[f"{base}.layernorm_before.weight"].unsqueeze(0)
        ln1_b = state_dict[f"{base}.layernorm_before.bias"].unsqueeze(0)
        ln2_w = state_dict[f"{base}.layernorm_after.weight"].unsqueeze(0)
        ln2_b = state_dict[f"{base}.layernorm_after.bias"].unsqueeze(0)

        self.ln1_weight = ttnn.from_torch(ln1_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.ln1_bias = ttnn.from_torch(ln1_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.ln2_weight = ttnn.from_torch(ln2_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.ln2_bias = ttnn.from_torch(ln2_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through encoder layer."""
        B = hidden_states.shape[0]
        N = hidden_states.shape[1] if len(hidden_states.shape) == 3 else hidden_states.shape[2]
        C = hidden_states.shape[-1]
        H = self.num_heads

        # LayerNorm 1
        normed = ttnn.layer_norm(hidden_states, weight=self.ln1_weight, bias=self.ln1_bias, epsilon=1e-12)

        # Fused QKV linear with bfloat8_b
        qkv = ttnn.linear(normed, self.qkv_weight, bias=self.qkv_bias, dtype=ttnn.bfloat8_b)

        # Reshape for attention
        if len(qkv.shape) == 4:
            qkv = ttnn.reshape(qkv, (B, N, 3 * C))

        # Split QKV and heads (transpose_key=True for Q @ K^T)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=H, transpose_key=True)

        # Attention: Q @ K^T
        attn_scores = ttnn.matmul(q, k, dtype=ttnn.bfloat8_b)
        attn_scores = ttnn.multiply(attn_scores, self.scale)

        # Softmax
        attn_probs = ttnn.softmax(attn_scores, dim=-1)

        # Context: attn @ V
        context = ttnn.matmul(attn_probs, v, dtype=ttnn.bfloat8_b)
        context = ttnn.transformer.concatenate_heads(context)

        # Output projection (bfloat16 for residual precision)
        attn_out = ttnn.linear(context, self.proj_weight, bias=self.proj_bias, dtype=ttnn.bfloat16)

        # Residual connection
        hidden_states = ttnn.add(hidden_states, attn_out)

        # LayerNorm 2
        normed = ttnn.layer_norm(hidden_states, weight=self.ln2_weight, bias=self.ln2_bias, epsilon=1e-12)

        # FFN with bfloat8_b
        ff_out = ttnn.linear(normed, self.ff1_weight, bias=self.ff1_bias, dtype=ttnn.bfloat8_b)
        ff_out = ttnn.gelu(ff_out)
        ff_out = ttnn.linear(ff_out, self.ff2_weight, bias=self.ff2_bias, dtype=ttnn.bfloat16)

        # Residual connection
        hidden_states = ttnn.add(hidden_states, ff_out)

        return hidden_states


class DPTEncoder:
    """
    DPT-Large encoder (24 transformer layers).

    Returns intermediate outputs at layers 5, 11, 17, 23 for the DPT neck.
    """

    def __init__(self, state_dict, device, num_layers=24):
        self.device = device
        self.layers = [DPTEncoderLayer(state_dict, i, device) for i in range(num_layers)]
        # DPT output layers (0-indexed: 4, 10, 16, 22 correspond to layers 5, 11, 17, 23)
        self.output_layer_indices = [4, 10, 16, 22]

    def __call__(self, hidden_states):
        """Forward pass returning intermediate outputs for DPT neck."""
        outputs = []
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if i in self.output_layer_indices:
                outputs.append(hidden_states)
        return outputs


def create_encoder(model_or_state_dict, device, num_layers=24):
    """
    Create optimized DPT encoder from HuggingFace model or state dict.

    Args:
        model_or_state_dict: DPTForDepthEstimation model or its state_dict
        device: ttnn device
        num_layers: Number of encoder layers (default 24 for DPT-Large)

    Returns:
        DPTEncoder instance
    """
    if hasattr(model_or_state_dict, "state_dict"):
        state_dict = model_or_state_dict.state_dict()
    else:
        state_dict = model_or_state_dict

    return DPTEncoder(state_dict, device, num_layers)
