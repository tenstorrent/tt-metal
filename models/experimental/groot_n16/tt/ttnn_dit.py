# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
AlternateVLDiT (Diffusion Transformer) - TTNN Implementation for GR00T N1.6.

32-layer transformer with alternating attention pattern:
    - Even blocks: cross-attention (Q from actions [1536], K/V from backbone [2048])
    - Odd blocks: self-attention (Q/K/V all from actions [1536])

All blocks share the same weight structure (attn1 + ff + norm1), but even blocks
have K/V weights of shape [1536, 2048] while odd blocks have [1536, 1536].

Uses AdaLN (Adaptive Layer Normalization) conditioned on timestep.
Output: two-stage projection with SiLU-gated split.

Inner dim: 32 heads * 48 head_dim = 1536
Output dim: 1024 (via proj_out_1 [1536→3072, SiLU gate] -> proj_out_2 [1536→1024])
"""

import math
from typing import Any, Dict, List, Optional

import torch

import ttnn
from models.experimental.groot_n16.common.configs import DiTConfig
from models.experimental.groot_n16.tt.ttnn_common import (
    CORE_GRID_BH,
    preprocess_linear_weight,
    preprocess_linear_bias,
)


class AdaLayerNormTTNN:
    """
    Adaptive Layer Normalization: (1 + scale(t)) * LN(x) + shift(t)

    norm1.linear.weight shape: [3072, 1536] -> projects timestep to 2*inner_dim (scale + shift)
    """

    def __init__(self, weights: Dict[str, torch.Tensor], prefix: str, hidden_size: int, device: Any):
        self.device = device
        self.hidden_size = hidden_size

        # The AdaLN in GR00T uses a linear to project timestep_emb -> scale, shift
        # No separate LayerNorm weight/bias in the actual model (just the linear)
        proj_w = weights.get(f"{prefix}linear.weight")
        proj_b = weights.get(f"{prefix}linear.bias")
        if proj_w is not None:
            self.proj_weight = preprocess_linear_weight(proj_w, device)
            self.proj_bias = preprocess_linear_bias(proj_b, device) if proj_b is not None else None

    def __call__(self, hidden_states: ttnn.Tensor, timestep_emb: ttnn.Tensor) -> ttnn.Tensor:
        """Apply AdaLN: (1 + scale) * LN(x) + shift."""
        # Project timestep -> scale, shift [batch, 1, 2*hidden]
        scale_shift = ttnn.linear(
            timestep_emb, self.proj_weight, bias=self.proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH,
        )

        scale = ttnn.slice(scale_shift, [0, 0, 0], [scale_shift.shape[0], 1, self.hidden_size])
        shift = ttnn.slice(scale_shift, [0, 0, self.hidden_size],
                           [scale_shift.shape[0], 1, 2 * self.hidden_size])
        ttnn.deallocate(scale_shift)

        # LayerNorm (without learned weight/bias)
        normed = ttnn.layer_norm(
            hidden_states, epsilon=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # (1 + scale) * normed + shift
        ones = ttnn.ones_like(scale)
        scale_p1 = ttnn.add(ones, scale)
        ttnn.deallocate(ones)
        ttnn.deallocate(scale)

        output = ttnn.mul(normed, scale_p1)
        ttnn.deallocate(normed)
        ttnn.deallocate(scale_p1)

        output = ttnn.add(output, shift)
        ttnn.deallocate(shift)
        return output


class DiTAttentionTTNN:
    """
    Attention for DiT blocks. Handles both self-attention and cross-attention
    based on the weight shapes:
        - Even blocks: Q [1536,1536], K/V [1536,2048] (cross-attention)
        - Odd blocks: Q/K/V all [1536,1536] (self-attention)
    """

    def __init__(self, weights: Dict[str, torch.Tensor], prefix: str,
                 num_heads: int, head_dim: int, device: Any):
        self.device = device
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        for proj in ["to_q", "to_k", "to_v", "to_out.0"]:
            w = weights.get(f"{prefix}{proj}.weight")
            b = weights.get(f"{prefix}{proj}.bias")
            attr = proj.replace(".", "_")
            if w is not None:
                setattr(self, f"{attr}_weight", preprocess_linear_weight(w, device))
                setattr(self, f"{attr}_bias",
                        preprocess_linear_bias(b, device) if b is not None else None)

        # Detect if this is cross-attention based on K weight input dim
        k_w = weights.get(f"{prefix}to_k.weight")
        self.is_cross_attention = (k_w is not None and k_w.shape[1] != k_w.shape[0])

    def __call__(self, hidden_states: ttnn.Tensor,
                 encoder_hidden_states: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """
        Args:
            hidden_states: [B, q_seq, inner_dim] action tokens (Q source)
            encoder_hidden_states: [B, kv_seq, backbone_dim] backbone features (K/V source for cross-attn)
        """
        scale = 1.0 / math.sqrt(self.head_dim)

        # Q always from action tokens
        q = ttnn.linear(hidden_states, self.to_q_weight, bias=self.to_q_bias,
                        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)

        # K/V from backbone (cross-attn) or action tokens (self-attn)
        kv_source = encoder_hidden_states if (self.is_cross_attention and encoder_hidden_states is not None) else hidden_states
        k = ttnn.linear(kv_source, self.to_k_weight, bias=self.to_k_bias,
                        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)
        v = ttnn.linear(kv_source, self.to_v_weight, bias=self.to_v_bias,
                        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)

        batch_size = hidden_states.shape[0]
        q_seq = hidden_states.shape[1]
        kv_seq = kv_source.shape[1]

        # Reshape for multi-head: [B, seq, dim] -> [B, heads, seq, head_dim]
        q = ttnn.reshape(q, (batch_size, q_seq, self.num_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (batch_size, kv_seq, self.num_heads, self.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (batch_size, kv_seq, self.num_heads, self.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))

        q = ttnn.mul(q, scale)
        attn = ttnn.matmul(q, k, memory_config=ttnn.L1_MEMORY_CONFIG,
                           dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        attn = ttnn.softmax_in_place(attn, numeric_stable=True)

        context = ttnn.matmul(attn, v, memory_config=ttnn.L1_MEMORY_CONFIG,
                              dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)
        ttnn.deallocate(attn)
        ttnn.deallocate(v)

        # [B, heads, q_seq, head_dim] -> [B, q_seq, dim]
        context = ttnn.permute(context, (0, 2, 1, 3))
        context = ttnn.reshape(context, (batch_size, q_seq, self.inner_dim))

        output = ttnn.linear(context, self.to_out_0_weight, bias=self.to_out_0_bias,
                             memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)
        ttnn.deallocate(context)
        return output


class DiTFFNTTNN:
    """GEGLU FFN: gate projection (split + GELU gate) -> down projection."""

    def __init__(self, weights: Dict[str, torch.Tensor], prefix: str, device: Any):
        self.device = device

        w_gate = weights.get(f"{prefix}net.0.proj.weight")
        b_gate = weights.get(f"{prefix}net.0.proj.bias")
        w_down = weights.get(f"{prefix}net.2.weight")
        b_down = weights.get(f"{prefix}net.2.bias")

        if w_gate is not None:
            self.gate_weight = preprocess_linear_weight(w_gate, device)
            self.gate_bias = preprocess_linear_bias(b_gate, device) if b_gate is not None else None
            self.down_weight = preprocess_linear_weight(w_down, device)
            self.down_bias = preprocess_linear_bias(b_down, device) if b_down is not None else None
            self.gate_out_dim = w_gate.shape[0]  # 6144 = 2 * 3072

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        gate_proj = ttnn.linear(hidden_states, self.gate_weight, bias=self.gate_bias,
                                memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)

        half_dim = self.gate_out_dim // 2
        gate = ttnn.slice(gate_proj, [0, 0, 0], [gate_proj.shape[0], gate_proj.shape[1], half_dim])
        value = ttnn.slice(gate_proj, [0, 0, half_dim],
                           [gate_proj.shape[0], gate_proj.shape[1], self.gate_out_dim])
        ttnn.deallocate(gate_proj)

        gate = ttnn.gelu(gate)
        intermediate = ttnn.mul(gate, value)
        ttnn.deallocate(gate)
        ttnn.deallocate(value)

        output = ttnn.linear(intermediate, self.down_weight, bias=self.down_bias,
                             memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)
        ttnn.deallocate(intermediate)
        return output


class DiTBlockTTNN:
    """
    Single AlternateVLDiT block.

    Even blocks: AdaLN -> CrossAttn(Q=actions, K/V=backbone) -> Residual -> FFN -> Residual
    Odd blocks: AdaLN -> SelfAttn(Q/K/V=actions) -> Residual -> FFN -> Residual
    """

    def __init__(self, weights: Dict[str, torch.Tensor], prefix: str,
                 config: DiTConfig, block_idx: int, device: Any):
        self.device = device
        self.block_idx = block_idx
        inner_dim = config.inner_dim

        self.adaln = AdaLayerNormTTNN(weights, f"{prefix}norm1.", inner_dim, device)
        self.attn = DiTAttentionTTNN(
            weights, f"{prefix}attn1.",
            config.num_attention_heads, config.attention_head_dim, device,
        )
        self.ffn = DiTFFNTTNN(weights, f"{prefix}ff.", device)

    def __call__(self, hidden_states: ttnn.Tensor, timestep_emb: ttnn.Tensor,
                 backbone_features: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        # AdaLN
        normed = self.adaln(hidden_states, timestep_emb)

        # Attention (cross or self based on weight shapes)
        attn_out = self.attn(normed, encoder_hidden_states=backbone_features)

        # Residual
        hidden_states = ttnn.add(hidden_states, attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        # FFN + Residual
        ffn_out = self.ffn(hidden_states)
        hidden_states = ttnn.add(hidden_states, ffn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ffn_out)

        return hidden_states


class AlternateVLDiTTTNN:
    """
    Complete AlternateVLDiT for GR00T N1.6.

    32 blocks + SiLU-gated output projection.
    proj_out_1: 1536 -> 3072 (split into 2x1536, SiLU gate)
    proj_out_2: 1536 -> 1024
    """

    def __init__(self, config: DiTConfig, weights: Dict[str, torch.Tensor], device: Any):
        self.config = config
        self.device = device

        self.blocks = []
        for i in range(config.num_layers):
            prefix = f"transformer_blocks.{i}."
            self.blocks.append(DiTBlockTTNN(weights, prefix, config, i, device))

        # Output projection with SiLU gating
        for proj_name in ["proj_out_1", "proj_out_2"]:
            w = weights.get(f"{proj_name}.weight")
            b = weights.get(f"{proj_name}.bias")
            if w is not None:
                setattr(self, f"{proj_name}_weight", preprocess_linear_weight(w, device))
                setattr(self, f"{proj_name}_bias",
                        preprocess_linear_bias(b, device) if b is not None else None)

        # proj_out_1 output dim for gating (3072 = 2 * 1536)
        w1 = weights.get("proj_out_1.weight")
        self.proj_out_1_dim = w1.shape[0] if w1 is not None else 3072

    def __call__(self, hidden_states: ttnn.Tensor, timestep_emb: ttnn.Tensor,
                 backbone_features: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward through 32 DiT blocks + output projection.

        Args:
            hidden_states: [B, action_seq, 1536]
            timestep_emb: [B, 1, 1536]
            backbone_features: [B, vl_seq, 2048]

        Returns:
            [B, action_seq, 1024]
        """
        for block in self.blocks:
            hidden_states = block(hidden_states, timestep_emb, backbone_features)

        # SiLU-gated output projection
        # proj_out_1: 1536 -> 3072
        h = ttnn.linear(
            hidden_states, self.proj_out_1_weight, bias=self.proj_out_1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH,
        )

        # Split 3072 -> 2x1536, SiLU gate
        half = self.proj_out_1_dim // 2
        gate = ttnn.slice(h, [0, 0, 0], [h.shape[0], h.shape[1], half])
        value = ttnn.slice(h, [0, 0, half], [h.shape[0], h.shape[1], self.proj_out_1_dim])
        ttnn.deallocate(h)

        gate = ttnn.silu(gate)
        gated = ttnn.mul(gate, value)
        ttnn.deallocate(gate)
        ttnn.deallocate(value)

        # proj_out_2: 1536 -> 1024
        output = ttnn.linear(
            gated, self.proj_out_2_weight, bias=self.proj_out_2_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(gated)

        return output
