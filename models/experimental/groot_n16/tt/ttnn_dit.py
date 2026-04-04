# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
AlternateVLDiT (Diffusion Transformer) - TTNN Implementation for GR00T N1.6.

32-layer transformer with alternating VL attention pattern:
    - Even layers: self-attention over [action_tokens; selected_backbone_tokens]
      * Alternates which backbone tokens to include (image vs text) every 2 blocks
    - Odd layers: self-attention over action_tokens only

There are NO separate cross-attention weights. The "cross-attention" is achieved
by concatenating backbone features into the KV of self-attention. All blocks
share the same structure: attn1 (self-attn) + ff (GEGLU FFN) + norm1 (AdaLN).

Uses AdaLN (Adaptive Layer Normalization) conditioned on timestep.

Inner dim: 32 heads * 48 head_dim = 1536
Output dim: 1024 (via two-stage proj_out_1 -> proj_out_2)
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
    preprocess_layernorm_params,
)


class AdaLayerNormTTNN:
    """
    Adaptive Layer Normalization: (1 + scale(t)) * LN(x) + shift(t)

    Weight keys: norm.weight, norm.bias (optional), linear.weight, linear.bias
    The linear projects timestep embedding to 2*hidden_size (scale + shift).
    """

    def __init__(self, weights: Dict[str, torch.Tensor], prefix: str, hidden_size: int, device: Any):
        self.device = device
        self.hidden_size = hidden_size

        ln_w = weights.get(f"{prefix}norm.weight")
        ln_b = weights.get(f"{prefix}norm.bias")
        if ln_w is not None:
            self.norm_weight = ttnn.from_torch(
                ln_w.unsqueeze(0).to(torch.bfloat16), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=device,
            )
            self.norm_bias = ttnn.from_torch(
                ln_b.unsqueeze(0).to(torch.bfloat16), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=device,
            ) if ln_b is not None else None

        proj_w = weights.get(f"{prefix}linear.weight")
        proj_b = weights.get(f"{prefix}linear.bias")
        if proj_w is not None:
            self.proj_weight = preprocess_linear_weight(proj_w, device)
            self.proj_bias = preprocess_linear_bias(proj_b, device) if proj_b is not None else None

    def __call__(self, hidden_states: ttnn.Tensor, timestep_emb: ttnn.Tensor) -> ttnn.Tensor:
        """Apply AdaLN: (1 + scale) * LN(x) + shift."""
        # Project timestep -> scale, shift
        scale_shift = ttnn.linear(
            timestep_emb, self.proj_weight, bias=self.proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH,
        )

        scale = ttnn.slice(scale_shift, [0, 0, 0], [scale_shift.shape[0], 1, self.hidden_size])
        shift = ttnn.slice(scale_shift, [0, 0, self.hidden_size],
                           [scale_shift.shape[0], 1, 2 * self.hidden_size])
        ttnn.deallocate(scale_shift)

        # LayerNorm
        normed = ttnn.layer_norm(
            hidden_states, weight=self.norm_weight, bias=self.norm_bias,
            epsilon=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG,
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


class DiTSelfAttentionTTNN:
    """
    Self-attention for DiT blocks.

    Weight keys: to_q.{weight,bias}, to_k.{weight,bias}, to_v.{weight,bias}, to_out.0.{weight,bias}
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

    def __call__(self, hidden_states: ttnn.Tensor,
                 kv_states: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """
        Self-attention, optionally with separate KV source.

        Args:
            hidden_states: [B, q_seq, dim] - query source
            kv_states: [B, kv_seq, dim] - if provided, K/V come from here (for VL concat)
                       If None, K/V come from hidden_states (pure self-attention)
        """
        scale = 1.0 / math.sqrt(self.head_dim)
        kv_source = kv_states if kv_states is not None else hidden_states

        q = ttnn.linear(hidden_states, self.to_q_weight, bias=self.to_q_bias,
                        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH)
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
            self.gate_out_dim = w_gate.shape[0]

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

    All blocks have the same structure: AdaLN -> SelfAttn -> Residual -> FFN -> Residual
    The difference is what goes into the KV of self-attention:
        - Even blocks: KV = concat(action_tokens, selected_backbone_tokens)
        - Odd blocks: KV = action_tokens only

    Weight keys under prefix: norm1.*, attn1.*, ff.*
    """

    def __init__(self, weights: Dict[str, torch.Tensor], prefix: str,
                 config: DiTConfig, block_idx: int, device: Any):
        self.device = device
        self.block_idx = block_idx
        self.is_vl_block = (block_idx % 2 == 0)  # even blocks attend to VL features
        inner_dim = config.inner_dim

        self.adaln = AdaLayerNormTTNN(weights, f"{prefix}norm1.", inner_dim, device)
        self.self_attn = DiTSelfAttentionTTNN(
            weights, f"{prefix}attn1.",
            config.num_attention_heads, config.attention_head_dim, device,
        )
        self.ffn = DiTFFNTTNN(weights, f"{prefix}ff.", device)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        timestep_emb: ttnn.Tensor,
        backbone_features: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Forward pass.

        For VL blocks (even): concatenate backbone features into KV sequence.
        Q comes from action tokens only, KV comes from [action; backbone].
        Only action-token outputs are returned (first q_seq tokens of output).
        """
        # AdaLN
        normed = self.adaln(hidden_states, timestep_emb)

        if self.is_vl_block and backbone_features is not None:
            # Concatenate action tokens + backbone features for KV
            kv_input = ttnn.concat([normed, backbone_features], dim=1)
            attn_out = self.self_attn(normed, kv_states=kv_input)
            ttnn.deallocate(kv_input)
        else:
            attn_out = self.self_attn(normed)

        # Residual
        hidden_states = ttnn.add(hidden_states, attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        # FFN (no separate AdaLN for FFN based on weight structure)
        ffn_out = self.ffn(hidden_states)
        hidden_states = ttnn.add(hidden_states, ffn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ffn_out)

        return hidden_states


class AlternateVLDiTTTNN:
    """
    Complete AlternateVLDiT for GR00T N1.6.

    32 blocks + two-stage output projection (proj_out_1 -> proj_out_2).
    """

    def __init__(self, config: DiTConfig, weights: Dict[str, torch.Tensor], device: Any):
        self.config = config
        self.device = device

        self.blocks = []
        for i in range(config.num_layers):
            prefix = f"transformer_blocks.{i}."
            self.blocks.append(DiTBlockTTNN(weights, prefix, config, i, device))

        # Two-stage output projection: inner_dim -> output_dim
        for proj_name in ["proj_out_1", "proj_out_2"]:
            w = weights.get(f"{proj_name}.weight")
            b = weights.get(f"{proj_name}.bias")
            if w is not None:
                setattr(self, f"{proj_name}_weight", preprocess_linear_weight(w, device))
                setattr(self, f"{proj_name}_bias",
                        preprocess_linear_bias(b, device) if b is not None else None)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        timestep_emb: ttnn.Tensor,
        backbone_features: ttnn.Tensor,
        image_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Forward through all 32 DiT blocks.

        Args:
            hidden_states: [B, action_seq, inner_dim]
            timestep_emb: [B, 1, inner_dim]
            backbone_features: [B, vl_seq, backbone_dim] VLM features
            image_mask: Optional mask for selecting image vs text tokens

        Returns:
            [B, action_seq, output_dim]
        """
        for i, block in enumerate(self.blocks):
            # For even (VL) blocks: select which backbone tokens to attend to
            vl_features = None
            if i % 2 == 0:
                # Alternate image/text selection every attend_text_every_n_blocks
                # For now pass all backbone features; image_mask filtering can be added
                vl_features = backbone_features

            hidden_states = block(hidden_states, timestep_emb, vl_features)

        # Two-stage output projection
        output = ttnn.linear(
            hidden_states, self.proj_out_1_weight, bias=self.proj_out_1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH,
        )
        output = ttnn.silu(output)
        output = ttnn.linear(
            output, self.proj_out_2_weight, bias=self.proj_out_2_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH,
        )

        return output
