# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
AlternateVLDiT (Diffusion Transformer) - TTNN Implementation for GR00T N1.6.

32-layer transformer with alternating cross-attention pattern:
    - Even layers (0,2,4,...): cross-attention to VLM backbone features
      * Alternates between image tokens and text tokens every attend_text_every_n_blocks=2
    - Odd layers (1,3,5,...): self-attention only

Uses AdaLN (Adaptive Layer Normalization) conditioned on timestep.

Architecture per block:
    AdaLN -> Self-Attention -> AdaLN -> Cross-Attention (even only) -> AdaLN -> FFN

Inner dim: 32 heads * 48 head_dim = 1536
Cross-attention dim: 2048 (backbone features)
Output dim: 1024
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

import ttnn
from models.experimental.groot_n16.common.configs import DiTConfig
from models.experimental.groot_n16.tt.ttnn_common import (
    CORE_GRID_BH,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_params,
    to_tt_tensor,
)


class AdaLayerNormTTNN:
    """
    Adaptive Layer Normalization conditioned on timestep embedding.

    AdaLN(x, t) = (1 + scale(t)) * LayerNorm(x) + shift(t)

    The scale and shift are produced by a linear projection of the
    timestep embedding (from the DiT's global conditioning).
    """

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        prefix: str,
        hidden_size: int,
        device: Any,
    ):
        self.device = device
        self.hidden_size = hidden_size

        # LayerNorm base parameters
        ln_w = weights.get(f"{prefix}norm.weight")
        ln_b = weights.get(f"{prefix}norm.bias")
        if ln_w is not None:
            self.norm_weight, self.norm_bias = preprocess_layernorm_params(
                ln_w, ln_b, device,
            )

        # Linear projection for scale and shift: timestep_dim -> 2 * hidden_size
        proj_w = weights.get(f"{prefix}linear.weight")
        proj_b = weights.get(f"{prefix}linear.bias")
        if proj_w is not None:
            self.proj_weight = preprocess_linear_weight(proj_w, device)
            self.proj_bias = preprocess_linear_bias(proj_b, device) if proj_b is not None else None

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        timestep_emb: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Apply adaptive layer norm.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            timestep_emb: [batch, 1, timestep_dim] timestep embedding

        Returns:
            [batch, seq_len, hidden_size] normalized output
        """
        # Get scale and shift from timestep
        scale_shift = ttnn.linear(
            timestep_emb, self.proj_weight, bias=self.proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )

        # Split into scale and shift (each hidden_size)
        # scale_shift: [batch, 1, 2*hidden_size] -> scale, shift each [batch, 1, hidden_size]
        scale = ttnn.slice(scale_shift, [0, 0, 0], [scale_shift.shape[0], 1, self.hidden_size])
        shift = ttnn.slice(scale_shift, [0, 0, self.hidden_size], [scale_shift.shape[0], 1, 2 * self.hidden_size])
        ttnn.deallocate(scale_shift)

        # LayerNorm
        normed = ttnn.layer_norm(
            hidden_states,
            weight=self.norm_weight,
            bias=self.norm_bias,
            epsilon=1e-6,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Apply: (1 + scale) * normed + shift
        ones = ttnn.ones_like(scale)
        scale_plus_one = ttnn.add(ones, scale)
        ttnn.deallocate(ones)
        ttnn.deallocate(scale)

        output = ttnn.mul(normed, scale_plus_one)
        ttnn.deallocate(normed)
        ttnn.deallocate(scale_plus_one)

        output = ttnn.add(output, shift)
        ttnn.deallocate(shift)

        return output


class DiTSelfAttentionTTNN:
    """Self-attention block for DiT."""

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        prefix: str,
        num_heads: int,
        head_dim: int,
        device: Any,
    ):
        self.device = device
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        # Q, K, V projections (separate in DiT, not fused)
        for proj_name in ["to_q", "to_k", "to_v", "to_out.0"]:
            w = weights.get(f"{prefix}{proj_name}.weight")
            b = weights.get(f"{prefix}{proj_name}.bias")
            attr_w = f"{proj_name.replace('.', '_')}_weight"
            attr_b = f"{proj_name.replace('.', '_')}_bias"
            if w is not None:
                setattr(self, attr_w, preprocess_linear_weight(w, device))
                setattr(self, attr_b, preprocess_linear_bias(b, device) if b is not None else None)

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Self-attention forward.

        Args:
            hidden_states: [batch, seq_len, inner_dim]

        Returns:
            [batch, seq_len, inner_dim]
        """
        scale = 1.0 / math.sqrt(self.head_dim)

        # Separate Q, K, V projections
        q = ttnn.linear(
            hidden_states, self.to_q_weight, bias=self.to_q_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        k = ttnn.linear(
            hidden_states, self.to_k_weight, bias=self.to_k_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        v = ttnn.linear(
            hidden_states, self.to_v_weight, bias=self.to_v_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )

        batch_size, seq_len, _ = hidden_states.shape

        # Reshape for multi-head: [B, seq, heads*head_dim] -> [B, heads, seq, head_dim]
        q = ttnn.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        q = ttnn.mul(q, scale)
        attn = ttnn.matmul(
            q, k,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        attn = ttnn.softmax_in_place(attn, numeric_stable=True)

        context = ttnn.matmul(
            attn, v,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(v)

        # Reshape back: [B, heads, seq, head_dim] -> [B, seq, inner_dim]
        context = ttnn.permute(context, (0, 2, 1, 3))
        context = ttnn.reshape(context, (batch_size, seq_len, self.inner_dim))

        # Output projection
        output = ttnn.linear(
            context, self.to_out_0_weight, bias=self.to_out_0_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(context)

        return output


class DiTCrossAttentionTTNN:
    """Cross-attention block for DiT (attends to VLM backbone features)."""

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        prefix: str,
        num_heads: int,
        head_dim: int,
        cross_attention_dim: int,
        device: Any,
    ):
        self.device = device
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.cross_attention_dim = cross_attention_dim

        # Q from action tokens, K/V from backbone features
        for proj_name in ["to_q", "to_k", "to_v", "to_out.0"]:
            w = weights.get(f"{prefix}{proj_name}.weight")
            b = weights.get(f"{prefix}{proj_name}.bias")
            attr_w = f"{proj_name.replace('.', '_')}_weight"
            attr_b = f"{proj_name.replace('.', '_')}_bias"
            if w is not None:
                setattr(self, attr_w, preprocess_linear_weight(w, device))
                setattr(self, attr_b, preprocess_linear_bias(b, device) if b is not None else None)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Cross-attention: query from action tokens, key/value from backbone.

        Args:
            hidden_states: [batch, action_seq, inner_dim] action tokens
            encoder_hidden_states: [batch, vl_seq, cross_attention_dim] backbone features

        Returns:
            [batch, action_seq, inner_dim]
        """
        scale = 1.0 / math.sqrt(self.head_dim)

        q = ttnn.linear(
            hidden_states, self.to_q_weight, bias=self.to_q_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        k = ttnn.linear(
            encoder_hidden_states, self.to_k_weight, bias=self.to_k_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        v = ttnn.linear(
            encoder_hidden_states, self.to_v_weight, bias=self.to_v_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )

        batch_size = hidden_states.shape[0]
        q_seq = hidden_states.shape[1]
        kv_seq = encoder_hidden_states.shape[1]

        # Reshape for multi-head
        q = ttnn.reshape(q, (batch_size, q_seq, self.num_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (batch_size, kv_seq, self.num_heads, self.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (batch_size, kv_seq, self.num_heads, self.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))

        q = ttnn.mul(q, scale)
        attn = ttnn.matmul(
            q, k,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        attn = ttnn.softmax_in_place(attn, numeric_stable=True)

        context = ttnn.matmul(
            attn, v,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(v)

        context = ttnn.permute(context, (0, 2, 1, 3))
        context = ttnn.reshape(context, (batch_size, q_seq, self.inner_dim))

        output = ttnn.linear(
            context, self.to_out_0_weight, bias=self.to_out_0_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(context)

        return output


class DiTFFNTTNN:
    """Feed-forward network for DiT blocks: inner_dim -> 4*inner_dim -> inner_dim with GELU."""

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        prefix: str,
        device: Any,
    ):
        self.device = device

        # net.0.proj = GEGLU gate projection (inner_dim -> 2 * 4*inner_dim for gate+value)
        # net.2 = down projection (4*inner_dim -> inner_dim)
        # GEGLU: split into gate and value, apply GELU to gate, multiply
        w_gate = weights.get(f"{prefix}net.0.proj.weight")
        b_gate = weights.get(f"{prefix}net.0.proj.bias")
        w_down = weights.get(f"{prefix}net.2.weight")
        b_down = weights.get(f"{prefix}net.2.bias")

        if w_gate is not None:
            self.gate_weight = preprocess_linear_weight(w_gate, device)
            self.gate_bias = preprocess_linear_bias(b_gate, device) if b_gate is not None else None
            self.down_weight = preprocess_linear_weight(w_down, device)
            self.down_bias = preprocess_linear_bias(b_down, device) if b_down is not None else None
            self.gate_out_dim = w_gate.shape[0]  # 2 * ff_dim for GEGLU split

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """FFN with GEGLU activation."""
        # GEGLU: project to 2x, split, gate with GELU
        gate_proj = ttnn.linear(
            hidden_states, self.gate_weight, bias=self.gate_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )

        half_dim = self.gate_out_dim // 2
        # Split into gate and value
        gate = ttnn.slice(gate_proj, [0, 0, 0], [gate_proj.shape[0], gate_proj.shape[1], half_dim])
        value = ttnn.slice(gate_proj, [0, 0, half_dim], [gate_proj.shape[0], gate_proj.shape[1], self.gate_out_dim])
        ttnn.deallocate(gate_proj)

        gate = ttnn.gelu(gate)
        intermediate = ttnn.mul(gate, value)
        ttnn.deallocate(gate)
        ttnn.deallocate(value)

        # Down projection
        output = ttnn.linear(
            intermediate, self.down_weight, bias=self.down_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(intermediate)

        return output


class DiTBlockTTNN:
    """
    Single AlternateVLDiT transformer block.

    Even blocks: AdaLN -> SelfAttn -> AdaLN -> CrossAttn (to VLM features) -> AdaLN -> FFN
    Odd blocks: AdaLN -> SelfAttn -> AdaLN -> FFN (no cross-attention)
    """

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        prefix: str,
        config: DiTConfig,
        block_idx: int,
        device: Any,
    ):
        self.device = device
        self.config = config
        self.block_idx = block_idx
        self.has_cross_attention = (block_idx % 2 == 0)  # even blocks
        inner_dim = config.inner_dim

        # AdaLN for self-attention
        self.adaln1 = AdaLayerNormTTNN(weights, f"{prefix}norm1.", inner_dim, device)

        # Self-attention
        self.self_attn = DiTSelfAttentionTTNN(
            weights, f"{prefix}attn1.",
            config.num_attention_heads, config.attention_head_dim, device,
        )

        if self.has_cross_attention:
            # AdaLN for cross-attention
            self.adaln2 = AdaLayerNormTTNN(weights, f"{prefix}norm2.", inner_dim, device)

            # Cross-attention
            self.cross_attn = DiTCrossAttentionTTNN(
                weights, f"{prefix}attn2.",
                config.num_attention_heads, config.attention_head_dim,
                config.cross_attention_dim, device,
            )

        # AdaLN for FFN
        self.adaln_ff = AdaLayerNormTTNN(weights, f"{prefix}norm3.", inner_dim, device)

        # FFN
        self.ffn = DiTFFNTTNN(weights, f"{prefix}ff.", device)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        timestep_emb: ttnn.Tensor,
        encoder_hidden_states: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Forward pass for a single DiT block.

        Args:
            hidden_states: [batch, seq, inner_dim] action tokens
            timestep_emb: [batch, 1, timestep_dim] conditioning
            encoder_hidden_states: [batch, vl_seq, cross_dim] VLM features (for even blocks)

        Returns:
            [batch, seq, inner_dim]
        """
        # Self-attention with AdaLN
        normed = self.adaln1(hidden_states, timestep_emb)
        attn_out = self.self_attn(normed)
        hidden_states = ttnn.add(hidden_states, attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        # Cross-attention (even blocks only)
        if self.has_cross_attention and encoder_hidden_states is not None:
            normed = self.adaln2(hidden_states, timestep_emb)
            cross_out = self.cross_attn(normed, encoder_hidden_states)
            hidden_states = ttnn.add(hidden_states, cross_out, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(cross_out)

        # FFN with AdaLN
        normed = self.adaln_ff(hidden_states, timestep_emb)
        ffn_out = self.ffn(normed)
        hidden_states = ttnn.add(hidden_states, ffn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ffn_out)

        return hidden_states


class AlternateVLDiTTTNN:
    """
    Complete AlternateVLDiT action head for GR00T N1.6.

    32 layers with alternating cross-attention pattern.
    Even layers cross-attend to VLM features (alternating image/text tokens).
    Odd layers do self-attention only.
    """

    def __init__(
        self,
        config: DiTConfig,
        weights: Dict[str, torch.Tensor],
        device: Any,
    ):
        self.config = config
        self.device = device

        # Build all DiT blocks
        self.blocks = []
        for i in range(config.num_layers):
            prefix = f"transformer_blocks.{i}."
            block = DiTBlockTTNN(weights, prefix, config, i, device)
            self.blocks.append(block)

        # Final output projection: inner_dim -> output_dim
        out_w = weights.get("proj_out.weight")
        out_b = weights.get("proj_out.bias")
        if out_w is not None:
            self.proj_out_weight = preprocess_linear_weight(out_w, device)
            self.proj_out_bias = preprocess_linear_bias(out_b, device) if out_b is not None else None

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        timestep_emb: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        image_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Forward pass through all 32 DiT blocks.

        Args:
            hidden_states: [batch, action_seq, inner_dim] action token embeddings
            timestep_emb: [batch, 1, timestep_dim]
            encoder_hidden_states: [batch, vl_seq, backbone_dim] backbone features
            image_mask: Optional mask indicating which backbone tokens are image tokens

        Returns:
            [batch, action_seq, output_dim] predicted velocity field
        """
        for i, block in enumerate(self.blocks):
            # For even blocks: select image or text tokens based on alternation
            cross_features = None
            if i % 2 == 0:  # cross-attention block
                # Alternate between image and text every attend_text_every_n_blocks
                # Block index within cross-attn blocks: i // 2
                cross_block_idx = i // 2
                use_text = (cross_block_idx % self.config.attend_text_every_n_blocks) != 0

                # For now, pass all backbone features
                # TODO: implement image/text token masking when image_mask is available
                cross_features = encoder_hidden_states

            hidden_states = block(hidden_states, timestep_emb, cross_features)

        # Final projection
        output = ttnn.linear(
            hidden_states, self.proj_out_weight, bias=self.proj_out_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )

        return output
