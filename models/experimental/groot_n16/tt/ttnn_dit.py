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
import torch.nn.functional

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
        """Apply AdaLN: (1 + scale) * LN(x) + shift.

        Upstream: temb = self.linear(self.silu(temb))
        temb is [B, inner_dim] (2D), we handle [B, 1, inner_dim] (3D) too.
        """
        # Apply SiLU THEN linear (matches upstream AdaLayerNorm.forward)
        temb_activated = ttnn.silu(timestep_emb)
        scale_shift = ttnn.linear(
            temb_activated, self.proj_weight, bias=self.proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(temb_activated)

        scale = ttnn.slice(scale_shift, [0, 0, 0], [scale_shift.shape[0], 1, self.hidden_size])
        shift = ttnn.slice(scale_shift, [0, 0, self.hidden_size],
                           [scale_shift.shape[0], 1, 2 * self.hidden_size])
        ttnn.deallocate(scale_shift)

        # LayerNorm (no learned weight/bias, eps=1e-5 matching upstream)
        normed = ttnn.layer_norm(
            hidden_states, epsilon=1e-5, memory_config=ttnn.L1_MEMORY_CONFIG,
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

        head_dim=48 is not tile-aligned, so we pad to 64, use SDPA, then unpad.
        Q/K/V projections and head reshape done with CPU assist for correctness.
        """
        batch_size = hidden_states.shape[0]
        q_seq = hidden_states.shape[1]
        padded_head_dim = ((self.head_dim + 31) // 32) * 32  # 48 -> 64

        # Q projection: [B, q_seq, inner_dim] -> [B, q_seq, inner_dim]
        q = ttnn.linear(hidden_states, self.to_q_weight, bias=self.to_q_bias,
                        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)

        # K/V source
        kv_source = encoder_hidden_states if (self.is_cross_attention and encoder_hidden_states is not None) else hidden_states
        kv_seq = kv_source.shape[1]

        k = ttnn.linear(kv_source, self.to_k_weight, bias=self.to_k_bias,
                        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        v = ttnn.linear(kv_source, self.to_v_weight, bias=self.to_v_bias,
                        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)

        # Reshape + pad on CPU for SDPA compatibility
        def to_heads_padded(tensor, seq_len):
            t = ttnn.to_torch(tensor).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            t = torch.nn.functional.pad(t, (0, padded_head_dim - self.head_dim))
            return ttnn.from_torch(
                t.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=hidden_states.device(), memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        q_heads = to_heads_padded(q, q_seq)
        k_heads = to_heads_padded(k, kv_seq)
        v_heads = to_heads_padded(v, kv_seq)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # SDPA
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_heads, k_heads, v_heads, is_causal=False,
        )
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # Unpad + reshape back: [B, heads, q_seq, padded_hd] -> [B, q_seq, inner_dim]
        out_cpu = ttnn.to_torch(attn_output)[:, :, :, :self.head_dim]
        ttnn.deallocate(attn_output)
        out_cpu = out_cpu.permute(0, 2, 1, 3).contiguous().reshape(batch_size, q_seq, self.inner_dim)

        context = ttnn.from_torch(
            out_cpu.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=hidden_states.device(), memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Output projection
        output = ttnn.linear(context, self.to_out_0_weight, bias=self.to_out_0_bias,
                             memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        ttnn.deallocate(context)
        return output


class DiTFFNTTNN:
    """FFN: up projection with GELU -> down projection.

    net.0.proj: Linear(1536, 6144) — up projection
    GELU activation
    net.2: Linear(6144, 1536) — down projection

    No GEGLU split — the ff weight shapes confirm this:
    net.0.proj.weight [6144, 1536] and net.2.weight [1536, 6144]
    """

    def __init__(self, weights: Dict[str, torch.Tensor], prefix: str, device: Any):
        self.device = device

        w_up = weights.get(f"{prefix}net.0.proj.weight")
        b_up = weights.get(f"{prefix}net.0.proj.bias")
        w_down = weights.get(f"{prefix}net.2.weight")
        b_down = weights.get(f"{prefix}net.2.bias")

        if w_up is not None:
            self.up_weight = preprocess_linear_weight(w_up, device)
            self.up_bias = preprocess_linear_bias(b_up, device) if b_up is not None else None
            self.down_weight = preprocess_linear_weight(w_down, device)
            self.down_bias = preprocess_linear_bias(b_down, device) if b_down is not None else None

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        # Up projection with GELU: 1536 -> 6144
        h = ttnn.linear(hidden_states, self.up_weight, bias=self.up_bias,
                        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                        core_grid=CORE_GRID_BH, activation="gelu")

        # Down projection: 6144 -> 1536
        output = ttnn.linear(h, self.down_weight, bias=self.down_bias,
                             memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                             core_grid=CORE_GRID_BH)
        ttnn.deallocate(h)
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

        # AdaLN for self-attention (norm1)
        self.adaln = AdaLayerNormTTNN(weights, f"{prefix}norm1.", inner_dim, device)

        # Self-attention (attn1)
        self.attn = DiTAttentionTTNN(
            weights, f"{prefix}attn1.",
            config.num_attention_heads, config.attention_head_dim, device,
        )

        # norm3: parameter-free LayerNorm for FFN (elementwise_affine=False in upstream)

        # FFN
        self.ffn = DiTFFNTTNN(weights, f"{prefix}ff.", device)

    def __call__(self, hidden_states: ttnn.Tensor, timestep_emb: ttnn.Tensor,
                 backbone_features: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        # AdaLN -> Attention -> Residual
        normed = self.adaln(hidden_states, timestep_emb)
        attn_out = self.attn(normed, encoder_hidden_states=backbone_features)
        hidden_states = ttnn.add(hidden_states, attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        # norm3 (parameter-free LayerNorm) -> FFN -> Residual
        normed_ff = ttnn.layer_norm(
            hidden_states, epsilon=1e-5, memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ffn_out = self.ffn(normed_ff)
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

        # Output processing (AdaLN-style, NOT a gated FFN):
        # conditioning = temb
        # shift, scale = proj_out_1(silu(conditioning)).chunk(2, dim=1)
        # hidden_states = norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        # output = proj_out_2(hidden_states)

        # proj_out_1 projects temb (NOT hidden_states) to shift+scale
        conditioning = ttnn.silu(timestep_emb)
        scale_shift = ttnn.linear(
            conditioning, self.proj_out_1_weight, bias=self.proj_out_1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(conditioning)

        half = self.proj_out_1_dim // 2
        shift = ttnn.slice(scale_shift, [0, 0, 0], [scale_shift.shape[0], 1, half])
        scale = ttnn.slice(scale_shift, [0, 0, half], [scale_shift.shape[0], 1, self.proj_out_1_dim])
        ttnn.deallocate(scale_shift)

        # norm_out: parameter-free LayerNorm
        normed = ttnn.layer_norm(
            hidden_states, epsilon=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # AdaLN: (1 + scale) * normed + shift
        ones = ttnn.ones_like(scale)
        scale_p1 = ttnn.add(ones, scale)
        ttnn.deallocate(ones)
        ttnn.deallocate(scale)

        hidden_states = ttnn.mul(normed, scale_p1)
        ttnn.deallocate(normed)
        ttnn.deallocate(scale_p1)

        hidden_states = ttnn.add(hidden_states, shift)
        ttnn.deallocate(shift)

        # proj_out_2: 1536 -> 1024
        output = ttnn.linear(
            hidden_states, self.proj_out_2_weight, bias=self.proj_out_2_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH,
        )

        return output
