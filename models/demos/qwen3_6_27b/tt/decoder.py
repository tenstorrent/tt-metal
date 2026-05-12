# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hybrid decoder layer for Qwen3.6: dispatches DeltaNet or Gated-Attention
based on layer_types[i], plus pre-attention RMSNorm + residual + post-attention
RMSNorm + MLP + residual.
"""
from __future__ import annotations

import ttnn
from models.demos.qwen3_6_27b.tt.attention_v2 import TtGatedAttentionBlock
from models.demos.qwen3_6_27b.tt.linear_attention import TtDeltaNetBlock, _t2t


def _make_norm_weight(weight_torch, device, zero_centered=True):
    """Build the device-side norm weight tensor. For zero-centered Qwen3NextRMSNorm,
    pre-add 1 so ttnn.rms_norm's standard `x*rsqrt*gamma` matches `x*rsqrt*(1+w)`.
    ttnn.rms_norm requires gamma shape [1, 1, dim/32, 32] (tile-aligned)."""
    TILE = 32
    w = weight_torch.float()
    if zero_centered:
        w = 1.0 + w
    dim = w.numel()
    assert dim % TILE == 0, f"norm dim {dim} not divisible by tile size {TILE}"
    w = w.reshape(1, 1, dim // TILE, TILE)
    return ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _rms_norm_device(x_tt, weight_tt, eps):
    """On-device RMSNorm. weight_tt already incorporates the zero-centered (1+w) adjustment."""
    return ttnn.rms_norm(x_tt, weight=weight_tt, epsilon=eps, memory_config=ttnn.L1_MEMORY_CONFIG)


def _mlp_device(hidden_tt, gate_w, up_w, down_w):
    """SwiGLU MLP entirely on device: down(silu(gate(x)) * up(x))."""
    gate_out = ttnn.linear(hidden_tt, gate_w, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    up_out = ttnn.linear(hidden_tt, up_w, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    silu_gate = ttnn.silu(gate_out, memory_config=ttnn.L1_MEMORY_CONFIG)
    intermediate = ttnn.multiply(silu_gate, up_out, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.linear(intermediate, down_w, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    return out


class TtDecoderLayer:
    """One Qwen3.6 decoder layer: norm → attn (or DeltaNet) → +res → norm → MLP → +res."""

    def __init__(self, device, state_dict, layer_idx, layer_type, hf_config):
        self.device = device
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.hf_cfg = hf_config
        self.eps = hf_config.rms_norm_eps

        base = f"model.language_model.layers.{layer_idx}"

        # Layer norms — on device, with (1+w) pre-adjustment for zero-centered Qwen3NextRMSNorm
        self.input_ln_w = _make_norm_weight(state_dict[f"{base}.input_layernorm.weight"], device, zero_centered=True)
        self.post_attn_ln_w = _make_norm_weight(
            state_dict[f"{base}.post_attention_layernorm.weight"], device, zero_centered=True
        )

        # MLP weights — push to device DRAM as transposed [in, out] for ttnn.linear
        def load_proj(name):
            t = state_dict[f"{base}.mlp.{name}"].float()
            return _t2t(t.T, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG)

        self.mlp_gate_w = load_proj("gate_proj.weight")
        self.mlp_up_w = load_proj("up_proj.weight")
        self.mlp_down_w = load_proj("down_proj.weight")

        # Attention block
        if layer_type == "linear_attention":
            self.attn = TtDeltaNetBlock(device, state_dict, f"{base}.linear_attn", hf_config)
        else:
            self.attn = TtGatedAttentionBlock(device, state_dict, f"{base}.self_attn", hf_config)

    def __call__(self, hidden_tt, cos=None, sin=None, attention_mask=None):
        # Residual 1: input_layernorm + attn
        residual = hidden_tt
        x = _rms_norm_device(hidden_tt, self.input_ln_w, self.eps)

        if self.layer_type == "linear_attention":
            attn_out = self.attn(x)
        else:
            attn_out = self.attn(x, cos, sin, attention_mask)

        x = ttnn.add(residual, attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Residual 2: post_attention_layernorm + MLP
        residual = x
        x = _rms_norm_device(x, self.post_attn_ln_w, self.eps)
        mlp_out = _mlp_device(x, self.mlp_gate_w, self.mlp_up_w, self.mlp_down_w)
        x = ttnn.add(residual, mlp_out, memory_config=ttnn.L1_MEMORY_CONFIG)

        return x
