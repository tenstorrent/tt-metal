# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6 Gated Attention block — on-device forward (single device).

Implements HF Qwen3NextAttention semantics entirely on device:
  - q_proj output split into [query, gate] via slice (was: host chunk)
  - q_norm/k_norm zero-centered RMSNorm per head_dim (was: host RMS)
  - Partial RoPE applied with on-device slice/neg/concat/mul/add (was: host rotate)
  - SDPA via ttnn.transformer.scaled_dot_product_attention with causal mask
  - sigmoid-gate on attn output (on device)
  - o_proj
"""
from __future__ import annotations

import ttnn

TILE = 32


def _t2t(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=None):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=mem or ttnn.L1_MEMORY_CONFIG,
    )


def _make_qknorm_weight(weight_torch, device):
    """Zero-centered (1+w) per-head_dim norm weight. Tile-aligned [1,1,dim/32,32]."""
    w = 1.0 + weight_torch.float()
    dim = w.numel()
    assert dim % TILE == 0
    return ttnn.from_torch(
        w.reshape(1, 1, dim // TILE, TILE), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


class TtGatedAttentionBlock:
    def __init__(self, device, state_dict, prefix, hf_config):
        self.device = device
        self.prefix = prefix
        self.hf_config = hf_config
        self.n_q = hf_config.num_attention_heads
        self.n_kv = hf_config.num_key_value_heads
        self.head_dim = hf_config.head_dim
        self.H = hf_config.hidden_size
        self.eps = hf_config.rms_norm_eps
        self.rotary_dim = int(hf_config.head_dim * hf_config.partial_rotary_factor)
        self.gqa_ratio = self.n_q // self.n_kv

        def load(name):
            t = state_dict[f"{prefix}.{name}"].float().T
            return _t2t(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG)

        self.w_q = load("q_proj.weight")
        self.w_k = load("k_proj.weight")
        self.w_v = load("v_proj.weight")
        self.w_o = load("o_proj.weight")

        # QK-norm weights on device (zero-centered)
        self.q_norm_w = _make_qknorm_weight(state_dict[f"{prefix}.q_norm.weight"], device)
        self.k_norm_w = _make_qknorm_weight(state_dict[f"{prefix}.k_norm.weight"], device)

    def _apply_partial_rope_device(self, x_tt, cos_tt, sin_tt):
        """On-device partial RoPE: rotate first rotary_dim dims, pass through the rest.

        x_tt: [B, n_heads, T, head_dim]
        cos_tt, sin_tt: [1, 1, T, rotary_dim]
        Returns: same shape as x_tt.
        """
        B, n, T, hd = x_tt.shape
        rd = self.rotary_dim
        # Slice rotated part and pass-through part
        x_rot = ttnn.slice(x_tt, [0, 0, 0, 0], [B, n, T, rd], memory_config=ttnn.L1_MEMORY_CONFIG)
        x_pass = ttnn.slice(x_tt, [0, 0, 0, rd], [B, n, T, hd], memory_config=ttnn.L1_MEMORY_CONFIG)

        # rotate_half: split rd into two halves along last dim
        half = rd // 2
        x1 = ttnn.slice(x_rot, [0, 0, 0, 0], [B, n, T, half], memory_config=ttnn.L1_MEMORY_CONFIG)
        x2 = ttnn.slice(x_rot, [0, 0, 0, half], [B, n, T, rd], memory_config=ttnn.L1_MEMORY_CONFIG)
        neg_x2 = ttnn.neg(x2, memory_config=ttnn.L1_MEMORY_CONFIG)
        rotated = ttnn.concat([neg_x2, x1], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        # x_rot * cos + rotated * sin
        x_cos = ttnn.multiply(x_rot, cos_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
        x_sin = ttnn.multiply(rotated, sin_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
        x_embed = ttnn.add(x_cos, x_sin, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Concat with pass-through
        return ttnn.concat([x_embed, x_pass], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

    def __call__(self, hidden_states, cos, sin, attention_mask):
        """
        hidden_states: ttnn [B, T, H] BF16
        cos, sin: torch [B, T, rotary_dim]   ← will move to device here
        attention_mask: torch [1, 1, T, T] additive ← will move to device here
        """
        B, T = hidden_states.shape[0], hidden_states.shape[1]

        # 1. Projections
        q_gate_tt = ttnn.linear(hidden_states, self.w_q, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        k_tt = ttnn.linear(hidden_states, self.w_k, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        v_tt = ttnn.linear(hidden_states, self.w_v, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        # 2. Split Q and gate from q_gate (shape [B, T, n_q * head_dim * 2]).
        # Layout: per-head [Q_h0(head_dim) ‖ gate_h0(head_dim) ‖ Q_h1 ‖ gate_h1 ‖ ...]
        # We reshape to [B, T, n_q, 2*head_dim] then slice along last dim.
        q_gate_r = ttnn.reshape(q_gate_tt, [B, T, self.n_q, 2 * self.head_dim])
        q_r = ttnn.slice(q_gate_r, [0, 0, 0, 0], [B, T, self.n_q, self.head_dim], memory_config=ttnn.L1_MEMORY_CONFIG)
        gate_r = ttnn.slice(
            q_gate_r, [0, 0, 0, self.head_dim], [B, T, self.n_q, 2 * self.head_dim], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        gate_flat = ttnn.reshape(gate_r, [B, T, self.n_q * self.head_dim])

        # 3. Reshape K, V to per-head
        k_r = ttnn.reshape(k_tt, [B, T, self.n_kv, self.head_dim])
        v_r = ttnn.reshape(v_tt, [B, T, self.n_kv, self.head_dim])

        # 4. QK-norm on device (zero-centered RMSNorm per head_dim)
        # rms_norm normalizes over last dim and applies weight; our weight is already (1+w).
        q_normed = ttnn.rms_norm(q_r, weight=self.q_norm_w, epsilon=self.eps, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_normed = ttnn.rms_norm(k_r, weight=self.k_norm_w, epsilon=self.eps, memory_config=ttnn.L1_MEMORY_CONFIG)

        # 5. Transpose to [B, n_heads, T, head_dim] for attention
        q_t = ttnn.transpose(q_normed, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        k_t = ttnn.transpose(k_normed, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        v_t = ttnn.transpose(v_r, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)

        # 6. Partial RoPE on device — cos/sin onto device first
        cos_tt = _t2t(cos.unsqueeze(1).float(), self.device, dtype=ttnn.bfloat16)
        sin_tt = _t2t(sin.unsqueeze(1).float(), self.device, dtype=ttnn.bfloat16)
        q_t = self._apply_partial_rope_device(q_t, cos_tt, sin_tt)
        k_t = self._apply_partial_rope_device(k_t, cos_tt, sin_tt)

        # 7. GQA: repeat_interleave K, V along head dim (n_kv → n_q)
        # ttnn.repeat doesn't directly do repeat_interleave; use reshape+broadcast trick
        # k_t: [B, n_kv, T, head_dim] → unsqueeze head dim → broadcast → reshape
        if self.gqa_ratio > 1:
            # [B, n_kv, T, hd] → [B, n_kv, 1, T, hd] → repeat to [B, n_kv, gqa_ratio, T, hd] → reshape
            k_t = ttnn.unsqueeze(k_t, 2)
            k_t = ttnn.repeat(k_t, ttnn.Shape([1, 1, self.gqa_ratio, 1, 1]))
            k_t = ttnn.reshape(k_t, [B, self.n_q, T, self.head_dim])
            v_t = ttnn.unsqueeze(v_t, 2)
            v_t = ttnn.repeat(v_t, ttnn.Shape([1, 1, self.gqa_ratio, 1, 1]))
            v_t = ttnn.reshape(v_t, [B, self.n_q, T, self.head_dim])

        # 8. SDPA via ttnn.transformer.scaled_dot_product_attention
        # Build mask on device
        mask_tt = _t2t(attention_mask.float(), self.device, dtype=ttnn.bfloat16)
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            is_causal=False,
            attn_mask=mask_tt,
            scale=self.head_dim**-0.5,
        )

        # 9. Transpose back to [B, T, n_q*head_dim] and gate
        attn_out = ttnn.transpose(attn_out, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn_flat = ttnn.reshape(attn_out, [B, T, self.n_q * self.head_dim])
        gate_sig = ttnn.sigmoid(gate_flat, memory_config=ttnn.L1_MEMORY_CONFIG)
        gated = ttnn.multiply(attn_flat, gate_sig, memory_config=ttnn.L1_MEMORY_CONFIG)

        # 10. o_proj
        out = ttnn.linear(gated, self.w_o, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        return out
