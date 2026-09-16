# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ttnn

from models.experimental.audiox.tt.common import linear_weight, to_tt
from models.experimental.audiox.tt.rotary import apply_rotary_pos_emb


def _deallocate_tensor(tensor) -> None:
    if tensor is None:
        return
    ttnn.deallocate(tensor, force=True)


def _self_attention(x, qkv_w, ow, num_heads, cos=None, sin=None):
    """Self-attention with fused QKV projection. Optionally applies rotary to Q/K."""
    qkv = ttnn.linear(x, qkv_w)
    q, k, v = ttnn.chunk(qkv, 3, dim=-1)

    batch, seq, dim = q.shape
    head_dim = dim // num_heads

    q = ttnn.transpose(ttnn.reshape(q, (batch, seq, num_heads, head_dim)), 1, 2)
    k = ttnn.transpose(ttnn.reshape(k, (batch, seq, num_heads, head_dim)), 1, 2)
    v = ttnn.transpose(ttnn.reshape(v, (batch, seq, num_heads, head_dim)), 1, 2)

    if cos is not None and sin is not None:
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = ttnn.reshape(ttnn.transpose(out, 1, 2), (batch, seq, dim))
    return ttnn.linear(out, ow)


def _cross_attention(x, context, qw, kvw, ow, num_heads, kv_heads):
    """Cross-attention with separate Q and fused KV projections. When
    kv_heads < num_heads (dim_context < dim), repeat-interleaves K/V so
    SDPA sees matching head counts, mirroring the AudioX flash path."""
    q = ttnn.linear(x, qw)
    kv = ttnn.linear(context, kvw)
    k, v = ttnn.chunk(kv, 2, dim=-1)

    batch, sq, dim = q.shape
    sk = k.shape[1]
    head_dim = dim // num_heads

    q = ttnn.transpose(ttnn.reshape(q, (batch, sq, num_heads, head_dim)), 1, 2)
    k = ttnn.transpose(ttnn.reshape(k, (batch, sk, kv_heads, head_dim)), 1, 2)
    v = ttnn.transpose(ttnn.reshape(v, (batch, sk, kv_heads, head_dim)), 1, 2)

    if kv_heads != num_heads:
        repeats = num_heads // kv_heads
        k = ttnn.repeat_interleave(k, repeats, dim=1)
        v = ttnn.repeat_interleave(v, repeats, dim=1)

    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = ttnn.reshape(ttnn.transpose(out, 1, 2), (batch, sq, dim))
    return ttnn.linear(out, ow)


def _feedforward(x, glu_w, glu_b, out_w, out_b):
    """SwiGLU feedforward: Linear -> chunk(x, gate) -> x * SiLU(gate) -> Linear."""
    h = ttnn.linear(x, glu_w, bias=glu_b)
    x_part, gate = ttnn.chunk(h, 2, dim=-1)
    h = ttnn.multiply(x_part, ttnn.silu(gate))
    return ttnn.linear(h, out_w, bias=out_b)


class TtTransformerBlock:
    """TTNN port of the AudioX continuous-transformer block: pre-norm
    self-attention with rotary, optional cross-attention, SwiGLU FFN.
    Mirrors the prepend-conditioning path used by AudioX DiT (no adaLN, no
    conformer, no qk_norm)."""

    def __init__(
        self,
        mesh_device,
        state_dict: dict,
        dim_heads: int = 64,
        cross_attend: bool = True,
    ):
        sd = state_dict
        self.mesh_device = mesh_device
        self.dim_heads = dim_heads
        self.cross_attend = cross_attend

        dim = sd["self_attn.to_qkv.weight"].shape[1]
        self.num_heads = dim // dim_heads

        self.pre_norm_w = to_tt(sd["pre_norm.gamma"], mesh_device)
        self.pre_norm_b = to_tt(sd["pre_norm.beta"], mesh_device)

        self.self_qkv_w = to_tt(linear_weight(sd["self_attn.to_qkv.weight"]), mesh_device)
        self.self_o_w = to_tt(linear_weight(sd["self_attn.to_out.weight"]), mesh_device)

        if cross_attend:
            self.cross_norm_w = to_tt(sd["cross_attend_norm.gamma"], mesh_device)
            self.cross_norm_b = to_tt(sd["cross_attend_norm.beta"], mesh_device)
            self.cross_q_w = to_tt(linear_weight(sd["cross_attn.to_q.weight"]), mesh_device)
            self.cross_kv_w = to_tt(linear_weight(sd["cross_attn.to_kv.weight"]), mesh_device)
            self.cross_o_w = to_tt(linear_weight(sd["cross_attn.to_out.weight"]), mesh_device)
            self.cross_kv_heads = sd["cross_attn.to_kv.weight"].shape[1] // dim_heads

        self.ff_norm_w = to_tt(sd["ff_norm.gamma"], mesh_device)
        self.ff_norm_b = to_tt(sd["ff_norm.beta"], mesh_device)

        self.ff_glu_w = to_tt(linear_weight(sd["ff.ff.0.proj.weight"]), mesh_device)
        self.ff_glu_b = to_tt(sd["ff.ff.0.proj.bias"], mesh_device)
        self.ff_out_w = to_tt(linear_weight(sd["ff.ff.2.weight"]), mesh_device)
        self.ff_out_b = to_tt(sd["ff.ff.2.bias"], mesh_device)

    def deallocate(self) -> None:
        _deallocate_tensor(self.pre_norm_w)
        _deallocate_tensor(self.pre_norm_b)
        _deallocate_tensor(self.self_qkv_w)
        _deallocate_tensor(self.self_o_w)

        if self.cross_attend:
            _deallocate_tensor(self.cross_norm_w)
            _deallocate_tensor(self.cross_norm_b)
            _deallocate_tensor(self.cross_q_w)
            _deallocate_tensor(self.cross_kv_w)
            _deallocate_tensor(self.cross_o_w)

        _deallocate_tensor(self.ff_norm_w)
        _deallocate_tensor(self.ff_norm_b)
        _deallocate_tensor(self.ff_glu_w)
        _deallocate_tensor(self.ff_glu_b)
        _deallocate_tensor(self.ff_out_w)
        _deallocate_tensor(self.ff_out_b)

    def __call__(
        self,
        x: ttnn.Tensor,
        context: ttnn.Tensor = None,
        rotary_cos: ttnn.Tensor = None,
        rotary_sin: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        normed = ttnn.layer_norm(x, weight=self.pre_norm_w, bias=self.pre_norm_b)
        x = ttnn.add(
            x,
            _self_attention(normed, self.self_qkv_w, self.self_o_w, self.num_heads, rotary_cos, rotary_sin),
        )

        if context is not None and self.cross_attend:
            normed = ttnn.layer_norm(x, weight=self.cross_norm_w, bias=self.cross_norm_b)
            x = ttnn.add(
                x,
                _cross_attention(
                    normed,
                    context,
                    self.cross_q_w,
                    self.cross_kv_w,
                    self.cross_o_w,
                    self.num_heads,
                    self.cross_kv_heads,
                ),
            )

        normed = ttnn.layer_norm(x, weight=self.ff_norm_w, bias=self.ff_norm_b)
        x = ttnn.add(x, _feedforward(normed, self.ff_glu_w, self.ff_glu_b, self.ff_out_w, self.ff_out_b))
        return x
