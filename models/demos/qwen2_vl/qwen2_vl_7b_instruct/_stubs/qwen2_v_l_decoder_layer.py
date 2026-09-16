# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `qwen2_v_l_decoder_layer` of Qwen/Qwen2-VL-7B-Instruct.

Reference: `language_model.layers.0`, a `Qwen2VLDecoderLayer`:

    h = input_layernorm(hidden_states)                   # RMSNorm
    attn_out, _ = self_attn(h, position_embeddings=(cos, sin), ...)  # GQA + mrope
    hidden_states = hidden_states + attn_out
    h2 = post_attention_layernorm(hidden_states)          # RMSNorm
    mlp_out = down_proj(silu(gate_proj(h2)) * up_proj(h2))
    hidden_states = hidden_states + mlp_out

`self_attn` uses Qwen2-VL's MULTIMODAL ("mrope") rotary embedding: `cos`/`sin`
arrive as `(3, B, S, head_dim)` (temporal/height/width sections) and are
first collapsed to `(B, S, head_dim)` by picking `mrope_section`-sized chunks
round-robin from the 3 leading slices -- a cheap host-side table construction
(same pattern `apply_multimodal_rotary_pos_emb` uses), not a stand-in for the
actual layer math. The rotation itself (`x*cos + rotate_half(x)*sin`), the
projections, GQA attention, and MLP all run as ttnn ops on device.
"""

from __future__ import annotations

import torch

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    layer = torch_module  # Qwen2VLDecoderLayer
    attn = layer.self_attn
    mlp = layer.mlp
    input_ln = layer.input_layernorm
    post_ln = layer.post_attention_layernorm

    nh = int(attn.num_heads)
    nkv = int(attn.num_key_value_heads)
    hd = int(attn.head_dim)
    groups = nh // nkv
    scaling = float(attn.scaling)
    mrope_section = list(attn.config.rope_parameters["mrope_section"]) * 2
    eps = float(input_ln.variance_epsilon)

    # HiFi4 + fp32 dest accumulation: bf16 weights alone across 28 stacked
    # layers drop per-token logit PCC below the e2e bar on low-confidence
    # positions; high-fidelity matmul accumulation recovers it.
    _kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def _w(t):
        return ttnn.from_torch(t.detach(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    input_ln_w = _w(input_ln.weight)
    post_ln_w = _w(post_ln.weight)

    q_w, q_b = _w(attn.q_proj.weight), _w(attn.q_proj.bias.reshape(1, -1))
    k_w, k_b = _w(attn.k_proj.weight), _w(attn.k_proj.bias.reshape(1, -1))
    v_w, v_b = _w(attn.v_proj.weight), _w(attn.v_proj.bias.reshape(1, -1))
    o_w = _w(attn.o_proj.weight)

    gate_w = _w(mlp.gate_proj.weight)
    up_w = _w(mlp.up_proj.weight)
    down_w = _w(mlp.down_proj.weight)

    def _linear(x, weight, bias=None):
        return ttnn.linear(x, weight, bias=bias, transpose_b=True, memory_config=_DRAM, compute_kernel_config=_kcfg)

    def _rotate_half(x):
        x1 = x[..., : hd // 2]
        x2 = x[..., hd // 2 :]
        return ttnn.concat([ttnn.neg(x2), x1], dim=-1)

    def _rotate_half4(x):
        # x: (1, heads, T, hd)
        b, hnum, t, _ = x.shape
        x1 = ttnn.slice(x, [0, 0, 0, 0], [b, hnum, t, hd // 2])
        x2 = ttnn.slice(x, [0, 0, 0, hd // 2], [b, hnum, t, hd])
        return ttnn.concat([ttnn.neg(x2), x1], dim=-1)

    def _repeat_kv4(x, T):
        # (1, nkv, T, hd) -> (1, nh, T, hd) block-repeat (HF repeat_kv order)
        if groups == 1:
            return x
        x = ttnn.reshape(x, (1, nkv, 1, T, hd))
        x = ttnn.repeat(x, (1, 1, groups, 1, 1))
        return ttnn.reshape(x, (1, nkv * groups, T, hd))

    def _kv_attention(q, k, v, cos_dev, sin_dev, mask_dev, kv_buf, cache_pos, write_onehot):
        """Fixed-capacity KV-cache attention (4D, float32). Prefill (S>1) seeds
        the cache with `fill_cache`; decode (S==1) writes the single token's K/V
        via a traceable one-hot select (`write_onehot`) or `update_cache` at
        `cache_pos`, then attends over the whole capacity C. Returns attn_out
        (1, S, nh*hd) bf16 (o_proj applied)."""
        S = int(q.shape[-2])
        q4 = ttnn.permute(ttnn.reshape(ttnn.typecast(q, ttnn.float32), (1, S, nh, hd)), (0, 2, 1, 3))
        k4 = ttnn.permute(ttnn.reshape(ttnn.typecast(k, ttnn.float32), (1, S, nkv, hd)), (0, 2, 1, 3))
        v4 = ttnn.permute(ttnn.reshape(ttnn.typecast(v, ttnn.float32), (1, S, nkv, hd)), (0, 2, 1, 3))
        cos4 = ttnn.reshape(ttnn.typecast(cos_dev, ttnn.float32), (1, 1, S, hd))
        sin4 = ttnn.reshape(ttnn.typecast(sin_dev, ttnn.float32), (1, 1, S, hd))
        q4 = ttnn.add(ttnn.mul(q4, cos4), ttnn.mul(_rotate_half4(q4), sin4))
        k4 = ttnn.add(ttnn.mul(k4, cos4), ttnn.mul(_rotate_half4(k4), sin4))

        k_cache, v_cache = kv_buf
        if S > 1:  # prefill: seed cache from position 0, attend locally causal
            ttnn.fill_cache(k_cache, k4, 0)
            ttnn.fill_cache(v_cache, v4, 0)
            k_att, v_att, T_kv = k4, v4, S
        elif write_onehot is not None:  # traceable decode write (device-side position)
            from models.demos.qwen2_vl.qwen2_vl_7b_instruct._stubs.kv_cache_select_op import kv_cache_select

            kv_cache_select(k_cache, write_onehot, k4)
            kv_cache_select(v_cache, write_onehot, v4)
            k_att, v_att, T_kv = k_cache, v_cache, int(k_cache.shape[2])
        else:  # eager decode write at a python index
            ttnn.update_cache(k_cache, k4, cache_pos)
            ttnn.update_cache(v_cache, v4, cache_pos)
            k_att, v_att, T_kv = k_cache, v_cache, int(k_cache.shape[2])

        kr = _repeat_kv4(k_att, T_kv)
        vr = _repeat_kv4(v_att, T_kv)
        scores = ttnn.matmul(q4, kr, transpose_b=True, memory_config=_DRAM, compute_kernel_config=_kcfg)
        scores = ttnn.mul(scores, scaling)
        m = mask_dev if mask_dev.get_dtype() == ttnn.float32 else ttnn.typecast(mask_dev, ttnn.float32)
        scores = ttnn.add(scores, m)
        probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=_kcfg)
        out = ttnn.matmul(probs, vr, memory_config=_DRAM, compute_kernel_config=_kcfg)  # (1,nh,S,hd)
        out = ttnn.reshape(ttnn.permute(out, (0, 2, 1, 3)), (1, S, nh * hd))
        return ttnn.typecast(_linear(out, o_w), ttnn.bfloat16)

    def forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        position_embeddings=None,
        cos_dev=None,
        sin_dev=None,
        mask_dev=None,
        kv_buf=None,
        cache_pos=None,
        write_onehot=None,
        **kwargs,
    ):
        # Trace/2CQ path: when `cos_dev`/`sin_dev`/`mask_dev` are supplied as
        # already-uploaded persistent device buffers, NO host op (torch collapse
        # / from_torch) runs inside this step -- the forward is pure ttnn.
        if hidden_states.get_dtype() != ttnn.bfloat16:
            hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)

        S = int(hidden_states.shape[-2])

        residual = hidden_states
        h = ttnn.rms_norm(hidden_states, weight=input_ln_w, epsilon=eps)

        q = _linear(h, q_w, q_b)
        k = _linear(h, k_w, k_b)
        v = _linear(h, v_w, v_b)

        if kv_buf is not None:
            # Fixed-capacity KV-cache path (prefill fills it, decode appends one
            # token + attends over the whole capacity). Requires pre-collapsed
            # mrope cos/sin device buffers + an additive mask, both supplied by
            # the pipeline's resident KV setup.
            attn_out = _kv_attention(q, k, v, cos_dev, sin_dev, mask_dev, kv_buf, cache_pos, write_onehot)
            hidden_states = ttnn.add(residual, attn_out)

            residual2 = hidden_states
            h2 = ttnn.rms_norm(hidden_states, weight=post_ln_w, epsilon=eps)
            gate = _linear(h2, gate_w)
            up = _linear(h2, up_w)
            mlp_out = _linear(ttnn.mul(ttnn.silu(gate), up), down_w)
            return ttnn.add(residual2, mlp_out)

        q = ttnn.permute(ttnn.reshape(q, (S, nh, hd)), (1, 0, 2))  # (nh, S, hd)
        k = ttnn.permute(ttnn.reshape(k, (S, nkv, hd)), (1, 0, 2))  # (nkv, S, hd)
        v = ttnn.permute(ttnn.reshape(v, (S, nkv, hd)), (1, 0, 2))

        if cos_dev is None:
            cos_raw, sin_raw = position_embeddings  # each (3, B, S, hd) torch tensors
            cos_cat = torch.concatenate([m[i % 3] for i, m in enumerate(cos_raw.split(mrope_section, dim=-1))], dim=-1)
            sin_cat = torch.concatenate([m[i % 3] for i, m in enumerate(sin_raw.split(mrope_section, dim=-1))], dim=-1)
            cos_dev = ttnn.from_torch(
                cos_cat[0:1].float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )  # (1, S, hd)
            sin_dev = ttnn.from_torch(sin_cat[0:1].float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        q = ttnn.add(ttnn.mul(q, cos_dev), ttnn.mul(_rotate_half(q), sin_dev))
        k = ttnn.add(ttnn.mul(k, cos_dev), ttnn.mul(_rotate_half(k), sin_dev))

        if groups > 1:
            k = ttnn.repeat_interleave(k, groups, dim=0)
            v = ttnn.repeat_interleave(v, groups, dim=0)

        attn_scores = ttnn.matmul(
            q, k, transpose_b=True, memory_config=_DRAM, compute_kernel_config=_kcfg
        )  # (nh, S, S)
        attn_scores = ttnn.mul(attn_scores, scaling)

        if mask_dev is not None:
            attn_scores = ttnn.add(attn_scores, mask_dev)
        elif attention_mask is not None:
            mask_dev = ttnn.from_torch(
                attention_mask.reshape(1, 1, -1).float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            attn_scores = ttnn.add(attn_scores, mask_dev)
        elif S > 1:
            # `Qwen2VLAttention` runs under sdpa (`config._attn_implementation
            # == "sdpa"`), which auto-derives `is_causal=True` whenever no
            # explicit `attention_mask` is given and `q_len > 1` (see
            # `sdpa_attention_forward`) -- i.e. the causal mask is applied
            # even though this per-layer call passes `attention_mask=None`.
            # Skipping it reproduces bidirectional (non-causal) attention and
            # silently diverges from the reference for any seq_len > 1.
            causal_mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
            mask_dev = ttnn.from_torch(
                causal_mask.reshape(1, S, S), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            attn_scores = ttnn.add(attn_scores, mask_dev)

        attn_probs = ttnn.softmax(attn_scores, dim=-1, compute_kernel_config=_kcfg)
        attn_out = ttnn.matmul(attn_probs, v, memory_config=_DRAM, compute_kernel_config=_kcfg)  # (nh, S, hd)

        attn_out = ttnn.reshape(ttnn.permute(attn_out, (1, 0, 2)), (1, S, nh * hd))
        attn_out = _linear(attn_out, o_w)

        hidden_states = ttnn.add(residual, attn_out)

        residual2 = hidden_states
        h2 = ttnn.rms_norm(hidden_states, weight=post_ln_w, epsilon=eps)
        gate = _linear(h2, gate_w)
        up = _linear(h2, up_w)
        mlp_out = _linear(ttnn.mul(ttnn.silu(gate), up), down_w)

        hidden_states = ttnn.add(residual2, mlp_out)
        return hidden_states

    return forward


def qwen2_v_l_decoder_layer(*args, **kwargs):
    raise RuntimeError(
        "qwen2_v_l_decoder_layer requires build(device, torch_module) to bind trained weights; "
        "the bare callable has no parameters."
    )
