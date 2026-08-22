# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `qwen2_decoder_layer` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.language_model.layers.0`, a standard HF
`transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer`
(hidden_size=1536, num_attention_heads=12, num_key_value_heads=2,
head_dim=128, intermediate_size=8960, rms_norm_eps=1e-6):

    residual = x
    x = input_layernorm(x)                      # RMSNorm
    x = self_attn(x, position_embeddings, attention_mask)  # GQA + RoPE. The optional additive
                                                  # attention_mask ([1,1,T,T] or [1,T,T] of 0/-inf) is added to
                                                  # the scaled scores right before the softmax; when None
                                                  # (the isolated per-layer PCC harness) behavior is unchanged.
                                                  # The full model (qwen2_model) composes this layer and passes
                                                  # the causal mask through this kwarg.
    x = residual + x

    residual = x
    x = post_attention_layernorm(x)              # RMSNorm
    x = mlp(x)                                   # SwiGLU
    x = residual + x

`self_attn` GQA: 12 query heads, 2 KV heads (n_rep=6, `repeat_kv`-style
block-repeat, not interleaved), RoPE via `rotate_half(x)*sin + x*cos`
(cos/sin supplied by the harness as the `position_embeddings` kwarg,
broadcast over the head dim), scaling = head_dim**-0.5, softmax attention,
o_proj (no bias). `mlp` is the same SwiGLU FFN as `_stubs/feed_forward_network.py`.
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs import _precision as prec

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the trained Qwen2DecoderLayer weights and return a native ttnn forward closure."""
    m = torch_module
    attn = m.self_attn
    mlp = m.mlp

    hidden_size = attn.q_proj.weight.shape[1]
    num_heads = attn.q_proj.weight.shape[0] // attn.head_dim
    num_kv_heads = attn.k_proj.weight.shape[0] // attn.head_dim
    head_dim = attn.head_dim
    n_rep = num_heads // num_kv_heads
    scaling = float(attn.scaling)

    input_ln_eps = float(m.input_layernorm.variance_epsilon)
    post_ln_eps = float(m.post_attention_layernorm.variance_epsilon)

    def _mm_weight(w):
        # Projection / MLP weights are stored in the perf precision (bf16 default): these matmuls
        # dominate LM cost and are memory-bound at decode (T=1), so bf16 weights ~halve their DRAM
        # traffic. Activations stay f32 (cast in/out around each matmul), and the attention SCORE
        # matmuls (q@kᵀ, attn@v — no stored weight, softmax-sensitive) stay full fp32 below.
        return prec.mm_weight(w.detach().float().t().contiguous(), device)

    def _row(t_1d):
        return ttnn.from_torch(
            t_1d.detach().float().reshape(1, 1, -1).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    input_ln_w = _row(m.input_layernorm.weight)
    post_ln_w = _row(m.post_attention_layernorm.weight)

    q_w = _mm_weight(attn.q_proj.weight)
    q_b = _row(attn.q_proj.bias) if attn.q_proj.bias is not None else None
    k_w = _mm_weight(attn.k_proj.weight)
    k_b = _row(attn.k_proj.bias) if attn.k_proj.bias is not None else None
    v_w = _mm_weight(attn.v_proj.weight)
    v_b = _row(attn.v_proj.bias) if attn.v_proj.bias is not None else None
    o_w = _mm_weight(attn.o_proj.weight)

    gate_w = _mm_weight(mlp.gate_proj.weight)
    up_w = _mm_weight(mlp.up_proj.weight)
    down_w = _mm_weight(mlp.down_proj.weight)

    # weight matmuls follow the perf precision; the attention score matmuls stay full fp32.
    mm_config = prec.compute_config(device)
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def _to_ttnn_f32(t):
        if not isinstance(t, ttnn.Tensor):
            return ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        return ttnn.typecast(t, ttnn.float32) if t.get_dtype() != ttnn.float32 else t

    def _rms_norm(x, weight_row, eps):
        # Native fused RMSNorm (single device op) replaces the manual
        # mul+mean(reduce)+add+rsqrt+mul+mul chain -- removes the dispatch-bound
        # ReduceDeviceOperation call entirely for this call site (GUIDELINES/02).
        return ttnn.rms_norm(x, epsilon=eps, weight=weight_row, memory_config=_DRAM)

    def _linear(x, w, b):
        y = prec.matmul(x, w, mm_config)  # bf16 matmul (weight bf16, activation cast to bf16)
        if y.get_dtype() != ttnn.float32:
            y = ttnn.typecast(y, ttnn.float32)  # back to f32 for RoPE / norms / softmax downstream
        if b is not None:
            y = ttnn.add(y, b, memory_config=_DRAM)
        return y

    def _split_heads(x, B, T, n_heads):
        # [B, T, n_heads*head_dim] -> [B, n_heads, T, head_dim]
        x = ttnn.reshape(x, (B, T, n_heads, head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _merge_heads(x, B, T, n_heads):
        # [B, n_heads, T, head_dim] -> [B, T, n_heads*head_dim]
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (B, T, n_heads * head_dim))

    def _rotate_half(x):
        half = head_dim // 2
        shape = list(x.shape)
        x1 = ttnn.slice(x, [0, 0, 0, 0], shape[:3] + [half])
        x2 = ttnn.slice(x, [0, 0, 0, half], shape[:3] + [head_dim])
        return ttnn.concat([ttnn.mul(x2, -1.0), x1], dim=-1, memory_config=_DRAM)

    def _apply_rope(x, cos, sin):
        # cos, sin: [B, 1, T, head_dim] (already unsqueezed for the head dim)
        return ttnn.add(
            ttnn.mul(x, cos, memory_config=_DRAM),
            ttnn.mul(_rotate_half(x), sin, memory_config=_DRAM),
            memory_config=_DRAM,
        )

    def _repeat_kv(x, B, T):
        # [B, num_kv_heads, T, head_dim] -> [B, num_heads, T, head_dim], block-repeat (not interleaved).
        if n_rep == 1:
            return x
        x = ttnn.reshape(x, (B, num_kv_heads, 1, T, head_dim))
        x = ttnn.repeat(x, (1, 1, n_rep, 1, 1))
        return ttnn.reshape(x, (B, num_kv_heads * n_rep, T, head_dim))

    def _swiglu_mlp(x):
        gate = ttnn.silu(prec.matmul(x, gate_w, mm_config), memory_config=_DRAM)
        up = prec.matmul(x, up_w, mm_config)
        h = ttnn.mul(gate, up, memory_config=_DRAM)
        out = prec.matmul(h, down_w, mm_config)
        return ttnn.typecast(out, ttnn.float32) if out.get_dtype() != ttnn.float32 else out

    def forward(
        hidden_states,
        *args,
        position_embeddings=None,
        attention_mask=None,
        past_kv=None,
        use_cache=False,
        kv_buf=None,
        cache_pos=None,
        write_onehot=None,
        **kwargs,
    ):
        """GQA + RoPE + SwiGLU decoder layer.

        Three cache modes (all opt-in; the defaults reproduce the original full-sequence path
        byte-for-byte, so the isolated per-layer PCC harness is unaffected):

        * **none** (`past_kv=None, kv_buf=None`): full attention over the T input tokens.
        * **growing cache** (`use_cache=True`, `past_kv=(k,v)`): the new tokens' post-RoPE K /
          pre-RoPE V are `concat`-appended to `past_kv` (both `[B, num_kv_heads, T_past, head_dim]`)
          and the updated `(k_all, v_all)` is returned. O(L) but grows the allocation each step.
        * **fixed-capacity** (`kv_buf=(k_cache, v_cache)` pre-allocated `[B, num_kv_heads, C, head_dim]`):
          the K/V are written IN PLACE — `fill_cache` at seq 0 for a prefill block (T>1), or
          `update_cache` at `cache_pos` for a single decode token (T==1) — so no per-step allocation
          grows. A decode step attends over the FULL capacity C (fixed shape, trace-ready); the
          caller supplies an additive `[B,1,1,C]` mask that blocks the not-yet-written positions.

        RoPE is applied at each token's own absolute position (via the caller's cos/sin), so cached
        K stays correct across steps."""
        x = _to_ttnn_f32(hidden_states)
        B, T, _ = x.shape

        cos, sin = position_embeddings
        cos = _to_ttnn_f32(cos)
        sin = _to_ttnn_f32(sin)
        cos = ttnn.reshape(cos, (B, 1, T, head_dim))
        sin = ttnn.reshape(sin, (B, 1, T, head_dim))

        residual = x
        h = _rms_norm(x, input_ln_w, input_ln_eps)

        q = _split_heads(_linear(h, q_w, q_b), B, T, num_heads)
        k = _split_heads(_linear(h, k_w, k_b), B, T, num_kv_heads)
        v = _split_heads(_linear(h, v_w, v_b), B, T, num_kv_heads)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        new_kv = None
        if kv_buf is not None:
            # fixed-capacity KV: write in place, attend over the full pre-allocated capacity.
            k_cache, v_cache = kv_buf
            if T > 1:  # prefill: seed the cache from seq 0
                ttnn.fill_cache(k_cache, k, 0)
                ttnn.fill_cache(v_cache, v, 0)
                k_att, v_att, T_kv = k, v, T  # prefill still does local causal attention
            elif write_onehot is not None:
                # TRACEABLE decode write: no baked seq index. Write the token's K/V into the
                # `write_onehot`-selected row via a single fused C++ Metalium kernel
                # (models/demos/vibevoice_1_5b/_stubs/kv_cache_select_op.py, invoked through
                # ttnn.generic_op) that computes cache_new = cache + onehot*(new - cache) in one
                # pass (reads cache/onehot/new tiles from DRAM, does the broadcast + select
                # entirely in L1/dst-registers, writes straight back into the cache's own DRAM
                # buffer) — no materialized DRAM intermediates. Because the position lives in the
                # staged `write_onehot` tensor (not a Python int), one captured trace is valid at
                # every decode position; the in-place write accumulates the cache across replays.
                #   cache = cache*(1 - onehot) + k*onehot     (onehot: [B,1,C,1], k: [B,kv,1,head_dim])
                from models.demos.vibevoice_1_5b._stubs.kv_cache_select_op import kv_cache_select

                kv_cache_select(k_cache, write_onehot, k)
                kv_cache_select(v_cache, write_onehot, v)
                k_att, v_att, T_kv = k_cache, v_cache, int(k_cache.shape[2])
            else:  # decode: append one token at cache_pos, attend over all C
                ttnn.update_cache(k_cache, k, cache_pos)
                ttnn.update_cache(v_cache, v, cache_pos)
                k_att, v_att, T_kv = k_cache, v_cache, int(k_cache.shape[2])
        else:
            # growing cache: append new tokens' K/V (pre-repeat) onto the running cache
            if past_kv is not None:
                k_past, v_past = past_kv
                k = ttnn.concat([k_past, k], dim=2, memory_config=_DRAM)
                v = ttnn.concat([v_past, v], dim=2, memory_config=_DRAM)
            new_kv = (k, v) if use_cache else None
            k_att, v_att, T_kv = k, v, int(k.shape[2])

        kr = _repeat_kv(k_att, B, T_kv)
        vr = _repeat_kv(v_att, B, T_kv)

        attn_weights = ttnn.matmul(
            q, ttnn.permute(kr, (0, 1, 3, 2)), compute_kernel_config=compute_config, memory_config=_DRAM
        )
        attn_weights = ttnn.mul(attn_weights, scaling, memory_config=_DRAM)
        if attention_mask is not None:
            mask = _to_ttnn_f32(attention_mask)
            if len(mask.shape) == 3:
                # [1, T, T] -> [1, 1, T, T] so it broadcasts over the head dim.
                mask = ttnn.reshape(mask, (mask.shape[0], 1, mask.shape[1], mask.shape[2]))
            attn_weights = ttnn.add(attn_weights, mask, memory_config=_DRAM)
        attn_weights = ttnn.softmax(attn_weights, dim=-1)
        attn_output = ttnn.matmul(attn_weights, vr, compute_kernel_config=compute_config, memory_config=_DRAM)
        attn_output = _merge_heads(attn_output, B, T, num_heads)
        attn_output = _linear(attn_output, o_w, None)

        x = ttnn.add(residual, attn_output, memory_config=_DRAM)

        residual = x
        h = _rms_norm(x, post_ln_w, post_ln_eps)
        h = _swiglu_mlp(h)
        x = ttnn.add(residual, h, memory_config=_DRAM)

        if use_cache:
            return x, new_kv
        return x

    return forward


def qwen2_decoder_layer(*args, **kwargs):
    raise RuntimeError(
        "qwen2_decoder_layer requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
