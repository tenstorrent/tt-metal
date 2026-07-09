# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `image3_decoder_layer` (HunyuanImage3DecoderLayer) of
tencent/HunyuanImage-3.0.

A full decoder block:

    residual = x
    x = input_layernorm(x)                    # RMSNorm
    x = self_attn(x, custom_pos_emb)          # fused-QKV GQA + HF rope + qk-norm + SDPA + o_proj
    x = residual + x
    residual = x
    x = post_attention_layernorm(x)           # RMSNorm
    x = mlp(x)                                 # MoE: shared SwiGLU + top-8 gate + 64 expert SwiGLUs
    x = residual + x

Everything runs in TTNN ops on device; weights are extracted from the HF
reference module at `build()` time. No torch reference is invoked at forward
time.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0._stubs import mo_e as _mo_e

HF_MODEL_ID = "tencent/HunyuanImage-3.0"


def _to_ttnn(t: torch.Tensor, device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t.to(torch.float32),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _linear_weight(w: torch.Tensor, device, dtype=ttnn.bfloat16):
    """nn.Linear stores weight as [out, in]; ttnn.linear(x, W) computes x @ W,
    so upload the transpose [in, out]."""
    return _to_ttnn(w.t().contiguous(), device, dtype=dtype)


class _TtDecoderLayer:
    def __init__(self, device, torch_module):
        self.device = device
        cfg = torch_module.self_attn

        self.num_heads = int(cfg.num_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(cfg.head_dim)
        self.groups = self.num_heads // self.num_kv_heads
        self.hidden_size = int(cfg.hidden_size)
        self.use_qk_norm = bool(getattr(cfg, "use_qk_norm", False))
        self.use_rope = bool(getattr(cfg, "use_rotary_pos_emb", False))
        self.scale = 1.0 / (self.head_dim**0.5)

        self.eps = float(torch_module.input_layernorm.variance_epsilon)

        # --- norms ---
        self.input_ln_w = _to_ttnn(torch_module.input_layernorm.weight.reshape(1, 1, 1, -1), device)
        self.post_ln_w = _to_ttnn(torch_module.post_attention_layernorm.weight.reshape(1, 1, 1, -1), device)
        if self.use_qk_norm:
            self.q_norm_w = _to_ttnn(cfg.query_layernorm.weight.reshape(1, 1, 1, -1), device)
            self.k_norm_w = _to_ttnn(cfg.key_layernorm.weight.reshape(1, 1, 1, -1), device)

        # --- attention: permute fused-QKV rows so output is [q | k | v] contiguous ---
        # HF layout: qkv[b,q,6144] -> reshape [b,q,kv=8,groups+2=6,hd=128], split [4,1,1].
        # flat row for (kv, slot, d) = kv*768 + slot*128 + d.  q uses slots 0..3, k=4, v=5.
        # q head index = kv*groups + group (matches HF reshape->transpose order).
        nKV, g, hd = self.num_kv_heads, self.groups, self.head_dim
        block = (g + 2) * hd
        q_rows, k_rows, v_rows = [], [], []
        for h in range(self.num_heads):
            kv = h // g
            grp = h % g
            base = kv * block + grp * hd
            q_rows.extend(range(base, base + hd))
        for kv in range(nKV):
            base = kv * block
            k_rows.extend(range(base + g * hd, base + g * hd + hd))
            v_rows.extend(range(base + (g + 1) * hd, base + (g + 1) * hd + hd))
        perm = q_rows + k_rows + v_rows
        qkv_w = cfg.qkv_proj.weight[perm, :].contiguous()  # [6144, 4096]
        self.qkv_w = _linear_weight(qkv_w, device)
        self.o_w = _linear_weight(cfg.o_proj.weight, device)

        # --- MoE: composed graduated HunyuanMoE port ---
        # HunyuanImage3DecoderLayer.mlp == HunyuanMoE. Delegate the whole MoE
        # (shared SwiGLU + top-8 gate + 64 routed expert SwiGLUs) to the
        # graduated `mo_e` stub, which itself composes the graduated
        # `top_k_gate`. This is the real HF module nesting.
        self.moe = _mo_e.build(device, torch_module.mlp)
        # Gate 2 real-invocation counter.
        self.num_calls = 0

    # ------------------------------------------------------------------
    def _attention(self, x, custom_pos_emb):
        S = x.shape[1]
        qkv = ttnn.linear(x, self.qkv_w)  # [1, S, 6144] = q|k|v
        # nlp_create_qkv_heads expects a 4D [B, 1, S, dim] fused-qkv tensor.
        qkv = ttnn.reshape(qkv, [1, 1, S, qkv.shape[-1]])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        # q: [1, num_heads, S, hd]; k,v: [1, num_kv_heads, S, hd]

        # HF rope (rotate_half) THEN qk-norm, matching the reference order.
        if self.use_rope and custom_pos_emb is not None:
            cos_t, sin_t = custom_pos_emb
            # Fast path for trace+2CQ: cos/sin may be PRE-UPLOADED persistent
            # ttnn buffers ([1,1,S,hd]) — use them directly so the traced step
            # is host-op-free (no from_torch inside the forward). Otherwise
            # (per-component / eager path) convert the torch cos/sin here.
            if isinstance(cos_t, ttnn.Tensor):
                cos, sin = cos_t, sin_t
            else:
                cos = _to_ttnn(cos_t.reshape(1, 1, cos_t.shape[-2], cos_t.shape[-1]), self.device)
                sin = _to_ttnn(sin_t.reshape(1, 1, sin_t.shape[-2], sin_t.shape[-1]), self.device)
            q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
            k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)

        if self.use_qk_norm:
            q = ttnn.rms_norm(q, epsilon=self.eps, weight=self.q_norm_w)
            k = ttnn.rms_norm(k, epsilon=self.eps, weight=self.k_norm_w)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        # attn: [1, num_heads, S, hd] -> concat heads -> [1, 1, S, hidden]
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(attn, [1, S, self.hidden_size])
        out = ttnn.linear(attn, self.o_w)
        ttnn.deallocate(attn)
        return out

    def __call__(self, hidden_states, custom_pos_emb=None, return_l_aux=False, **kwargs):
        self.num_calls += 1
        residual = hidden_states
        x = ttnn.rms_norm(hidden_states, epsilon=self.eps, weight=self.input_ln_w)
        attn = self._attention(x, custom_pos_emb)
        ttnn.deallocate(x)
        hidden = ttnn.add(residual, attn)
        ttnn.deallocate(attn)

        residual2 = hidden
        x2 = ttnn.rms_norm(hidden, epsilon=self.eps, weight=self.post_ln_w)
        # Composed graduated MoE (which composes the graduated gate).
        moe, l_aux = self.moe(x2, return_l_aux=True)
        ttnn.deallocate(x2)
        out = ttnn.add(residual2, moe)
        ttnn.deallocate(moe)
        if return_l_aux:
            return out, l_aux
        ttnn.deallocate(l_aux)
        return out


def build(device, torch_module):
    return _TtDecoderLayer(device, torch_module)


def image3_decoder_layer(*args, **kwargs):  # pragma: no cover - build() is the entry point
    raise RuntimeError(
        "image3_decoder_layer must be constructed via build(device, torch_module); "
        "the bare module-level callable is not supported for this native port."
    )
