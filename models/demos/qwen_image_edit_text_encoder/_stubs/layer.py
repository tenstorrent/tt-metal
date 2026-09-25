# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of the Qwen2.5-VL text decoder layer
(`Qwen2_5_VLDecoderLayer`, `model.language_model.layers[i]`), prefill (no KV cache).

    h = x + o_proj(attn(rope_mrope(q, k), v))       GQA: 28 q heads / 4 kv heads, head_dim 128, causal
    y = h + down(silu(gate(norm2(h))) * up(norm2(h)))

TP scheme (tt_transformers attention.py / mlp.py) on the TP (column) axis, DP axis replicated:
    * q/k/v fused and column-parallel by KV GROUP: chip d owns kv heads [d*kvl, (d+1)*kvl) and the q heads
      that attend to them, so GQA needs no cross-chip traffic. Biases shard with their columns.
    * o_proj row-parallel over the same local heads -> all_reduce.
    * MLP gate/up column-parallel, down row-parallel -> all_reduce.
    * norms replicated; the residual stream is kept in fp32.
mRoPE: the (t, h, w) cos/sin sections are combined on host (metadata, per HF
apply_multimodal_rotary_pos_emb) and applied on device with rotate_half == x @ R.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import ttnn
from models.demos.qwen_image_edit_text_encoder._stubs.attention import (
    MASK_NEG,
    exact_all_reduce,
    hifi4_config,
    mesh_shape,
    pad_to_tile,
    rotate_half,
    shard_mapper,
    split_linear,
    split_matmul,
    upload,
    upload_rows,
)
from models.demos.qwen_image_edit_text_encoder._stubs.encoder_stack import TtRMSNorm, _bf16, _fp32


def _t(lin):
    return lin.weight.detach().float().t().contiguous()


def rotate_half_matrix(d):
    half = d // 2
    r = np.zeros((d, d), dtype=np.float32)
    r[np.arange(half) + half, np.arange(half)] = -1.0
    r[np.arange(half), np.arange(half) + half] = 1.0
    return r.reshape(1, 1, d, d)


def mrope_tables(cos, sin, mrope_section):
    """HF apply_multimodal_rotary_pos_emb section select: [3, B, S, D] -> [B, S, D] (numpy fp32).
    Already-combined [B, S, D] tables pass through."""
    cos = cos.float().numpy() if isinstance(cos, torch.Tensor) else np.asarray(cos, dtype=np.float32)
    sin = sin.float().numpy() if isinstance(sin, torch.Tensor) else np.asarray(sin, dtype=np.float32)
    if cos.ndim == 3:
        return cos, sin
    bounds = np.cumsum([0] + list(mrope_section) * 2)
    idx = np.zeros(cos.shape[-1], dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        idx[lo:hi] = i % 3
    cols = np.arange(cos.shape[-1])
    return cos[idx, :, :, cols].transpose(1, 2, 0), sin[idx, :, :, cols].transpose(1, 2, 0)


def text_attention_mask(attention_mask, b, s, s_pad, rep):
    """Additive [B, 1, rep*s_pad, s_pad] mask: causal (HF default when the mask is None) or the given
    4D mask (bool: True = attend; float: additive), tiled `rep` times along rows (GQA row packing)."""
    causal = np.where(np.arange(s_pad)[None, :] <= np.arange(s_pad)[:, None], 0.0, MASK_NEG).astype(np.float32)
    causal[:, s:] = MASK_NEG
    causal[s:, :] = 0.0
    mask = np.broadcast_to(causal, (b, 1, s_pad, s_pad)).copy()
    if attention_mask is not None:
        am = attention_mask.numpy() if attention_mask.dtype == torch.bool else attention_mask.float().numpy()
        if am.ndim == 2:  # [B, S] padding mask, nonzero = keep
            add = np.where(am.astype(bool), 0.0, MASK_NEG)[:, None, None, :]
        elif am.dtype == bool:  # [B, 1, S, S], True = attend
            add = np.where(am, 0.0, MASK_NEG)
        else:  # [B, 1, S, S] additive
            add = am
        add = np.broadcast_to(add, (b, 1, s, s)).astype(np.float32)
        mask[:, :, :s, :s] = np.maximum(mask[:, :, :s, :s] + add, MASK_NEG)
    return np.tile(mask, (1, 1, rep, 1))


class TtTextAttention:
    def __init__(self, device, torch_module, pair=None):
        """pair: a second attention module placed on mesh row 1 (row-staged layers); same TP split."""
        self.device = device
        _, self.tp = mesh_shape(device)
        cfg = torch_module.config
        self.num_heads = int(torch_module.num_heads)
        self.num_kv = int(torch_module.num_key_value_heads)
        self.head_dim = int(torch_module.head_dim)
        self.mrope_section = list(cfg.rope_parameters["mrope_section"])
        assert self.num_kv % self.tp == 0, f"{self.num_kv} kv heads not divisible by TP={self.tp}"
        self.kvl = self.num_kv // self.tp
        self.group = self.num_heads // self.num_kv
        self.hl = self.kvl * self.group
        self.scale = 1.0 / math.sqrt(self.head_dim)
        D, hl, kvl = self.head_dim, self.hl, self.kvl

        def _fused(m):
            wq, wk, wv = _t(m.q_proj), _t(m.k_proj), _t(m.v_proj)  # [C, out]
            bq, bk, bv = (p.bias.detach().float() for p in (m.q_proj, m.k_proj, m.v_proj))
            w_blocks, b_blocks = [], []
            for d in range(self.tp):
                qs, ks = slice(d * hl * D, (d + 1) * hl * D), slice(d * kvl * D, (d + 1) * kvl * D)
                w_blocks += [wq[:, qs], wk[:, ks], wv[:, ks]]
                b_blocks += [bq[qs], bk[ks], bv[ks]]
            return torch.cat(w_blocks, dim=1), torch.cat(b_blocks).reshape(1, 1, 1, -1), _t(m.o_proj)

        w, b, wo = _fused(torch_module)
        if pair is not None:
            w2, b2, wo2 = _fused(pair)
            self.wqkv = upload_rows(device, [w, w2], col_dim=-1)
            self.bqkv = upload_rows(device, [b, b2], col_dim=-1)
            self.wo = upload_rows(device, [wo, wo2], col_dim=0)
        else:
            col = shard_mapper(device, -1)
            self.wqkv = upload(device, w, mapper=col)
            self.bqkv = upload(device, b, mapper=col)
            self.wo = upload(device, wo, mapper=shard_mapper(device, 0))
        self.rot = upload(device, rotate_half_matrix(D), dtype=ttnn.float32)
        self.compute_cfg = hifi4_config()
        # precise: every matmul input carried as bf16 hi + lo (exact products), probs @ v in float32,
        # o_proj out in float32 with an exact TP reduce
        self.precise = False

    def rope_tables(self, position_embeddings, s_pad):
        cos, sin = mrope_tables(*position_embeddings, self.mrope_section)  # [B, S, D]
        b, s, d = cos.shape
        pad = np.zeros((b, s_pad - s, d), dtype=np.float32)
        cos = np.concatenate([cos, pad], axis=1).reshape(b, 1, s_pad, d)
        sin = np.concatenate([sin, pad], axis=1).reshape(b, 1, s_pad, d)
        return upload(self.device, cos, dtype=ttnn.float32), upload(self.device, sin, dtype=ttnn.float32)

    def mask(self, attention_mask, b, s, s_pad):
        return upload(self.device, text_attention_mask(attention_mask, b, s, s_pad, self.group), dtype=ttnn.float32)

    def forward_padded(self, x, tt_cos, tt_sin, tt_mask):
        """x: [B, 1, s_pad, C] bf16 replicated -> [B, 1, s_pad, C] bf16 replicated."""
        D, hl, kvl, G = self.head_dim, self.hl, self.kvl, self.group
        b, s_pad = x.shape[0], x.shape[-2]
        if self.precise:
            return self._forward_precise(x, tt_cos, tt_sin, tt_mask)
        # q/k carry large biases in Qwen2 (scores of O(100s)); keep the score path in fp32.
        qkv = ttnn.linear(x, self.wqkv, bias=self.bqkv, compute_kernel_config=self.compute_cfg, dtype=ttnn.float32)

        def _heads(lo, n):
            t = ttnn.slice(qkv, [0, 0, 0, lo * D], [b, 1, s_pad, (lo + n) * D])
            t = ttnn.reshape(t, (b, s_pad, n, D))
            return ttnn.permute(t, (0, 2, 1, 3))  # [B, n, S, D]

        q, k, v = _heads(0, hl), _heads(hl, kvl), _heads(hl + kvl, kvl)

        def _rope(t):
            rot = ttnn.matmul(t, self.rot, compute_kernel_config=self.compute_cfg)
            return ttnn.add(ttnn.multiply(t, tt_cos), ttnn.multiply(rot, tt_sin))

        q, k = _rope(q), _rope(k)
        # GQA: pack the G q heads sharing a kv head along rows -> [B, kvl, G*S, D]
        q = ttnn.reshape(q, (b, kvl, G * s_pad, D))
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.compute_cfg, dtype=ttnn.float32)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), tt_mask)
        probs = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=self.compute_cfg)
        o = ttnn.matmul(
            ttnn.typecast(probs, ttnn.bfloat16), ttnn.typecast(v, ttnn.bfloat16), compute_kernel_config=self.compute_cfg
        )  # [B, kvl, G*S, D]
        o = ttnn.reshape(o, (b, hl, s_pad, D))
        o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.reshape(o, (b, 1, s_pad, hl * D))

        out = ttnn.linear(o, self.wo, compute_kernel_config=self.compute_cfg)
        if self.tp > 1:
            out = ttnn.all_reduce(out, cluster_axis=1, topology=ttnn.Topology.Linear)
        return out

    def _forward_precise(self, x, tt_cos, tt_sin, tt_mask):
        D, hl, kvl, G = self.head_dim, self.hl, self.kvl, self.group
        b, s_pad = x.shape[0], x.shape[-2]
        cfg = self.compute_cfg
        ex = getattr(self, "exact", True)
        qkv = split_linear(x, self.wqkv, bias=self.bqkv, compute_kernel_config=cfg, exact=ex)

        def _heads(lo, n):
            t = ttnn.slice(qkv, [0, 0, 0, lo * D], [b, 1, s_pad, (lo + n) * D])
            t = ttnn.reshape(t, (b, s_pad, n, D))
            return ttnn.permute(t, (0, 2, 1, 3))

        q, k, v = _heads(0, hl), _heads(hl, kvl), _heads(hl + kvl, kvl)
        cos = tt_cos if tt_cos.dtype == ttnn.float32 else ttnn.typecast(tt_cos, ttnn.float32)
        sin = tt_sin if tt_sin.dtype == ttnn.float32 else ttnn.typecast(tt_sin, ttnn.float32)

        def _rope(t):
            return ttnn.add(ttnn.multiply(t, cos), ttnn.multiply(rotate_half(t), sin))

        q, k = _rope(q), _rope(k)
        q = ttnn.reshape(q, (b, kvl, G * s_pad, D))
        scores = split_matmul(q, k, transpose_b=True, compute_kernel_config=cfg, exact=ex)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), tt_mask)
        mx = ttnn.max(scores, dim=-1, keepdim=True)
        e = ttnn.exp(ttnn.subtract(scores, mx))
        probs = ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=cfg))
        o = split_matmul(probs, v, compute_kernel_config=cfg, exact=ex)  # [B, kvl, G*S, D]
        o = ttnn.reshape(o, (b, hl, s_pad, D))
        o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.reshape(o, (b, 1, s_pad, hl * D))
        out = split_linear(o, self.wo, compute_kernel_config=cfg, exact=ex)
        if self.tp > 1:
            out = exact_all_reduce(out, self.device)
        return out


class TtTextMLP:
    """down(silu(gate(x)) * up(x)), no biases -- gate/up column-parallel, down row-parallel."""

    def __init__(self, device, torch_module, pair=None):
        """pair: a second MLP placed on mesh row 1 (row-staged layers); same TP split."""
        _, self.tp = mesh_shape(device)
        self.device = device
        if pair is not None:
            self.w_gate = upload_rows(device, [_t(torch_module.gate_proj), _t(pair.gate_proj)], col_dim=-1)
            self.w_up = upload_rows(device, [_t(torch_module.up_proj), _t(pair.up_proj)], col_dim=-1)
            self.w_down = upload_rows(device, [_t(torch_module.down_proj), _t(pair.down_proj)], col_dim=0)
        else:
            col = shard_mapper(device, -1)
            self.w_gate = upload(device, _t(torch_module.gate_proj), mapper=col)
            self.w_up = upload(device, _t(torch_module.up_proj), mapper=col)
            self.w_down = upload(device, _t(torch_module.down_proj), mapper=shard_mapper(device, 0))
        self.compute_cfg = hifi4_config()

    def __call__(self, x):
        if getattr(self, "precise", False):
            cfg = self.compute_cfg
            ex = getattr(self, "exact", True)
            gate = split_linear(x, self.w_gate, compute_kernel_config=cfg, exact=ex)
            up = split_linear(x, self.w_up, compute_kernel_config=cfg, exact=ex)
            out = split_linear(ttnn.multiply(ttnn.silu(gate), up), self.w_down, compute_kernel_config=cfg, exact=ex)
            if self.tp > 1:
                out = exact_all_reduce(out, self.device)
            return out
        gate = ttnn.linear(x, self.w_gate, compute_kernel_config=self.compute_cfg)
        up = ttnn.linear(x, self.w_up, compute_kernel_config=self.compute_cfg)
        out = ttnn.linear(ttnn.multiply(ttnn.silu(gate), up), self.w_down, compute_kernel_config=self.compute_cfg)
        if self.tp > 1:
            out = ttnn.all_reduce(out, cluster_axis=1, topology=ttnn.Topology.Linear)
        return out


class TtTextDecoderLayer:
    def __init__(self, device, torch_module, mlp=None, pair=None):
        """pair: a second decoder layer whose weights live on mesh row 1 (row-staged layers)."""
        self.device = device
        pn = (lambda name: getattr(pair, name)) if pair is not None else (lambda name: None)
        self.input_layernorm = TtRMSNorm(device, torch_module.input_layernorm, pair=pn("input_layernorm"))
        self.post_attention_layernorm = TtRMSNorm(
            device, torch_module.post_attention_layernorm, pair=pn("post_attention_layernorm")
        )
        self.self_attn = TtTextAttention(device, torch_module.self_attn, pair=pn("self_attn"))
        self.mlp = mlp if mlp is not None else TtTextMLP(device, torch_module.mlp, pair=pn("mlp"))

    def forward_padded(self, x, tt_cos, tt_sin, tt_mask):
        """x: [B, 1, s_pad, C] fp32 residual stream (branch inputs stay fp32 with precise_inputs)."""
        cast = _fp32 if getattr(self, "precise_inputs", False) else _bf16
        a = self.self_attn.forward_padded(cast(self.input_layernorm(x)), tt_cos, tt_sin, tt_mask)
        x = ttnn.add(x, _fp32(a))
        return ttnn.add(x, _fp32(self.mlp(cast(self.post_attention_layernorm(x)))))

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        b, s, c = hidden_states.shape
        s_pad = pad_to_tile(s)
        tt_cos, tt_sin = self.self_attn.rope_tables(position_embeddings, s_pad)
        tt_mask = self.self_attn.mask(attention_mask, b, s, s_pad)
        x = ttnn.reshape(hidden_states, (b, 1, s, c))
        if s_pad != s:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, s_pad - s), (0, 0)], 0.0)
        x = self.forward_padded(_fp32(x), tt_cos, tt_sin, tt_mask)
        if s_pad != s:
            x = ttnn.slice(x, [0, 0, 0, 0], [b, 1, s, c])
        return ttnn.reshape(x, (b, s, c))


def build(device, torch_module=None):
    return TtTextDecoderLayer(device, torch_module)


def layer(device, torch_module=None):
    return TtTextDecoderLayer(device, torch_module)
