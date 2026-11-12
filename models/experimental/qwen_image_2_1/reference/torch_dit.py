# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compact torch re-implementation of the Qwen-Image-2.1 DiT math (from diffusers transformer_qwenimage21.py)
for CPU unit tests of single blocks. Works on the raw checkpoint tensors via LazyCheckpoint."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..common import rope as rope_mod
from ..common.config import DIT
from ..common.schedule import StepConditioning


def rms_head_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """diffusers RMSNorm over the last dim (fp32 stats), then * weight."""
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (xf.to(x.dtype) * w.to(x.dtype)) if w.dtype in (torch.float16, torch.bfloat16) else (xf * w).to(x.dtype)


def block_forward(
    ckpt,
    idx: int,
    h: torch.Tensor,
    cond: StepConditioning,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_prefix=None,
    causal: bool = False,
    dtype=torch.bfloat16,
):
    """One block. h: [S, 4096]. cos/sin: [S, 128] adjacent-pair tables for these S tokens.
    kv_prefix: optional (k, v) [T, 32, 128] prepended to keys/values (cached text).
    Returns new h and this block's (k, v) [S, 32, 128] (post norm+rope)."""
    p = f"transformer_blocks.{idx}."
    g = lambda k: ckpt.get(p + k, dtype)
    c = DIT
    S = h.shape[0]
    x = F.layer_norm(h.float(), (c.hidden,), eps=c.eps).to(dtype) * cond.one_plus_scale1.to(dtype)
    q = (x @ g("attn.to_q.weight").t()).view(S, c.heads, c.head_dim)
    k = (x @ g("attn.to_k.weight").t()).view(S, c.heads, c.head_dim)
    v = (x @ g("attn.to_v.weight").t()).view(S, c.heads, c.head_dim)
    q = rms_head_norm(q, g("attn.norm_q.weight"), c.eps)
    k = rms_head_norm(k, g("attn.norm_k.weight"), c.eps)
    cs = cos.float()[:, None, :]
    sn = sin.float()[:, None, :]
    q = rope_mod.apply_rope_adjacent(q.float(), cs, sn).to(dtype)
    k = rope_mod.apply_rope_adjacent(k.float(), cs, sn).to(dtype)
    kk, vv = k, v
    if kv_prefix is not None:
        kk = torch.cat([kv_prefix[0].to(dtype), k], 0)
        vv = torch.cat([kv_prefix[1].to(dtype), v], 0)
    qh = q.permute(1, 0, 2)  # [H, S, D]
    kh = kk.permute(1, 0, 2)
    vh = vv.permute(1, 0, 2)
    o = F.scaled_dot_product_attention(
        qh.float(), kh.float(), vh.float(), is_causal=causal, scale=1.0 / math.sqrt(c.head_dim)
    )
    o = o.to(dtype).permute(1, 0, 2).reshape(S, c.hidden)
    o = o @ g("attn.to_out.0.weight").t()
    h = h + cond.tanh_gate1.to(dtype) * o
    x2 = F.layer_norm(h.float(), (c.hidden,), eps=c.eps).to(dtype) * cond.one_plus_scale2.to(dtype)
    m = F.silu(x2 @ g("img_mlp.gate_layer.weight").t()) * (x2 @ g("img_mlp.proj.weight").t())
    m = m @ g("img_mlp.out.weight").t()
    h = h + cond.tanh_gate2.to(dtype) * m
    return h, (k, v)


def text_project(ckpt, text_embeds: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    w = ckpt.get("txt_in.text_norm.weight", torch.float32)
    xf = text_embeds.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + DIT.eps) * (w + 1)
    x = xf.to(dtype)
    x = x @ ckpt.get("txt_in.in_layer.weight", dtype).t()
    x = F.gelu(x, approximate="tanh")
    return x @ ckpt.get("txt_in.out_layer.weight", dtype).t()


def final_layer(ckpt, h: torch.Tensor, cond: StepConditioning, dtype=torch.bfloat16) -> torch.Tensor:
    x = F.layer_norm(h.float(), (DIT.hidden,), eps=DIT.eps).to(dtype) * cond.one_plus_scale_out.to(dtype)
    return x @ ckpt.get("proj_out.weight", dtype).t()
