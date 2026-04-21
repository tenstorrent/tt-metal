# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of DaViT's PreNorm + ChannelAttention block, on-device only.

ChannelAttention computes attention over CHANNELS rather than tokens:

    qkv = Linear(x)                                    # [B, T, 3C]
    q, k, v reshape to [B, groups, T, C/groups]
    q *= 1/sqrt(T)
    scores = q.T @ k                                   # [B, g, C/g, C/g]
    probs  = softmax(scores)
    out    = (probs @ v.T).T                           # [B, g, T, C/g]
    out reshape to [B, T, C]
    out = Linear_proj(out)

This implementation keeps everything on chip — no intermediate
torch round-trips. The 5D split-and-permute is decomposed into
4D ttnn ops: ttnn.split along the channel axis, ttnn.reshape to
introduce the groups dim, and ttnn.permute(0,2,1,3) to make
groups precede tokens.
"""

from __future__ import annotations

import torch
from torch import nn


def _bf16_tile(ttnn_mod, t: torch.Tensor, device):
    return ttnn_mod.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn_mod.bfloat16, layout=ttnn_mod.TILE_LAYOUT, device=device,
    )


class TTNNDaViTPreNormChannelAttn(nn.Module):
    def __init__(self, prenorm_torch: nn.Module, device) -> None:
        super().__init__()
        import ttnn

        self._ttnn = ttnn
        self.device = device

        norm = prenorm_torch.norm
        attn = prenorm_torch.fn
        self.groups = int(attn.groups)
        self.dim = int(attn.qkv.in_features)
        assert self.dim % self.groups == 0
        self.ch_per_group = self.dim // self.groups

        self.ln_w = _bf16_tile(ttnn, norm.weight.detach(), device)
        self.ln_b = _bf16_tile(ttnn, norm.bias.detach(), device)
        self.qkv_w = _bf16_tile(ttnn, attn.qkv.weight.detach().t().contiguous(), device)
        self.qkv_b = _bf16_tile(
            ttnn,
            (attn.qkv.bias.detach() if attn.qkv.bias is not None
             else torch.zeros(3 * self.dim)),
            device,
        )
        self.proj_w = _bf16_tile(ttnn, attn.proj.weight.detach().t().contiguous(), device)
        self.proj_b = _bf16_tile(
            ttnn,
            (attn.proj.bias.detach() if attn.proj.bias is not None
             else torch.zeros(self.dim)),
            device,
        )

    def forward(self, x: torch.Tensor, size):
        ttnn = self._ttnn
        residual_dtype = x.dtype
        B, T, C = x.shape
        g = self.groups
        cpg = self.ch_per_group
        scale = float(T) ** -0.5

        x_tt = _bf16_tile(ttnn, x, self.device)
        residual = x_tt

        # PreNorm + qkv linear
        h = ttnn.layer_norm(x_tt, weight=self.ln_w, bias=self.ln_b)
        qkv = ttnn.linear(h, self.qkv_w, bias=self.qkv_b)  # [B, T, 3C]
        ttnn.deallocate(h)

        # Split into Q, K, V each [B, T, C] along channel dim. ttnn.split's
        # second arg is chunk SIZE (like torch.split), not num chunks.
        q, k, v = ttnn.split(qkv, C, dim=-1)
        ttnn.deallocate(qkv)

        # Reshape [B, T, C] -> [B, T, g, cpg], then permute to [B, g, T, cpg]
        def _to_groups(t):
            t = ttnn.reshape(t, (B, T, g, cpg))
            return ttnn.permute(t, (0, 2, 1, 3))

        q = _to_groups(q)
        k = _to_groups(k)
        v = _to_groups(v)

        # scores = q.T @ k  ->  [B, g, cpg, cpg]
        q_T = ttnn.transpose(q, -2, -1)        # [B, g, cpg, T]
        ttnn.deallocate(q)
        scores = ttnn.matmul(q_T, k)
        ttnn.deallocate(q_T); ttnn.deallocate(k)
        scores = ttnn.multiply(scores, scale)
        probs = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)

        # (probs @ v.T).T  ->  [B, g, T, cpg]
        v_T = ttnn.transpose(v, -2, -1)        # [B, g, cpg, T]
        ttnn.deallocate(v)
        attn_out = ttnn.matmul(probs, v_T)     # [B, g, cpg, T]
        ttnn.deallocate(probs); ttnn.deallocate(v_T)
        attn_out = ttnn.transpose(attn_out, -2, -1)  # [B, g, T, cpg]

        # Merge groups back to channels: [B, g, T, cpg] -> [B, T, g, cpg] -> [B, T, C]
        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))
        attn_out = ttnn.reshape(attn_out, (B, T, C))

        # proj linear + residual
        out = ttnn.linear(attn_out, self.proj_w, bias=self.proj_b)
        ttnn.deallocate(attn_out)
        x_tt = ttnn.add(residual, out)
        ttnn.deallocate(out)

        out_torch = ttnn.to_torch(x_tt).to(residual_dtype)
        ttnn.deallocate(x_tt)
        return out_torch, size


def swap_davit_channel_attns(vision_tower: nn.Module, device) -> int:
    swapped = 0
    for stage_seq in vision_tower.blocks:
        for dual_block in stage_seq:
            cb = getattr(dual_block, "channel_block", None)
            if cb is None or not hasattr(cb, "channel_attn"):
                continue
            cb.channel_attn = TTNNDaViTPreNormChannelAttn(cb.channel_attn, device).to(torch.bfloat16)
            swapped += 1
    return swapped
