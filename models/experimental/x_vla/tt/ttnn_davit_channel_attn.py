# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of DaViT's PreNorm + ChannelAttention block.

ChannelAttention computes attention over CHANNELS rather than tokens:

    qkv = Linear(x)                                    # [B, T, 3C]
    q,k,v reshape to [B, groups, T, C/groups]
    q *= sqrt(1/T)
    scores = q.T @ k                                   # [B, g, C/g, C/g]
    probs  = softmax(scores)
    out    = (probs @ v.T).T                           # [B, g, T, C/g]
    out reshape to [B, T, C]
    out = Linear_proj(out)

The PreNorm wraps it as:  x + ChannelAttention(LayerNorm(x))
(drop_path is identity in eval mode.)
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
    """`PreNorm(LN, ChannelAttention)` evaluated end-to-end on the device."""

    def __init__(self, prenorm_torch: nn.Module, device) -> None:
        super().__init__()
        import ttnn

        self._ttnn = ttnn
        self.device = device

        norm = prenorm_torch.norm
        attn = prenorm_torch.fn  # ChannelAttention
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
        scale = float(T) ** -0.5

        x_tt = _bf16_tile(ttnn, x, self.device)
        residual = x_tt  # alias; ttnn.add will accept and we still hold a reference

        # PreNorm + qkv linear on device
        h = ttnn.layer_norm(x_tt, weight=self.ln_w, bias=self.ln_b)
        qkv = ttnn.linear(h, self.qkv_w, bias=self.qkv_b)  # [B, T, 3C]
        ttnn.deallocate(h)

        # Reshape [B, T, 3C] -> [B, T, 3, groups, ch_per_group]
        # then permute to [3, B, groups, T, ch_per_group]
        # We can express equivalently by:
        #   reshape -> [B*T, 3, groups, cpg]; ttnn supports 4D, so:
        #   reshape  to [1, B, T, 3*C]
        #   then we need [B, groups, T, 3*ch_per_group]: reshape & permute
        # Practical path: do the reshape+permute via to_torch -> torch -> from_torch
        # for now, since ttnn.permute on 5D may not be supported. Cheap because
        # we only round-trip the [B, T, 3C] tensor (small).
        qkv_torch = ttnn.to_torch(qkv).to(torch.bfloat16)
        ttnn.deallocate(qkv)
        # [B, T, 3, g, cpg] -> [3, B, g, T, cpg]
        qkv5 = qkv_torch.reshape(B, T, 3, self.groups, self.ch_per_group).permute(2, 0, 3, 1, 4).contiguous()
        q_t = qkv5[0]   # [B, g, T, cpg]
        k_t = qkv5[1]
        v_t = qkv5[2]
        # Re-upload q, k, v individually
        q_tt = _bf16_tile(ttnn, q_t, self.device)
        k_tt = _bf16_tile(ttnn, k_t, self.device)
        v_tt = _bf16_tile(ttnn, v_t, self.device)

        # q.T @ k -> [B, g, cpg, cpg]; ttnn matmul needs explicit transpose
        # Use transpose(-2, -1) to swap last two dims
        q_T = ttnn.transpose(q_tt, -2, -1)   # [B, g, cpg, T]
        ttnn.deallocate(q_tt)
        scores = ttnn.matmul(q_T, k_tt)      # [B, g, cpg, cpg]
        ttnn.deallocate(q_T); ttnn.deallocate(k_tt)
        scores = ttnn.multiply(scores, scale)
        probs = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)

        # (probs @ v.T).T  ->  [B, g, T, cpg]
        v_T = ttnn.transpose(v_tt, -2, -1)   # [B, g, cpg, T]
        ttnn.deallocate(v_tt)
        attn_out = ttnn.matmul(probs, v_T)   # [B, g, cpg, T]
        ttnn.deallocate(probs); ttnn.deallocate(v_T)
        attn_out = ttnn.transpose(attn_out, -2, -1)  # [B, g, T, cpg]

        # Merge groups back to channels: [B, g, T, cpg] -> [B, T, g*cpg=C]
        # ttnn supports permute on 4D; do it on torch for safety
        attn_torch = ttnn.to_torch(attn_out).to(torch.bfloat16)
        ttnn.deallocate(attn_out)
        merged = attn_torch.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        merged_tt = _bf16_tile(ttnn, merged, self.device)

        # proj linear + residual on device
        out = ttnn.linear(merged_tt, self.proj_w, bias=self.proj_b)
        ttnn.deallocate(merged_tt)
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
