# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of DaViT's PreNorm + WindowAttention block.

WindowAttention on the upstream side:

    1. View x: [B, T, C] -> [B, H, W, C], pad H/W to multiples of window
    2. window_partition: 6D reshape+permute -> [num_w * B, ws, ws, C]
    3. Flatten windows -> [num_w * B, ws*ws, C]
    4. Standard multi-head self-attention inside each window
    5. window_reverse: 6D reshape+permute -> [B, H_padded, W_padded, C]
    6. Unpad to [B, H, W, C] -> view [B, T, C]

The window partition / reverse uses 6D ops which we keep on torch. The
matmul-heavy interior (LayerNorm + qkv linear + attention + proj linear)
moves to the device. PreNorm's residual add stays on torch (cheap, and
the post-attention output is already a torch tensor at that point).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _bf16_tile(ttnn_mod, t: torch.Tensor, device):
    return ttnn_mod.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn_mod.bfloat16, layout=ttnn_mod.TILE_LAYOUT, device=device,
    )


def _bfp8_tile(ttnn_mod, t: torch.Tensor, device):
    return ttnn_mod.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn_mod.bfloat8_b, layout=ttnn_mod.TILE_LAYOUT, device=device,
    )


class TTNNDaViTPreNormWindowAttn(nn.Module):
    """`PreNorm(LN, WindowAttention)` with the LN+attention+proj on chip."""

    def __init__(self, prenorm_torch: nn.Module, device) -> None:
        super().__init__()
        import ttnn

        self._ttnn = ttnn
        self.device = device

        norm = prenorm_torch.norm
        attn = prenorm_torch.fn  # WindowAttention
        self.window_size = int(attn.window_size)
        self.num_heads = int(attn.num_heads)
        self.dim = int(attn.qkv.in_features)
        self.head_dim = self.dim // self.num_heads
        self.head_dim_inv_sqrt = float(self.head_dim) ** -0.5

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

    # --- on-device interior: attention over windows -----------------------

    def _on_chip_attn(self, windows_torch: torch.Tensor) -> torch.Tensor:
        """`windows_torch` is [BW, ws*ws, C]. Returns same shape."""
        ttnn = self._ttnn
        BW, T, C = windows_torch.shape

        x_tt = _bf16_tile(ttnn, windows_torch, self.device)
        h = ttnn.layer_norm(x_tt, weight=self.ln_w, bias=self.ln_b)
        ttnn.deallocate(x_tt)
        qkv = ttnn.linear(h, self.qkv_w, bias=self.qkv_b)
        ttnn.deallocate(h)

        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.num_heads
        )
        ttnn.deallocate(qkv)
        # k pre-transposed to [BW, H, head_dim, T]
        scores = ttnn.matmul(q, k)  # [BW, H, T, T]
        ttnn.deallocate(q); ttnn.deallocate(k)
        scores = ttnn.multiply(scores, self.head_dim_inv_sqrt)
        probs = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)
        attn_out = ttnn.matmul(probs, v)
        ttnn.deallocate(probs); ttnn.deallocate(v)
        attn_out = ttnn.transformer.concatenate_heads(attn_out)  # [BW, T, C]

        out = ttnn.linear(attn_out, self.proj_w, bias=self.proj_b)
        ttnn.deallocate(attn_out)
        result = ttnn.to_torch(out).to(windows_torch.dtype)
        ttnn.deallocate(out)
        return result

    # --- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor, size):
        """`x`: [B, T, C], T = H*W. Returns (x_with_residual, size)."""
        H, W = size
        B, T, C = x.shape
        assert T == H * W
        ws = self.window_size

        residual = x

        # Pad to multiples of window_size
        x4 = x.view(B, H, W, C)
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x4p = F.pad(x4, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x4p.shape[1], x4p.shape[2]

        # window_partition: [B, Hp/ws, ws, Wp/ws, ws, C] -> [B*nh*nw, ws, ws, C]
        nh, nw = Hp // ws, Wp // ws
        x6 = x4p.view(B, nh, ws, nw, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = x6.view(-1, ws * ws, C)  # [B*nh*nw, ws*ws, C]

        # On-chip self-attention over windows
        attn = self._on_chip_attn(windows)

        # window_reverse: [B*nh*nw, ws*ws, C] -> [B, Hp, Wp, C]
        attn = attn.view(B, nh, nw, ws, ws, C)
        attn = attn.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        # Unpad
        if pad_r or pad_b:
            attn = attn[:, :H, :W, :].contiguous()
        attn = attn.view(B, T, C)

        return residual + attn, size


def swap_davit_window_attns(vision_tower: nn.Module, device) -> int:
    swapped = 0
    for stage_seq in vision_tower.blocks:
        for dual_block in stage_seq:
            sb = getattr(dual_block, "spatial_block", None)
            if sb is None or not hasattr(sb, "window_attn"):
                continue
            sb.window_attn = TTNNDaViTPreNormWindowAttn(sb.window_attn, device).to(torch.bfloat16)
            swapped += 1
    return swapped
