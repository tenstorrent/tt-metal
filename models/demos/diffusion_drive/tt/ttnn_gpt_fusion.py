# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN GPT cross-modal fusion — Stage 3.6.

Ports ``TransfuserBackbone._fuse_features`` (run 4× between ResNet stages) onto
TTNN.  Each call:

  1. ``avgpool_img`` / ``avgpool_lidar``  AdaptiveAvgPool2d → ttnn.avg_pool2d
  2. ``lidar_channel_to_img[i]``          1×1 Conv2d        → ttnn.conv2d
  3. ``transformers[i]`` (GPT)            LN + SDPA + MLP   → ttnn.* (TtnnGPT)
  4. ``img_channel_to_lidar[i]``          1×1 Conv2d        → ttnn.conv2d
  5. ``F.interpolate(bilinear)`` residual upsample          → ttnn.upsample
  6. residual add                                           → ttnn.add

Why this was previously marked "blocked": ``AdaptiveAvgPool2d`` (variable→fixed)
and bilinear ``F.interpolate`` need *integer* pool/upsample ratios, and
``ttnn.upsample`` bilinear only supports integer scales.  At the **production
eval resolution** (camera 256×1024, LiDAR 256×256) every ratio *is* integer —
image features 64×256→32×128→16×64→8×32 pool to 8×32; LiDAR 64²→32²→16²→8² pool
to 8² — so the fusion is fully portable there.  At reduced test resolutions the
ratios are not integer; this module asserts divisibility and is therefore only
valid at production resolution (which is what the PDM eval uses).

All math runs in bfloat16; PCC ≥ 0.99 vs the PyTorch reference.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

import ttnn
from models.demos.diffusion_drive.tt.ttnn_perception import _prep_layernorm, _prep_linear


def _ln(x, n):
    return ttnn.layer_norm(x, weight=n[0], bias=n[1], epsilon=n[2])


# ---------------------------------------------------------------------------
# GPT transformer (image + LiDAR tokens) on device
# ---------------------------------------------------------------------------


class _TtnnGPTSelfAttn:
    """SelfAttention drop-in (separate q/k/v/proj Linears, SDPA scale 1/sqrt(hd))."""

    def __init__(self, sa, device) -> None:
        self._nh = sa.n_head
        self._wq, self._bq = _prep_linear(sa.query, device)
        self._wk, self._bk = _prep_linear(sa.key, device)
        self._wv, self._bv = _prep_linear(sa.value, device)
        self._wo, self._bo = _prep_linear(sa.proj, device)

    def _heads(self, t, B, T, C):
        hd = C // self._nh
        t = ttnn.reshape(t, (B, T, self._nh, hd))
        return ttnn.permute(t, (0, 2, 1, 3))  # (B, nh, T, hd)

    def __call__(self, x, B, T, C):
        hd = C // self._nh
        q = self._heads(ttnn.linear(x, self._wq, bias=self._bq), B, T, C)  # (B,nh,T,hd)
        k = self._heads(ttnn.linear(x, self._wk, bias=self._bk), B, T, C)
        v = self._heads(ttnn.linear(x, self._wv, bias=self._bv), B, T, C)
        # Manual attention (head_dim is 16 at the first GPT scale, below SDPA's
        # 32-wide minimum, so q@kᵀ·softmax·v is done explicitly).
        att = ttnn.matmul(q, ttnn.permute(k, (0, 1, 3, 2)))  # (B,nh,T,T)
        att = ttnn.multiply(att, 1.0 / math.sqrt(hd))
        att = ttnn.softmax(att, dim=-1)
        o = ttnn.matmul(att, v)  # (B,nh,T,hd)
        o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), (B, T, C))
        return ttnn.linear(o, self._wo, bias=self._bo)


class _TtnnGPTBlock:
    """GPTBlock drop-in: x = x + attn(ln1(x)); x = x + mlp(ln2(x))."""

    def __init__(self, blk, device) -> None:
        self._ln1 = _prep_layernorm(blk.ln1, device)
        self._ln2 = _prep_layernorm(blk.ln2, device)
        self._attn = _TtnnGPTSelfAttn(blk.attn, device)
        self._fc1w, self._fc1b = _prep_linear(blk.mlp[0], device)  # Linear(C, 4C)
        self._fc2w, self._fc2b = _prep_linear(blk.mlp[2], device)  # Linear(4C, C)

    def __call__(self, x, B, T, C):
        x = ttnn.add(x, self._attn(_ln(x, self._ln1), B, T, C))
        h = ttnn.relu(ttnn.linear(_ln(x, self._ln2), self._fc1w, bias=self._fc1b))
        h = ttnn.linear(h, self._fc2w, bias=self._fc2b)
        return ttnn.add(x, h)


class TtnnGPT:
    """GPT cross-modal fusion transformer (pos_emb + blocks + ln_f) on device.

    Operates on a packed token tensor (B, N, C) in TILE layout and returns the
    same; tokenization/splitting is handled by the caller (TtnnFuseFeatures).
    """

    def __init__(self, gpt, device) -> None:
        self._device = device
        self._blocks: List[_TtnnGPTBlock] = [_TtnnGPTBlock(b, device) for b in gpt.blocks]
        self._lnf = _prep_layernorm(gpt.ln_f, device)
        self._pos = ttnn.from_torch(
            gpt.pos_emb.detach().to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
        )  # (1, N, C)

    def __call__(self, tokens, B, N, C):
        x = ttnn.add(tokens, self._pos)  # broadcast batch
        for blk in self._blocks:
            x = blk(x, B, N, C)
        return _ln(x, self._lnf)


# ---------------------------------------------------------------------------
# Full _fuse_features port
# ---------------------------------------------------------------------------


class TtnnFuseFeatures(nn.Module):
    """TTNN drop-in for ``TransfuserBackbone._fuse_features`` (torch in → torch out).

    Built once per backbone; ``forward(image_features, lidar_features, layer_idx)``
    matches the reference signature so ``TtnnTransfuserBackbone.__call__`` can call
    it in place of ``ref._fuse_features``.
    """

    def __init__(self, ref_backbone, device) -> None:
        super().__init__()
        self._d = device
        cfg = ref_backbone.config
        self._iv, self._ih = cfg.img_vert_anchors, cfg.img_horz_anchors
        self._lv, self._lh = cfg.lidar_vert_anchors, cfg.lidar_horz_anchors
        self._n_img = self._iv * self._ih

        self._gpt = [TtnnGPT(ref_backbone.transformers[i], device) for i in range(4)]
        # 1×1 channel-projection convs == channel-wise Linear.  A 1×1 stride-1
        # conv is exactly a per-pixel matmul over channels; expressing it as
        # ttnn.linear avoids conv2d's tile-shard constraints on the tiny 8×8
        # LiDAR pool (where conv2d raises a shard-alignment fatal).
        self._l2i = [self._prep_1x1(ref_backbone.lidar_channel_to_img[i], device) for i in range(4)]
        self._i2l = [self._prep_1x1(ref_backbone.img_channel_to_lidar[i], device) for i in range(4)]

    @staticmethod
    def _prep_1x1(conv: nn.Conv2d, device):
        """1×1 Conv2d → (weight (Cin,Cout), bias (1,Cout)) TILE tensors for ttnn.linear."""
        cout, cin = conv.out_channels, conv.in_channels
        w = conv.weight.detach().reshape(cout, cin).t().contiguous().to(torch.bfloat16)  # (Cin, Cout)
        b = conv.bias.detach().reshape(1, cout).to(torch.bfloat16)
        return (
            ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=device),
            ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device),
        )

    # --- layout helpers -------------------------------------------------
    def _to_dev_rm(self, x: torch.Tensor):
        """torch (B,C,H,W) → ttnn ROW_MAJOR (1,1,B*H*W,C)."""
        B, C, H, W = x.shape
        nhwc = x.permute(0, 2, 3, 1).contiguous().reshape(1, 1, B * H * W, C).to(torch.bfloat16)
        return ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self._d), (B, C, H, W)

    def _avg_pool(self, x_rm, B, H, W, C, vert, horz):
        assert H % vert == 0 and W % horz == 0, (
            f"TtnnFuseFeatures needs integer pool ratio (got {H}×{W} → {vert}×{horz}); "
            "Stage-3.6 fusion is only valid at production resolution (256×1024 / 256×256)."
        )
        kh, kw = H // vert, W // horz
        out = ttnn.avg_pool2d(
            x_rm,
            batch_size=B,
            input_h=H,
            input_w=W,
            channels=C,
            kernel_size=[kh, kw],
            stride=[kh, kw],
            padding=[0, 0],
        )  # (1,1,B*vert*horz,C) — may be height-sharded
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        return out

    def _lin1x1(self, x, proj):
        """Channel-wise 1×1 projection via ttnn.linear.  x: (1,1,N,Cin) → (1,1,N,Cout)."""
        return ttnn.linear(self._clean_tile(x), proj[0], bias=proj[1])

    @staticmethod
    def _clean_tile(x):
        """Sanitize to interleaved DRAM + TILE (upsample/avg_pool auto-shard their outputs)."""
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    # --- forward --------------------------------------------------------
    def forward(self, image_features: torch.Tensor, lidar_features: torch.Tensor, layer_idx: int):
        img_rm, (B, Ci, Hi, Wi) = self._to_dev_rm(image_features)
        lid_rm, (_, Cl, Hl, Wl) = self._to_dev_rm(lidar_features)
        iv, ih, lv, lh = self._iv, self._ih, self._lv, self._lh

        # 1. adaptive avg-pool both branches to the anchor grids
        img_embd = self._avg_pool(img_rm, B, Hi, Wi, Ci, iv, ih)  # (1,1,B*iv*ih,Ci)
        lid_embd = self._avg_pool(lid_rm, B, Hl, Wl, Cl, lv, lh)  # (1,1,B*lv*lh,Cl)

        # 2. lidar_channel_to_img: project pooled LiDAR Cl→Ci
        lid_proj = self._lin1x1(lid_embd, self._l2i[layer_idx])  # (1,1,B*lv*lh,Ci) TILE

        # 3. GPT: tokenize (img then lidar), fuse, split
        img_tok = ttnn.to_layout(ttnn.reshape(img_embd, (B, iv * ih, Ci)), ttnn.TILE_LAYOUT)
        lid_tok = ttnn.reshape(lid_proj, (B, lv * lh, Ci))
        tokens = ttnn.concat([img_tok, lid_tok], dim=1)  # (B, N, Ci)
        N = iv * ih + lv * lh
        x = self._gpt[layer_idx](tokens, B, N, Ci)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        img_x = ttnn.slice(x, [0, 0, 0], [B, self._n_img, Ci])  # (B, iv*ih, Ci)
        lid_x = ttnn.slice(x, [0, self._n_img, 0], [B, N, Ci])  # (B, lv*lh, Ci)

        # 4. img_channel_to_lidar: project LiDAR tokens Ci→Cl
        lid_x = ttnn.reshape(lid_x, (1, 1, B * lv * lh, Ci))
        lid_out = self._lin1x1(lid_x, self._i2l[layer_idx])  # (1,1,B*lv*lh,Cl) TILE
        lid_out = ttnn.to_layout(ttnn.reshape(lid_out, (B, lv, lh, Cl)), ttnn.ROW_MAJOR_LAYOUT)

        # 5. bilinear upsample both back to feature resolution
        img_out = ttnn.reshape(img_x, (B, iv, ih, Ci))
        img_up = ttnn.upsample(img_out, [Hi // iv, Wi // ih], mode="bilinear")  # (B,Hi,Wi,Ci)
        lid_up = ttnn.upsample(lid_out, [Hl // lv, Wl // lh], mode="bilinear")  # (B,Hl,Wl,Cl)

        # 6. residual add (in NHWC), back to torch (B,C,H,W)
        img_res = ttnn.add(self._clean_tile(ttnn.reshape(img_rm, (B, Hi, Wi, Ci))), self._clean_tile(img_up))
        lid_res = ttnn.add(self._clean_tile(ttnn.reshape(lid_rm, (B, Hl, Wl, Cl))), self._clean_tile(lid_up))
        img_t = ttnn.to_torch(img_res).reshape(B, Hi, Wi, Ci).permute(0, 3, 1, 2).float()
        lid_t = ttnn.to_torch(lid_res).reshape(B, Hl, Wl, Cl).permute(0, 3, 1, 2).float()
        return img_t, lid_t
