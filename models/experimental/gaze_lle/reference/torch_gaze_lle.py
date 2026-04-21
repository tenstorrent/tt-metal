# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Self-contained PyTorch reference for Gaze-LLE.

Adapted from https://github.com/fkryan/gazelle without external deps on timm or
torch.hub. Matches the ViT-B/14 (default) and ViT-L/14 DINOv2 backbones and the
head architecture described in the paper.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DinoV2Config:
    img_size: int = 448
    patch_size: int = 14
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4


# `dinov2_vitb14` / `dinov2_vitl14` from facebookresearch/dinov2 torch.hub have NO
# register tokens. The `*_reg` variants have 4. Gaze-LLE ships against the
# non-register variants, so we default to 0 here to match pretrained weights.
DINOV2_CONFIGS = {
    "vitb14": DinoV2Config(embed_dim=768, depth=12, num_heads=12, num_register_tokens=0),
    "vitl14": DinoV2Config(embed_dim=1024, depth=24, num_heads=16, num_register_tokens=0),
}


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1.0):
        super().__init__()
        # ``gamma`` matches the official DINOv2 checkpoint key (`blocks.*.ls{1,2}.gamma`).
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class DinoV2Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoV2Backbone(nn.Module):
    """DINOv2 ViT backbone that returns patch tokens as a feature map."""

    def __init__(self, variant: str = "vitb14", img_size: int = 448):
        super().__init__()
        cfg = DINOV2_CONFIGS[variant]
        self.cfg = cfg
        self.img_size = img_size
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.embed_dim

        num_h = img_size // cfg.patch_size
        self.num_patches = num_h * num_h

        self.patch_embed_proj = nn.Conv2d(3, cfg.embed_dim, cfg.patch_size, cfg.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, cfg.num_register_tokens, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, cfg.embed_dim))
        self.blocks = nn.ModuleList(
            [DinoV2Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.reg_token, std=0.02)

    def get_out_size(self) -> Tuple[int, int]:
        return (self.img_size // self.patch_size, self.img_size // self.patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b, _, h, w = pixel_values.shape
        patches = self.patch_embed_proj(pixel_values)
        patches = patches.flatten(2).transpose(1, 2)  # b n c
        cls = self.cls_token.expand(b, -1, -1)
        reg = self.reg_token.expand(b, -1, -1)
        x = torch.cat([cls, patches], dim=1) + self.pos_embed
        x = torch.cat([x[:, :1], reg, x[:, 1:]], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        patch_tokens = x[:, 1 + self.cfg.num_register_tokens :]
        # reshape to feature map b c out_h out_w
        out_h, out_w = self.get_out_size()
        return patch_tokens.view(b, out_h, out_w, -1).permute(0, 3, 1, 2).contiguous()


def positionalencoding2d(d_model: int, height: int, width: int) -> torch.Tensor:
    if d_model % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4 (got {d_model})")
    pe = torch.zeros(d_model, height, width)
    d = d_model // 2
    div_term = torch.exp(torch.arange(0.0, d, 2) * -(math.log(10000.0) / d))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d + 1 :: 2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe


class GazeBlock(nn.Module):
    """Small transformer block for Gaze-LLE decoder (matches timm Block defaults).

    ``timm.models.vision_transformer.Block`` defaults to ``qkv_bias=False`` — the
    official Gaze-LLE checkpoint accordingly stores no ``transformer.*.attn.qkv.bias``.
    """

    def __init__(self, dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GazeLLE(nn.Module):
    def __init__(
        self,
        backbone: DinoV2Backbone,
        dim: int = 256,
        num_layers: int = 3,
        in_size: Tuple[int, int] = (448, 448),
        out_size: Tuple[int, int] = (64, 64),
        inout: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size()
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        self.linear = nn.Conv2d(backbone.embed_dim, dim, 1)
        self.head_token = nn.Embedding(1, dim)
        self.register_buffer(
            "pos_embed",
            positionalencoding2d(dim, self.featmap_h, self.featmap_w),
        )
        if inout:
            self.inout_token = nn.Embedding(1, dim)
        self.transformer = nn.Sequential(*[GazeBlock(dim=dim, num_heads=8) for _ in range(num_layers)])
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        if inout:
            # Layout matches the official Gaze-LLE checkpoint: index 0=Linear,
            # 1=ReLU, 2=Dropout(0.1), 3=Linear, 4=Sigmoid. Dropout is a no-op at
            # eval time but must be present so pretrained weights load.
            self.inout_head = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

    def _bbox_to_head_map(self, bbox: Sequence[float]) -> torch.Tensor:
        xmin, ymin, xmax, ymax = bbox
        h, w = self.featmap_h, self.featmap_w
        xmin = round(xmin * w)
        ymin = round(ymin * h)
        xmax = round(xmax * w)
        ymax = round(ymax * h)
        head_map = torch.zeros((h, w))
        head_map[ymin:ymax, xmin:xmax] = 1
        return head_map

    def forward(
        self,
        images: torch.Tensor,
        bboxes: List[Sequence[float]],
    ):
        """
        images: (B, 3, H, W) normalized image batch.
        bboxes: list of B normalized [xmin, ymin, xmax, ymax] head bounding boxes.
        Returns dict(heatmap: (B, 64, 64), inout: (B,) or None).
        """
        b = images.shape[0]
        assert len(bboxes) == b, "one bbox per image (simplified single-person API)"

        x = self.backbone(images)
        x = self.linear(x)
        x = x + self.pos_embed.unsqueeze(0)

        head_maps = torch.stack([self._bbox_to_head_map(bb) for bb in bboxes]).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings

        x = x.flatten(start_dim=2).permute(0, 2, 1)
        if self.inout:
            inout_tok = self.inout_token.weight.unsqueeze(0).repeat(b, 1, 1)
            x = torch.cat([inout_tok, x], dim=1)

        x = self.transformer(x)

        inout_preds: Optional[torch.Tensor] = None
        if self.inout:
            inout_preds = self.inout_head(x[:, 0, :]).squeeze(-1)
            x = x[:, 1:, :]

        x = x.reshape(b, self.featmap_h, self.featmap_w, self.dim).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(1)
        x = F.interpolate(x.unsqueeze(1), size=self.out_size, mode="bilinear", align_corners=False).squeeze(1)
        return {"heatmap": x, "inout": inout_preds}


def build_gaze_lle(variant: str = "vitb14", inout: bool = True) -> GazeLLE:
    backbone = DinoV2Backbone(variant=variant, img_size=448)
    return GazeLLE(backbone=backbone, dim=256, num_layers=3, inout=inout)
