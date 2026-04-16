# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure-PyTorch reference implementation of the Depth Anything V3 metric branch
(DinoV2-Large backbone + DPT head). Weights are loaded directly from the
HuggingFace safetensors checkpoint of DA3-NESTED-GIANT-LARGE-1.1, restricted to
the `model.da3_metric.*` keys."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


HF_SNAPSHOT = (
    "/home/ttuser/.cache/huggingface/hub/"
    "models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/"
    "b2359bdf726fb44ef62acca04d629dcf158053e7/model.safetensors"
)

DEFAULT_INPUT_SIZE = 518
PATCH_SIZE = 14
EMBED_DIM = 1024
NUM_HEADS = 16
NUM_LAYERS = 24
MLP_RATIO = 4
OUT_LAYERS = (4, 11, 17, 23)
HEAD_FEATURES = 256
HEAD_OUT_CHANNELS = (256, 512, 1024, 1024)


class _LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # qkv.weight already has the q-rows pre-multiplied by 1/sqrt(head_dim)
        # via `_fold_attention_scale_into_state_dict`, so q is implicitly scaled.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, dim * mlp_ratio)
        self.ls2 = _LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoV2Large(nn.Module):
    """DinoV2-Large backbone with intermediate-layer extraction (no register tokens)."""

    def __init__(self, img_size: int = DEFAULT_INPUT_SIZE) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = PATCH_SIZE
        self.num_patches_side = img_size // PATCH_SIZE
        self.num_patches = self.num_patches_side**2
        self.patch_embed = nn.ModuleDict(
            {
                "proj": nn.Conv2d(
                    3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=True
                ),
            }
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, EMBED_DIM))
        self.blocks = nn.ModuleList(
            [_Block(EMBED_DIM, NUM_HEADS, MLP_RATIO) for _ in range(NUM_LAYERS)]
        )
        self.norm = nn.LayerNorm(EMBED_DIM, eps=1e-6)

    def _interpolate_pos_embed(self, H: int, W: int) -> torch.Tensor:
        if H * W == self.num_patches:
            return self.pos_embed
        pos = self.pos_embed
        cls_pe, patch_pe = pos[:, :1], pos[:, 1:]
        side = int(math.sqrt(patch_pe.shape[1]))
        patch_pe = patch_pe.reshape(1, side, side, EMBED_DIM).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(patch_pe, size=(H, W), mode="bicubic", align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, H * W, EMBED_DIM)
        return torch.cat([cls_pe, patch_pe], dim=1)

    def forward(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        B, _, H, W = pixel_values.shape
        Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE
        x = self.patch_embed["proj"](pixel_values)  # B, C, Hp, Wp
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._interpolate_pos_embed(Hp, Wp)

        outs: List[torch.Tensor] = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in OUT_LAYERS:
                outs.append(x)
        # No final norm on intermediates (DPT consumes raw block outputs)
        return outs


class _ResConvUnit(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return x + out


class _FeatureFusion(nn.Module):
    """DPT FeatureFusionBlock. The deepest stage (refinenet4) receives only
    one input, so its `resConfUnit1` weights are absent; we make it optional."""

    def __init__(self, features: int, has_unit1: bool) -> None:
        super().__init__()
        if has_unit1:
            self.resConfUnit1 = _ResConvUnit(features)
        self.resConfUnit2 = _ResConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            x = x + self.resConfUnit1(skip)
        x = self.resConfUnit2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class DPTHead(nn.Module):
    """DPT depth head. Takes 4 ViT intermediate features [B, 1+N, C] and produces
    a single-channel depth map at full resolution."""

    def __init__(
        self,
        in_dim: int = EMBED_DIM,
        features: int = HEAD_FEATURES,
        out_channels=HEAD_OUT_CHANNELS,
    ) -> None:
        super().__init__()
        self.features = features
        self.out_channels = out_channels
        # 1x1 reassemble projects to per-stage channel widths
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_dim, c, kernel_size=1, bias=True) for c in out_channels]
        )
        # Resize layers: 4x upsample, 2x upsample, identity, 1x downsample (as 3x3 stride-2 conv)
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        # scratch
        self.scratch = nn.Module()
        self.scratch.layer1_rn = nn.Conv2d(out_channels[0], features, kernel_size=3, padding=1, bias=False)
        self.scratch.layer2_rn = nn.Conv2d(out_channels[1], features, kernel_size=3, padding=1, bias=False)
        self.scratch.layer3_rn = nn.Conv2d(out_channels[2], features, kernel_size=3, padding=1, bias=False)
        self.scratch.layer4_rn = nn.Conv2d(out_channels[3], features, kernel_size=3, padding=1, bias=False)
        self.scratch.refinenet1 = _FeatureFusion(features, has_unit1=True)
        self.scratch.refinenet2 = _FeatureFusion(features, has_unit1=True)
        self.scratch.refinenet3 = _FeatureFusion(features, has_unit1=True)
        self.scratch.refinenet4 = _FeatureFusion(features, has_unit1=False)
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, padding=1, bias=True)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
        )
        self.scratch.sky_output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
        )

    def forward(self, intermediates: List[torch.Tensor], img_hw):
        H, W = img_hw
        Hp, Wp = H // PATCH_SIZE, W // PATCH_SIZE

        feats = []
        for i, x in enumerate(intermediates):
            # drop cls token, B,N,C -> B,C,H,W
            x = x[:, 1:, :].transpose(1, 2).reshape(x.shape[0], -1, Hp, Wp)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            feats.append(x)

        # bottom-up DPT fusion
        layer_rns = [self.scratch.layer1_rn, self.scratch.layer2_rn, self.scratch.layer3_rn, self.scratch.layer4_rn]
        rn = [layer_rns[i](feats[i]) for i in range(4)]

        # refinenet4 has no skip
        path = self.scratch.refinenet4(rn[3])
        path = self.scratch.refinenet3(path, rn[2])
        path = self.scratch.refinenet2(path, rn[1])
        path = self.scratch.refinenet1(path, rn[0])

        path = self.scratch.output_conv1(path)
        path = F.interpolate(path, size=(H, W), mode="bilinear", align_corners=True)
        depth = self.scratch.output_conv2(path)
        return depth


class DA3Metric(nn.Module):
    """End-to-end DA3 metric model: DinoV2-L backbone + DPT head."""

    def __init__(self, img_size: int = DEFAULT_INPUT_SIZE) -> None:
        super().__init__()
        self.backbone = DinoV2Large(img_size=img_size)
        self.head = DPTHead()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        intermediates = self.backbone(pixel_values)
        depth = self.head(intermediates, img_hw=pixel_values.shape[-2:])
        return depth


def _fold_attention_scale_into_state_dict(remapped: dict) -> dict:
    """Pre-multiply the q-rows of every block's qkv weight & bias by
    1/sqrt(head_dim) so the runtime forward can drop the explicit scale op."""
    out = dict(remapped)
    head_dim = EMBED_DIM // NUM_HEADS
    scale = head_dim**-0.5
    block_indices = sorted(
        {int(k.split(".")[2]) for k in remapped if k.startswith("backbone.blocks.")}
    )
    for i in block_indices:
        wkey = f"backbone.blocks.{i}.attn.qkv.weight"
        bkey = f"backbone.blocks.{i}.attn.qkv.bias"
        # qkv.weight has shape (3*EMBED_DIM, EMBED_DIM); first EMBED_DIM rows are Q.
        w = out[wkey].clone()
        b = out[bkey].clone()
        w[:EMBED_DIM] = w[:EMBED_DIM] * scale
        b[:EMBED_DIM] = b[:EMBED_DIM] * scale
        out[wkey] = w
        out[bkey] = b
    return out


def load_da3_metric_state_dict(safetensors_path: str | Path = HF_SNAPSHOT) -> dict:
    """Read only the `model.da3_metric.*` slice of the checkpoint and remap keys
    onto the DA3Metric module above. Pre-folds attention scale into qkv."""
    from safetensors import safe_open

    state = {}
    prefix = "model.da3_metric."
    with safe_open(str(safetensors_path), framework="pt") as f:
        for k in f.keys():
            if not k.startswith(prefix):
                continue
            state[k[len(prefix):]] = f.get_tensor(k)

    remapped = {}
    for k, v in state.items():
        if k.startswith("backbone.pretrained."):
            remapped["backbone." + k[len("backbone.pretrained."):]] = v
        else:
            remapped[k] = v
    return _fold_attention_scale_into_state_dict(remapped)


def build_da3_metric(load_weights: bool = True, img_size: int = DEFAULT_INPUT_SIZE) -> DA3Metric:
    model = DA3Metric(img_size=img_size).eval()
    if load_weights:
        sd = load_da3_metric_state_dict()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # Fail loudly if backbone weights are missing — that would silently degrade accuracy.
        critical_missing = [k for k in missing if "sky_output_conv2" not in k]
        assert not critical_missing, f"missing keys: {critical_missing[:8]}"
    return model
