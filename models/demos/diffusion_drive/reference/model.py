# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DiffusionDrive reference model — pure PyTorch port, no nuplan/mmdet/mmcv dependencies.

Architecture (confirmed Stage 0, 2026-04-13):
  camera (B×3×256×1024) + lidar BEV (B×1×256×256)
  -> TransFuser backbone (ResNet-34 × 2, 4-scale GPT fusion)
  -> 3-level top-down FPN (LiDAR branch only)
  -> Perception TransformerDecoder (3 layers, 65 key-val tokens)
  -> TrajectoryHead (DDIM 2-step denoiser, K=20, T=8, output K×T×3)
  -> argmax over K scores -> best trajectory (T×3)

Upstream reference: https://github.com/hustvl/DiffusionDrive
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler

# ---------------------------------------------------------------------------
# Thin replacements for nuplan types (inference-only, no training deps)
# ---------------------------------------------------------------------------


class BoundingBox2DIndex(IntEnum):
    """2-D bounding box field indices (x, y, heading, length, width)."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls) -> int:
        return 5

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def POINT(cls):
        return slice(cls._X, cls._Y + 1)


HEADING_IDX = 2  # replaces StateSE2Index.HEADING


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DiffusionDriveConfig:
    """Minimal config for inference (no training-only fields)."""

    # --- backbone / perception -------------------------------------------------
    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    latent: bool = False  # set True to replace LiDAR with learned latent

    # camera input
    camera_width: int = 1024
    camera_height: int = 256

    # LiDAR BEV grid
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256
    lidar_min_x: float = -32.0
    lidar_max_x: float = 32.0
    lidar_min_y: float = -32.0
    lidar_max_y: float = 32.0
    lidar_split_height: float = 0.2
    lidar_seq_len: int = 1

    # attention pooling anchors (image: 8×32=256, lidar: 8×8=64)
    img_vert_anchors: int = 8
    img_horz_anchors: int = 32
    lidar_vert_anchors: int = 8
    lidar_horz_anchors: int = 8

    # GPT backbone
    n_layer: int = 2
    n_head: int = 4
    block_exp: int = 4
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    gpt_linear_layer_init_mean: float = 0.0
    gpt_linear_layer_init_std: float = 0.02
    gpt_layer_norm_init_weight: float = 1.0

    # BEV FPN
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2
    bev_num_classes: int = 7

    # --- perception Transformer decoder ---------------------------------------
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    num_bounding_boxes: int = 30

    # --- planner head --------------------------------------------------------
    num_poses: int = 8  # T = 8 waypoints (4 s at 0.5 s steps)
    ego_fut_mode: int = 20  # K = 20 anchor modes

    # anchor file (set at load time by prepare_assets.py / config)
    plan_anchor_path: Optional[str] = None

    @property
    def lidar_bev_h(self) -> int:
        return self.lidar_resolution_height // self.bev_down_sample_factor

    @property
    def lidar_bev_w(self) -> int:
        return self.lidar_resolution_width // self.bev_down_sample_factor


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def linear_relu_ln(embed_dims: int, in_loops: int, out_loops: int, input_dims: Optional[int] = None) -> List[nn.Module]:
    """Build repeated (Linear → ReLU)*in_loops → LayerNorm blocks."""
    if input_dims is None:
        input_dims = embed_dims
    layers: List[nn.Module] = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def gen_sineembed_for_position(pos_tensor: torch.Tensor, hidden_dim: int = 256) -> torch.Tensor:
    """Sinusoidal positional embedding for (x, y) trajectory waypoints.

    Args:
        pos_tensor: (..., 2) tensor of (x, y) coordinates
        hidden_dim: half-dim used per axis; output last-dim = 2 * hidden_dim

    Returns:
        (..., 2*hidden_dim) sinusoidal embedding
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


def bias_init_with_prob(prior_prob: float) -> float:
    return float(-np.log((1 - prior_prob) / prior_prob))


# ---------------------------------------------------------------------------
# Backbone helpers (GPT-style cross-modal fusion)
# ---------------------------------------------------------------------------


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal positional embedding for scalar timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float, resid_pdrop: float) -> None:
        super().__init__()
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        nh, hs = self.n_head, c // self.n_head
        k = self.key(x).view(b, t, nh, hs).transpose(1, 2)
        q = self.query(x).view(b, t, nh, hs).transpose(1, 2)
        v = self.value(x).view(b, t, nh, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y))


class GPTBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_exp: int, attn_pdrop: float, resid_pdrop: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-style cross-modal fusion block (image tokens + LiDAR tokens)."""

    def __init__(self, n_embd: int, config: DiffusionDriveConfig, lidar_time_frames: int = 1) -> None:
        super().__init__()
        self.seq_len = 1
        self.lidar_time_frames = lidar_time_frames
        self.config = config

        n_tokens = (
            config.img_vert_anchors * config.img_horz_anchors
            + lidar_time_frames * config.lidar_vert_anchors * config.lidar_horz_anchors
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(
            *[
                GPTBlock(n_embd, config.n_head, config.block_exp, config.attn_pdrop, config.resid_pdrop)
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor: torch.Tensor, lidar_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bz = lidar_tensor.shape[0]
        img_h, img_w = image_tensor.shape[2:4]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]

        img_tok = image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, image_tensor.shape[1])
        lidar_tok = lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, lidar_tensor.shape[1])
        tokens = torch.cat((img_tok, lidar_tok), dim=1)

        x = self.drop(self.pos_emb + tokens)
        x = self.blocks(x)
        x = self.ln_f(x)

        n_img = self.config.img_vert_anchors * self.config.img_horz_anchors
        img_out = x[:, :n_img, :].view(bz, img_h, img_w, -1).permute(0, 3, 1, 2).contiguous()
        lidar_out = x[:, n_img:, :].view(bz, lidar_h, lidar_w, -1).permute(0, 3, 1, 2).contiguous()
        return img_out, lidar_out


# ---------------------------------------------------------------------------
# TransFuser backbone
# ---------------------------------------------------------------------------


class TransfuserBackbone(nn.Module):
    """
    Dual ResNet-34 backbone (image + LiDAR) with 4-scale GPT cross-modal fusion.
    Returns (bev_upscale, bev_feature, None):
      bev_upscale: (B, 64, H//4, W//4) — top-down FPN output (P3)
      bev_feature:  (B, 512, H//32, W//32) — deepest LiDAR feature (8×8 for 256² input)
    """

    def __init__(self, config: DiffusionDriveConfig) -> None:
        super().__init__()
        self.config = config

        self.image_encoder = timm.create_model(config.image_architecture, pretrained=False, features_only=True)
        in_channels = config.lidar_seq_len

        if config.latent:
            self.lidar_latent = nn.Parameter(
                torch.randn(1, in_channels, config.lidar_resolution_width, config.lidar_resolution_height)
            )

        self.lidar_encoder = timm.create_model(
            config.lidar_architecture, pretrained=False, in_chans=in_channels, features_only=True
        )

        self.avgpool_img = nn.AdaptiveAvgPool2d((config.img_vert_anchors, config.img_horz_anchors))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((config.lidar_vert_anchors, config.lidar_horz_anchors))

        # Determine if there is a stem layer (some timm models expose it as a return layer)
        start_index = 0
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1

        self.transformers = nn.ModuleList(
            [
                GPT(n_embd=self.image_encoder.feature_info.info[start_index + i]["num_chs"], config=config)
                for i in range(4)
            ]
        )
        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
                    self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )
        self._start_index = start_index

        # 3-level top-down FPN (on LiDAR branch)
        ch = config.bev_features_channels
        deepest_lidar_chs = self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"]
        self.c5_conv = nn.Conv2d(deepest_lidar_chs, ch, kernel_size=1)
        self.up_conv5 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.up_conv4 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=config.bev_upsample_factor, mode="bilinear", align_corners=False)
        self.upsample2 = nn.Upsample(
            size=(
                config.lidar_resolution_height // config.bev_down_sample_factor,
                config.lidar_resolution_width // config.bev_down_sample_factor,
            ),
            mode="bilinear",
            align_corners=False,
        )

    # ------------------------------------------------------------------
    def _top_down(self, x: torch.Tensor) -> torch.Tensor:
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))
        return p3

    def _forward_layer_block(self, layers, return_layers, features):
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def _fuse_features(self, image_features, lidar_features, layer_idx):
        img_embd = self.avgpool_img(image_features)
        lidar_embd = self.avgpool_lidar(lidar_features)
        lidar_embd = self.lidar_channel_to_img[layer_idx](lidar_embd)

        img_out, lidar_out = self.transformers[layer_idx](img_embd, lidar_embd)
        lidar_out = self.img_channel_to_lidar[layer_idx](lidar_out)

        img_out = F.interpolate(img_out, size=image_features.shape[2:], mode="bilinear", align_corners=False)
        lidar_out = F.interpolate(lidar_out, size=lidar_features.shape[2:], mode="bilinear", align_corners=False)
        return image_features + img_out, lidar_features + lidar_out

    def forward(self, image: torch.Tensor, lidar: torch.Tensor):
        img_feats = image
        lidar_feats = lidar

        if self.config.latent:
            lidar_feats = self.lidar_latent.expand(image.shape[0], -1, -1, -1)

        img_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())

        si = self._start_index
        if si > 0:
            img_feats = self._forward_layer_block(img_layers, self.image_encoder.return_layers, img_feats)
            lidar_feats = self._forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_feats)

        for i in range(4):
            img_feats = self._forward_layer_block(img_layers, self.image_encoder.return_layers, img_feats)
            lidar_feats = self._forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_feats)
            img_feats, lidar_feats = self._fuse_features(img_feats, lidar_feats, i)

        bev_upscale = self._top_down(lidar_feats)
        return bev_upscale, lidar_feats, None


# ---------------------------------------------------------------------------
# Decoder sub-modules
# ---------------------------------------------------------------------------


class GridSampleCrossBEVAttention(nn.Module):
    """
    Deformable cross-BEV attention: samples BEV features at predicted
    trajectory waypoint positions using F.grid_sample.

    At each denoising step the decoder looks up contextual features from the
    BEV map at the K×T waypoint coordinates of the current noisy trajectory,
    then aggregates with learned attention weights.
    """

    def __init__(
        self, embed_dims: int, num_heads: int, num_points: int, config: DiffusionDriveConfig, in_bev_dims: int = 64
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.config = config

        self.attention_weights = nn.Linear(embed_dims, num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)
        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self._init_weight()

    def _init_weight(self) -> None:
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        queries: torch.Tensor,
        traj_points: torch.Tensor,
        bev_feature: torch.Tensor,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            queries:     (B, K, D)
            traj_points: (B, K, T, 2)  — (x, y) in ego-vehicle metres
            bev_feature: (B, C, H, W)  — BEV feature map (upscaled P3)
            spatial_shape: (H, W)
        Returns:
            (B, K, D) updated queries
        """
        bs, num_queries, num_points, _ = traj_points.shape

        # Normalise to [-1, 1] for grid_sample
        norm = traj_points.clone()
        norm[..., 0] = norm[..., 0] / self.config.lidar_max_y
        norm[..., 1] = norm[..., 1] / self.config.lidar_max_x
        norm = norm[..., [1, 0]]  # swap x↔y

        attn_w = self.attention_weights(queries)  # B, K, T
        attn_w = attn_w.view(bs, num_queries, num_points).softmax(-1)  # B, K, T

        value = self.value_proj(bev_feature)  # B, 256, H, W
        grid = norm.view(bs, num_queries, num_points, 2)
        sampled = F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        # sampled: B, 256, K, T
        attn_w = attn_w.unsqueeze(1)  # B, 1, K, T
        out = (attn_w * sampled).sum(dim=-1)  # B, 256, K
        out = out.permute(0, 2, 1).contiguous()  # B, K, 256
        out = self.output_proj(out)
        return self.dropout(out) + queries


class ModulationLayer(nn.Module):
    """FiLM conditioning layer: scales/shifts features by time-step embedding."""

    def __init__(self, embed_dims: int, condition_dims: int) -> None:
        super().__init__()
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims * 2),
        )

    def forward(self, traj_feature: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        scale_shift = self.scale_shift_mlp(time_embed)  # B, 1, 2D
        scale, shift = scale_shift.chunk(2, dim=-1)  # B, 1, D each
        return traj_feature * (1 + scale) + shift


class DiffMotionPlanningRefinementModule(nn.Module):
    """Per-layer regression + classification head inside the diffusion decoder."""

    def __init__(self, embed_dims: int, ego_fut_ts: int, ego_fut_mode: int) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self._init_weight()

    def _init_weight(self) -> None:
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(self, traj_feature: torch.Tensor):
        bs, ego_fut_mode, _ = traj_feature.shape
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)  # B, K
        traj_delta = self.plan_reg_branch(traj_feature)  # B, K, T*3
        plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)  # B, K, T, 3
        return plan_reg, plan_cls


class CustomTransformerDecoderLayer(nn.Module):
    """One layer of the DiffusionDrive planner decoder."""

    def __init__(self, num_poses: int, d_model: int, d_ffn: int, config: DiffusionDriveConfig) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)

        # cross_bev_feature is output of bev_proj (256-ch), NOT bev_upscale (64-ch)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            embed_dims=d_model,
            num_heads=config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.time_modulation = ModulationLayer(d_model, 256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=config.ego_fut_mode,
        )

    def forward(
        self,
        traj_feature,
        noisy_traj_points,
        bev_feature,
        bev_spatial_shape,
        agents_query,
        ego_query,
        time_embed,
        status_encoding,
    ):
        traj_feature = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(
            self.cross_agent_attention(traj_feature, agents_query, agents_query)[0]
        )
        traj_feature = self.norm1(traj_feature)

        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query, ego_query)[0])
        traj_feature = self.norm2(traj_feature)

        traj_feature = self.norm3(self.ffn(traj_feature))
        traj_feature = self.time_modulation(traj_feature, time_embed)

        poses_reg, poses_cls = self.task_decoder(traj_feature)
        poses_reg[..., :2] = poses_reg[..., :2] + noisy_traj_points
        poses_reg[..., HEADING_IDX] = poses_reg[..., HEADING_IDX].tanh() * math.pi
        return poses_reg, poses_cls


class CustomTransformerDecoder(nn.Module):
    """Stack of decoder layers that progressively refines trajectory predictions."""

    def __init__(self, decoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        traj_feature,
        noisy_traj_points,
        bev_feature,
        bev_spatial_shape,
        agents_query,
        ego_query,
        time_embed,
        status_encoding,
    ):
        traj_points = noisy_traj_points
        reg_list, cls_list = [], []
        for mod in self.layers:
            poses_reg, poses_cls = mod(
                traj_feature,
                traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
            )
            reg_list.append(poses_reg)
            cls_list.append(poses_cls)
            traj_points = poses_reg[..., :2].clone().detach()
        return reg_list, cls_list


# ---------------------------------------------------------------------------
# Agent head
# ---------------------------------------------------------------------------


class AgentHead(nn.Module):
    def __init__(self, num_agents: int, d_ffn: int, d_model: int) -> None:
        super().__init__()
        self._mlp_states = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, BoundingBox2DIndex.size()),
        )
        self._mlp_label = nn.Sequential(nn.Linear(d_model, 1))

    def forward(self, agent_queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        states = self._mlp_states(agent_queries)
        states[..., BoundingBox2DIndex.POINT] = states[..., BoundingBox2DIndex.POINT].tanh() * 32
        states[..., BoundingBox2DIndex.HEADING] = states[..., BoundingBox2DIndex.HEADING].tanh() * math.pi
        labels = self._mlp_label(agent_queries).squeeze(-1)
        return {"agent_states": states, "agent_labels": labels}


# ---------------------------------------------------------------------------
# Trajectory / planner head
# ---------------------------------------------------------------------------


class TrajectoryHead(nn.Module):
    """
    Truncated diffusion planner head.

    Inference procedure (step_num=2):
      1. Add truncated noise to K anchor trajectories at t=8.
      2. For each roll_timestep k ∈ [10, 0]:
         a. Embed current noisy trajectory positions sinusoidally.
         b. Encode positions via plan_anchor_encoder MLP.
         c. Embed timestep via time_mlp.
         d. Run 2-layer CustomTransformerDecoder.
         e. DDIM step to obtain less-noisy trajectory.
      3. Return best mode selected by argmax of scores.
    """

    def __init__(
        self, num_poses: int, d_model: int, d_ffn: int, plan_anchor_path: str, config: DiffusionDriveConfig
    ) -> None:
        super().__init__()
        self._num_poses = num_poses
        self.ego_fut_mode = config.ego_fut_mode

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        plan_anchor = np.load(plan_anchor_path)  # (K, T, 2) float32
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )

        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 512),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

    # ------------------------------------------------------------------
    # Coordinate normalisation helpers (match training protocol)
    # ------------------------------------------------------------------

    def _norm_odo(self, traj: torch.Tensor) -> torch.Tensor:
        x = traj[..., 0:1]
        y = traj[..., 1:2]
        h = traj[..., 2:3]
        x = 2 * (x + 1.2) / 56.9 - 1
        y = 2 * (y + 20) / 46 - 1
        h = 2 * (h + 2) / 3.9 - 1
        return torch.cat([x, y, h], dim=-1)

    def _denorm_odo(self, traj: torch.Tensor) -> torch.Tensor:
        x = traj[..., 0:1]
        y = traj[..., 1:2]
        h = traj[..., 2:3]
        x = (x + 1) / 2 * 56.9 - 1.2
        y = (y + 1) / 2 * 46 - 20
        h = (h + 1) / 2 * 3.9 - 2
        return torch.cat([x, y, h], dim=-1)

    # ------------------------------------------------------------------

    def forward(
        self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding
    ) -> Dict[str, torch.Tensor]:
        return self._forward_test(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding)

    def _forward_test(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding):
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device

        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        # Step 1: add truncated noise to anchor trajectories at t=8
        plan_anchor = self.plan_anchor.unsqueeze(0).expand(bs, -1, -1, -1)  # B, K, T, 2
        # Pad with zeros for heading channel to get K×T×3 before norm
        heading_zero = torch.zeros(*plan_anchor.shape[:-1], 1, device=device)
        plan_anchor_3 = torch.cat([plan_anchor, heading_zero], dim=-1)  # B, K, T, 3
        img = self._norm_odo(plan_anchor_3)
        noise = torch.randn(img.shape, device=device)
        trunc_t = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_t)
        ego_fut_mode = img.shape[1]

        # Step 2: iterative denoising
        for k in roll_timesteps:
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self._denorm_odo(x_boxes)  # B, K, T, 3

            # Only x,y fed into grid-sample attention (first 2 channels)
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points[..., :2], hidden_dim=64)  # B, K, T, 128
            traj_pos_embed = traj_pos_embed.flatten(-2)  # B, K, T*128
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)  # B, K, D
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            if not torch.is_tensor(k):
                k = torch.tensor([k], dtype=torch.long, device=device)
            elif k.dim() == 0:
                k = k[None].to(device)
            k = k.expand(bs)
            time_embed = self.time_mlp(k).view(bs, 1, -1)  # B, 1, D

            reg_list, cls_list = self.diff_decoder(
                traj_feature,
                noisy_traj_points[..., :2],
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
            )
            poses_reg = reg_list[-1]  # B, K, T, 3
            poses_cls = cls_list[-1]  # B, K

            x_start = poses_reg[..., :2]
            x_start = self._norm_odo(torch.cat([x_start, torch.zeros_like(x_start[..., :1])], dim=-1))
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k[0],
                sample=img,
            ).prev_sample

        # Best mode
        mode_idx = poses_cls.argmax(dim=-1)  # B
        mode_idx = mode_idx[..., None, None, None].expand(-1, 1, self._num_poses, 3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)  # B, T, 3
        return {"trajectory": best_reg, "scores": poses_cls}


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class DiffusionDriveModel(nn.Module):
    """
    Full DiffusionDrive model for inference.

    Forward inputs:
        features = {
            "camera_feature": Tensor (B, 3, 256, 1024)
            "lidar_feature":  Tensor (B, 1, 256, 256)    — or zeros if config.latent=True
            "status_feature": Tensor (B, 8)              — (driving_cmd[4], vel[2], accel[2])
        }
    Forward output:
        {
            "trajectory": Tensor (B, T=8, 3)  — best (x, y, heading) trajectory
            "scores":     Tensor (B, K=20)    — per-mode logits
        }
    """

    def __init__(self, config: DiffusionDriveConfig) -> None:
        super().__init__()
        assert (
            config.plan_anchor_path is not None
        ), "config.plan_anchor_path must be set (path to kmeans_navsim_traj_20.npy)"
        self._config = config
        self._backbone = TransfuserBackbone(config)

        # 65 key-val tokens: 8×8 BEV (64) + 1 status
        self._keyval_embedding = nn.Embedding(
            config.img_vert_anchors**2 + 1,  # 64 + 1 = 65  (lidar pool is 8×8)
            config.tf_d_model,
        )
        self._query_splits = [1, config.num_bounding_boxes]
        self._query_embedding = nn.Embedding(
            sum(self._query_splits),
            config.tf_d_model,
        )

        # Down-project deepest LiDAR feature (512 ch → d_model)
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        # Status vector: driving_cmd(4) + vel(2) + accel(2) = 8 inputs
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        # BEV semantic head (for completeness; output unused in inference)
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(config.bev_features_channels, config.bev_features_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.bev_features_channels, config.bev_num_classes, 1),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        # Perception DETR-style decoder
        tf_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(tf_layer, config.tf_num_layers)
        self._agent_head = AgentHead(config.num_bounding_boxes, config.tf_d_ffn, config.tf_d_model)

        # BEV cross-feature projection (upscale + encoder output concat → bev_proj)
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(config.tf_d_model, 1, 1, 320),
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.num_poses,
            d_model=config.tf_d_model,
            d_ffn=config.tf_d_ffn,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        camera_feat = features["camera_feature"]
        lidar_feat = features["lidar_feature"]
        status_feat = features["status_feature"]
        bs = status_feat.shape[0]

        bev_upscale, bev_feature, _ = self._backbone(camera_feat, lidar_feat)
        bev_spatial_shape = bev_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]

        # Flatten deepest BEV feature → 64 key-val tokens
        bev_flat = self._bev_downscale(bev_feature).flatten(-2, -1)  # B, D, 64
        bev_flat = bev_flat.permute(0, 2, 1)  # B, 64, D

        status_enc = self._status_encoding(status_feat)  # B, D
        keyval = torch.cat([bev_flat, status_enc[:, None]], dim=1)  # B, 65, D
        keyval += self._keyval_embedding.weight[None, ...]

        # Build cross-BEV feature for GridSampleCrossBEVAttention
        concat_cross_bev = (
            keyval[:, :-1]
            .permute(0, 2, 1)
            .contiguous()
            .view(bs, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        )
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode="bilinear", align_corners=False)
        cross_bev_feature = torch.cat([concat_cross_bev, bev_upscale], dim=1)  # B, D+64, H, W
        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))
        cross_bev_feature = (
            cross_bev_feature.permute(0, 2, 1).contiguous().view(bs, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        )

        # Perception decoder
        query = self._query_embedding.weight[None, ...].expand(bs, -1, -1)
        query_out = self._tf_decoder(query, keyval)
        traj_query, agents_query = query_out.split(self._query_splits, dim=1)

        # Planner head (inference only)
        output = self._trajectory_head(
            traj_query, agents_query, cross_bev_feature, bev_spatial_shape, status_enc[:, None]
        )

        # Agent head
        agents = self._agent_head(agents_query)
        output.update(agents)
        return output


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str, config: DiffusionDriveConfig, device: Optional[torch.device] = None
) -> DiffusionDriveModel:
    """Load a DiffusionDriveModel from a checkpoint (strict=False).

    Args:
        checkpoint_path: path to .pth checkpoint from hustvl/DiffusionDrive
        config: model config (must have plan_anchor_path set)
        device: target device; defaults to CPU

    Returns:
        model in eval mode
    """
    if device is None:
        device = torch.device("cpu")

    model = DiffusionDriveModel(config)
    ckpt = torch.load(checkpoint_path, map_location=device)

    # upstream checkpoints are nested under "state_dict"
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("agent.", "").replace("_transfuser_model.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_model] missing keys ({len(missing)}): {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[load_model] unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
    model.to(device).eval()
    return model
