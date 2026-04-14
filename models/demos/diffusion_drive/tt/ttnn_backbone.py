# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN TransFuser backbone — Stage 2.

Replaces the 32 ResNet-34 BasicBlock stages (16 per image/LiDAR encoder ×
4 backbone stages each) with native TTNN conv2d using BN-folded weights.

What runs in TTNN:
    All timm BasicBlock layers in image_encoder.layer{1-4} and
    lidar_encoder.layer{1-4} (conv3×3 + BN-fold + ReLU + optional 1×1).

What stays in PyTorch (TorchModuleFallback):
    • Stem: conv1 + bn1 + act1 + MaxPool2d
    • AdaptiveAvgPool2d (for GPT anchor tokens)
    • 1×1 channel-projection Conv2d (lidar↔image)
    • GPT self-attention fusion blocks
    • F.interpolate (bilinear)
    • 3-level top-down FPN (Conv2d + bilinear Upsample)

Public API::
    ttnn_bb = TtnnTransfuserBackbone(ref_backbone, device)
    bev_upscale, bev_feature, _ = ttnn_bb(image, lidar)

The return signature matches TransfuserBackbone.forward() so that
_TtnnBackboneAdapter can be assigned as a drop-in for the nn.Module.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

import ttnn
from models.demos.diffusion_drive.tt.ttnn_resnet34 import TtnnBasicBlock, prepare_resnet34_stage_params

# ---------------------------------------------------------------------------
# Tensor-format helpers
# ---------------------------------------------------------------------------


def _to_ttnn_tile(x: torch.Tensor, B: int, H: int, W: int, C: int, device: ttnn.Device) -> ttnn.Tensor:
    """Convert (B, C, H, W) float32 PyTorch → (1,1,B*H*W,C) bfloat16 TILE on device."""
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    x_flat = x_nhwc.reshape(1, 1, B * H * W, C).to(torch.bfloat16)
    return ttnn.from_torch(x_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _from_ttnn_tile(x_ttnn: ttnn.Tensor, B: int, H: int, W: int, C: int) -> torch.Tensor:
    """Convert (1,1,B*H*W,C) TTNN → (B, C, H, W) float32 PyTorch."""
    if x_ttnn.is_sharded():
        x_ttnn = ttnn.sharded_to_interleaved(x_ttnn, ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.to_torch(x_ttnn)  # (1, 1, B*H*W, C)
    out = out.reshape(B, H, W, C)
    return out.permute(0, 3, 1, 2).float()  # (B, C, H, W)


# ---------------------------------------------------------------------------
# TTNN TransFuser backbone
# ---------------------------------------------------------------------------


class TtnnTransfuserBackbone:
    """
    TTNN Stage-2 wrapper for TransfuserBackbone.

    All ResNet-34 BasicBlock stages run on-device via TtnnBasicBlock
    (TTNN conv2d + BN-folded weights). Everything else remains in PyTorch.

    Parameters
    ----------
    ref : TransfuserBackbone
        Pre-loaded, eval-mode reference backbone.
    device : ttnn.Device
        Opened Wormhole device (must have been opened with l1_small_size ≥ 32768).
    """

    def __init__(self, ref, device: ttnn.Device) -> None:
        self._ref = ref
        self._device = device

        # Pre-fold BN for all BasicBlocks in layers 1-4 of each encoder.
        # _img_stages[i] and _lidar_stages[i] are lists of (stride, params)
        # for the i-th ResNet-34 stage (0 = layer1, …, 3 = layer4).
        self._img_stages: List[List[Tuple[int, dict]]] = []
        self._lidar_stages: List[List[Tuple[int, dict]]] = []
        for i in range(4):
            img_layer = getattr(ref.image_encoder, f"layer{i + 1}")
            lidar_layer = getattr(ref.lidar_encoder, f"layer{i + 1}")
            self._img_stages.append(prepare_resnet34_stage_params(img_layer))
            self._lidar_stages.append(prepare_resnet34_stage_params(lidar_layer))

        # Stage 3: optional TTNN FPN (set by build_stage3)
        self._ttnn_fpn = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_ttnn_stage(
        self,
        x: torch.Tensor,
        stage_blocks: List[Tuple[int, dict]],
    ) -> torch.Tensor:
        """Convert to TTNN, run all BasicBlocks in one stage, convert back.

        Args:
            x:            (B, C, H, W) float32 PyTorch tensor.
            stage_blocks: [(stride, params_dict), …] from
                          prepare_resnet34_stage_params.
        Returns:
            (B, C_out, H_out, W_out) float32 PyTorch tensor.
        """
        B, C, H, W = x.shape
        x_ttnn = _to_ttnn_tile(x, B, H, W, C, self._device)
        shape = (B, H, W, C)

        for stride, params in stage_blocks:
            block = TtnnBasicBlock(params, stride=stride, device=self._device)
            x_ttnn, shape = block(x_ttnn, shape)

        B_out, H_out, W_out, C_out = shape
        return _from_ttnn_tile(x_ttnn, B_out, H_out, W_out, C_out)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: torch.Tensor,
        lidar: torch.Tensor,
    ):
        """
        Args:
            image: (B, 3, H_cam, W_cam)   — camera feature map.
            lidar: (B, 1, H_lid, W_lid)   — LiDAR BEV; ignored if config.latent.
        Returns:
            (bev_upscale, bev_feature, None) matching TransfuserBackbone.forward().
              bev_upscale:  (B, 64, H_bev, W_bev)
              bev_feature:  (B, 512, H_bev/8, W_bev/8)
        """
        ref = self._ref

        if ref.config.latent:
            lidar = ref.lidar_latent.expand(image.shape[0], -1, -1, -1)

        img_feats: torch.Tensor = image
        lidar_feats: torch.Tensor = lidar

        # ------------------------------------------------------------------
        # Step 1: Stem — conv1 + bn1 + act1 (PyTorch).
        # timm resnet34 features_only has 5 return_layers (start_index = 1):
        #   {'act1': '0', 'layer1': '1', …, 'layer4': '4'}
        # ------------------------------------------------------------------
        si = ref._start_index  # 1 for standard timm resnet34
        if si > 0:
            img_feats = ref.image_encoder.act1(ref.image_encoder.bn1(ref.image_encoder.conv1(img_feats)))
            lidar_feats = ref.lidar_encoder.act1(ref.lidar_encoder.bn1(ref.lidar_encoder.conv1(lidar_feats)))

        # ------------------------------------------------------------------
        # Steps 2-5: Four backbone stages, each followed by GPT fusion.
        # For stage 0 (i=0) the timm iterator first runs maxpool (PyTorch).
        # ------------------------------------------------------------------
        for i in range(4):
            # MaxPool is part of the stage-0 iteration in the reference model.
            if i == 0 and si > 0:
                img_feats = ref.image_encoder.maxpool(img_feats)
                lidar_feats = ref.lidar_encoder.maxpool(lidar_feats)

            # TTNN: run layer{i+1} BasicBlocks for both encoders.
            img_feats = self._run_ttnn_stage(img_feats, self._img_stages[i])
            lidar_feats = self._run_ttnn_stage(lidar_feats, self._lidar_stages[i])

            # PyTorch: GPT cross-modal fusion (avgpool + 1×1 proj + attention +
            # F.interpolate + residual add).
            img_feats, lidar_feats = ref._fuse_features(img_feats, lidar_feats, i)

        # ------------------------------------------------------------------
        # Step 6: 3-level top-down FPN.
        # Stage 3+: TTNN conv2d (bilinear upsample stays in PyTorch).
        # Stage 2:  pure PyTorch reference.
        # ------------------------------------------------------------------
        if self._ttnn_fpn is not None:
            bev_upscale = self._ttnn_fpn(lidar_feats)
        else:
            bev_upscale = ref._top_down(lidar_feats)
        return bev_upscale, lidar_feats, None


# ---------------------------------------------------------------------------
# Drop-in nn.Module adapter
# ---------------------------------------------------------------------------


class _TtnnBackboneAdapter(nn.Module):
    """Thin nn.Module wrapper so TtnnTransfuserBackbone can be assigned
    directly to ``DiffusionDriveModel._backbone``."""

    def __init__(self, ttnn_backbone: TtnnTransfuserBackbone) -> None:
        super().__init__()
        self._ttnn = ttnn_backbone
        # Forward config so downstream code (FPN size queries) still works.
        self.config = ttnn_backbone._ref.config

    def forward(self, image: torch.Tensor, lidar: torch.Tensor):
        return self._ttnn(image, lidar)
