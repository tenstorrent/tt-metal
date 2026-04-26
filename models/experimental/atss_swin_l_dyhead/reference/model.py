# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Full ATSS model (Swin-L + FPN + DyHead + ATSS Head) — inference only.
# Assembles the standalone reference components and provides
# checkpoint loading from mmdet-format .pth files.

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .swin_transformer import SwinTransformer, build_swin_l_backbone
from .fpn import FPN, build_fpn_for_atss
from .dyhead import DyHead, build_dyhead_for_atss
from .atss_head import ATSSHead, build_atss_head
from .postprocess import atss_postprocess


class ATSSModel(nn.Module):
    """Full ATSS detector: Swin-L backbone + FPN + DyHead + ATSS head.

    Inference-only reference implementation for PCC validation.
    """

    def __init__(
        self,
        backbone: SwinTransformer,
        fpn: FPN,
        dyhead: DyHead,
        head: ATSSHead,
        # Data preprocessor config
        pixel_mean: Tuple[float, ...] = (123.675, 116.28, 103.53),
        pixel_std: Tuple[float, ...] = (58.395, 57.12, 57.375),
        pad_size_divisor: int = 128,
        bgr_to_rgb: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.dyhead = dyhead
        self.head = head

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(1, 3, 1, 1), persistent=False)
        self.pad_size_divisor = pad_size_divisor
        self.bgr_to_rgb = bgr_to_rgb

    def preprocess(self, img: Tensor) -> Tensor:
        """Normalize and pad an image tensor.

        Args:
            img: (1, 3, H, W) in BGR uint8 or float [0, 255].

        Returns:
            Normalized, padded tensor.
        """
        x = img.float()
        if self.bgr_to_rgb:
            x = x[:, [2, 1, 0], :, :]
        x = (x - self.pixel_mean) / self.pixel_std

        _, _, h, w = x.shape
        pad_h = (self.pad_size_divisor - h % self.pad_size_divisor) % self.pad_size_divisor
        pad_w = (self.pad_size_divisor - w % self.pad_size_divisor) % self.pad_size_divisor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x

    def forward_backbone(self, x: Tensor) -> List[Tensor]:
        """Run backbone → FPN → DyHead, return refined multi-scale features."""
        feats = self.backbone(x)
        fpn_feats = self.fpn(tuple(feats))
        dy_feats = self.dyhead(list(fpn_feats))
        return dy_feats

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Full forward pass.

        Args:
            x: Preprocessed image tensor (1, 3, H, W).

        Returns:
            Tuple of (cls_scores, bbox_preds, centernesses) per FPN level.
        """
        dy_feats = self.forward_backbone(x)
        cls_scores, bbox_preds, centernesses = self.head(tuple(dy_feats))
        return cls_scores, bbox_preds, centernesses

    @torch.no_grad()
    def predict(
        self,
        img: Tensor,
        img_shape: Tuple[int, int],
        score_thr: float = 0.05,
        nms_iou_thr: float = 0.6,
        max_per_img: int = 100,
    ) -> Dict[str, Tensor]:
        """End-to-end inference: preprocess → forward → postprocess.

        Args:
            img: (1, 3, H, W) raw image in BGR, float [0, 255].
            img_shape: (H, W) of the original image (before padding).
            score_thr: Score threshold.
            nms_iou_thr: NMS IoU threshold.
            max_per_img: Max detections per image.

        Returns:
            Dict with 'bboxes', 'scores', 'labels'.
        """
        x = self.preprocess(img)
        cls_scores, bbox_preds, centernesses = self.forward(x)
        return atss_postprocess(
            cls_scores,
            bbox_preds,
            centernesses,
            img_shape=img_shape,
            score_thr=score_thr,
            nms_iou_thr=nms_iou_thr,
            max_per_img=max_per_img,
        )


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _swin_converter(ckpt: dict) -> dict:
    """Convert original Swin weights (from official repo) to our format.

    Handles the key mapping differences:
    - attn.* → attn.w_msa.*
    - mlp.fc1.* → ffn.fc1.*
    - mlp.fc2.* → ffn.fc2.*
    - layers.* → stages.*
    - patch_embed.proj.* → patch_embed.projection.*
    - Reorders unfold-based PatchMerging reduction weights.
    """
    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith("head"):
            continue
        elif k.startswith("layers"):
            new_v = v
            if "attn." in k:
                new_k = k.replace("attn.", "attn.w_msa.")
            elif "mlp." in k:
                if "mlp.fc1." in k:
                    new_k = k.replace("mlp.fc1.", "ffn.fc1.")
                elif "mlp.fc2." in k:
                    new_k = k.replace("mlp.fc2.", "ffn.fc2.")
                else:
                    new_k = k.replace("mlp.", "ffn.")
            elif "downsample" in k:
                new_k = k
                if "reduction." in k:
                    new_v = correct_unfold_reduction_order(v)
                elif "norm." in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace("layers", "stages", 1)
        elif k.startswith("patch_embed"):
            new_v = v
            if "proj" in k:
                new_k = k.replace("proj", "projection")
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt[new_k] = new_v
    return new_ckpt


def load_mmdet_checkpoint(
    model: ATSSModel,
    checkpoint_path: str,
    map_location: str = "cpu",
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load an mmdet-format ATSS checkpoint into our standalone model.

    The mmdet checkpoint has keys like:
        backbone.stages.0.blocks.0.norm1.weight
        neck.0.lateral_convs.0.conv.weight    (FPN)
        neck.1.dyhead_blocks.0.spatial_conv_mid.conv.weight  (DyHead)
        bbox_head.atss_cls.weight

    We map these to our model structure:
        backbone.stages.0.blocks.0.norm1.weight
        fpn.lateral_convs.0.weight
        dyhead.dyhead_blocks.0.spatial_conv_mid.conv.weight
        head.atss_cls.weight
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k

        # Backbone: backbone.* → backbone.*
        if k.startswith("backbone."):
            new_k = k
            # FFN key mapping: mmdet uses ffn.layers.0.0.* and ffn.layers.1.*
            # Our standalone uses ffn.fc1.* and ffn.fc2.*
            new_k = new_k.replace(".ffn.layers.0.0.", ".ffn.fc1.")
            new_k = new_k.replace(".ffn.layers.1.", ".ffn.fc2.")

        # FPN: neck.0.* → fpn.*
        elif k.startswith("neck.0."):
            new_k = k.replace("neck.0.", "fpn.")
            # mmdet wraps in ConvModule: .conv.weight, .conv.bias
            # our standalone uses nn.Conv2d directly (no .conv. prefix)
            new_k = new_k.replace(".conv.weight", ".weight")
            new_k = new_k.replace(".conv.bias", ".bias")

        # DyHead: neck.1.* → dyhead.*
        elif k.startswith("neck.1."):
            new_k = k.replace("neck.1.", "dyhead.")
            # DyReLU conv layers in mmdet use ConvModule wrapper
            # task_attn_module.conv1.conv.weight → task_attn_module.conv1.0.weight
            new_k = new_k.replace("task_attn_module.conv1.conv.", "task_attn_module.conv1.0.")
            new_k = new_k.replace("task_attn_module.conv2.conv.", "task_attn_module.conv2.0.")
            # Scale attention conv layers
            # scale_attn_module.1.weight → scale_attn_module.1.weight  (already correct)

        # ATSS head: bbox_head.* → head.*
        elif k.startswith("bbox_head."):
            new_k = k.replace("bbox_head.", "head.")

        # Data preprocessor (skip)
        elif k.startswith("data_preprocessor."):
            continue

        else:
            new_k = k

        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
    return missing, unexpected


def build_atss_model() -> ATSSModel:
    """Build the full ATSS model with default config."""
    backbone = build_swin_l_backbone()
    fpn = build_fpn_for_atss()
    dyhead = build_dyhead_for_atss()
    head = build_atss_head()
    return ATSSModel(backbone, fpn, dyhead, head)
