# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
BEVFormer detector in PyTorch: camera images and their metadata in, per-layer class logits and
boxes and the frame's BEV out.

This module is the inference path of upstream's ``BEVFormer.simple_test`` and
``BEVFormerHead.forward`` (``projects/mmdet3d_plugin/bevformer/{detectors/bevformer.py,
dense_heads/bevformer_head.py}`` in fundamentalvision/BEVFormer), and the reference the TTNN
detector in ``tt/tt_bevformer.py`` is checked against:
1. The ResNet backbone and FPN turn the ``bs * num_cams`` images into four feature levels.
2. ``PerceptionTransformer.get_bev_features`` builds the BEV map from them, the learned BEV queries
   and positional encoding, and the previous frame's BEV.
3. The head decodes the BEV map into per-layer class logits and boxes.

The caller carries the previous BEV between frames and makes the CAN bus relative to the previous
frame (:func:`relative_can_bus`), as upstream's ``forward_test`` does.

:func:`build_bevformer_base` builds BEVFormer-base from ``model_config.py``, and
:func:`load_bevformer_checkpoint` loads a BEVFormer checkpoint into it, every key but the loss's
``code_weights`` accounted for.
"""

import torch
import torch.nn as nn

from models.experimental.bevformer.model_config import BEV_H, BEV_W, EMBED_DIMS, FPN_KWARGS, RESNET_KWARGS
from models.experimental.bevformer.reference.encoder import BEVFormerEncoder
from models.experimental.bevformer.reference.fpn import FPN
from models.experimental.bevformer.reference.head import BEVFormerHead
from models.experimental.bevformer.reference.perception_transformer import PerceptionTransformer
from models.experimental.bevformer.reference.resnet import ResNet


class LearnedPositionalEncoding(nn.Module):
    """mmdet's ``LearnedPositionalEncoding``: each BEV cell's encoding is its column's embedding
    followed by its row's."""

    def __init__(self, num_feats, row_num_embed, col_num_embed):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)

    def forward(self, bev_h, bev_w):
        """``(bev_h * bev_w, 2 * num_feats)``, row-major over the grid."""
        x_embed = self.col_embed(torch.arange(bev_w))
        y_embed = self.row_embed(torch.arange(bev_h))
        pos = torch.cat([x_embed[None].expand(bev_h, -1, -1), y_embed[:, None].expand(-1, bev_w, -1)], dim=-1)
        return pos.reshape(bev_h * bev_w, -1)


def relative_can_bus(can_bus, previous_can_bus=None):
    """Upstream ``forward_test``'s CAN bus: one frame's absolute 18-vector with its translation
    ``[:3]`` and heading ``[-1]`` made relative to the previous frame's, or zero without one."""
    can_bus = torch.as_tensor(can_bus, dtype=torch.float64).clone()
    if previous_can_bus is None:
        can_bus[:3] = 0
        can_bus[-1] = 0
    else:
        previous_can_bus = torch.as_tensor(previous_can_bus, dtype=torch.float64)
        can_bus[:3] -= previous_can_bus[:3]
        can_bus[-1] -= previous_can_bus[-1]
    return can_bus


class BEVFormer(nn.Module):
    """
    The detector over a ``(bev_h, bev_w)`` grid.

    Args:
        img_backbone (nn.Module): ``reference.resnet.ResNet``.
        img_neck (nn.Module): ``reference.fpn.FPN``.
        transformer (nn.Module): ``reference.perception_transformer.PerceptionTransformer``.
        head (nn.Module): ``reference.head.BEVFormerHead``, which holds the decoder.
        bev_h, bev_w (int): BEV grid.
        embed_dims (int): Channels of the BEV queries and their positional encoding.

    ``bev_embedding`` and ``positional_encoding`` are upstream's head's; they are kept here, next to
    the BEV side that uses them.
    """

    def __init__(self, img_backbone, img_neck, transformer, head, bev_h=BEV_H, bev_w=BEV_W, embed_dims=EMBED_DIMS):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.transformer = transformer
        self.head = head
        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
        self.positional_encoding = LearnedPositionalEncoding(embed_dims // 2, bev_h, bev_w)

    def extract_img_feat(self, img):
        """``img`` ``(bs, num_cams, 3, H, W)`` -> per FPN level ``(bs, num_cams, C, h, w)``."""
        bs, num_cams = img.shape[:2]
        levels = self.img_neck(list(self.img_backbone(img.flatten(0, 1))))
        return [feat.view(bs, num_cams, *feat.shape[1:]) for feat in levels]

    def forward(self, img, img_metas, prev_bev=None):
        """``img`` ``(bs, num_cams, 3, H, W)``, normalized; ``img_metas`` per sample with
        ``lidar2img``, ``img_shape`` and the relative ``can_bus``; ``prev_bev`` the previous frame's
        ``(bs, bev_h * bev_w, C)`` BEV or None. Returns the per-layer class logits
        ``(L, bs, num_query, num_classes)``, boxes ``(L, bs, num_query, code_size)`` and this frame's
        BEV ``(bs, bev_h * bev_w, C)``."""
        mlvl_feats = self.extract_img_feat(img)
        bev_pos = self.positional_encoding(self.bev_h, self.bev_w)[:, None, :].expand(-1, len(img_metas), -1)
        bev_embed = self.transformer.get_bev_features(
            mlvl_feats, self.bev_embedding.weight, self.bev_h, self.bev_w, bev_pos, img_metas, prev_bev=prev_bev
        )
        cls_scores, bbox_preds = self.head(bev_embed)
        return cls_scores, bbox_preds, bev_embed


def build_bevformer_base(bev_h=BEV_H, bev_w=BEV_W):
    """BEVFormer-base with PyTorch's default init, ready for :func:`load_bevformer_checkpoint`."""
    return BEVFormer(
        ResNet(**RESNET_KWARGS),
        FPN(**FPN_KWARGS),
        PerceptionTransformer(BEVFormerEncoder()),
        BEVFormerHead(bev_h, bev_w),
        bev_h,
        bev_w,
    )


def load_bevformer_checkpoint(model, state_dict):
    """Load a BEVFormer checkpoint's ``state_dict`` into ``model``, strictly. Upstream's
    ``pts_bbox_head`` holds the BEV queries, the positional encoding and the whole transformer;
    here the decoder and its reference-point Linear are the head's and the rest of the BEV side is
    the detector's."""
    # The first matching prefix wins, so the more specific ones come first.
    renames = (
        ("pts_bbox_head.transformer.decoder.", "head.decoder."),
        ("pts_bbox_head.transformer.reference_points.", "head.reference_points."),
        ("pts_bbox_head.transformer.", "transformer."),
        ("pts_bbox_head.bev_embedding.", "bev_embedding."),
        ("pts_bbox_head.positional_encoding.", "positional_encoding."),
        ("pts_bbox_head.", "head."),
    )
    mapped = {}
    for key, value in state_dict.items():
        if key == "pts_bbox_head.code_weights":
            continue
        for old, new in renames:
            if key.startswith(old):
                key = new + key[len(old) :]
                break
        mapped[key] = value
    model.load_state_dict(mapped, strict=True)
    return model
