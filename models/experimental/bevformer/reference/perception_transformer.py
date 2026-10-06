# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
BEV half of BEVFormer's ``PerceptionTransformer`` in PyTorch: camera features and BEV queries in,
the frame's BEV map out.

This module is upstream's ``PerceptionTransformer.get_bev_features``
(``projects/mmdet3d_plugin/bevformer/modules/transformer.py`` in fundamentalvision/BEVFormer), the
glue around the encoder, and the reference the TTNN version in ``tt/tt_perception_transformer.py``
is checked against.

Per frame:
1. The BEV queries get the CAN-bus MLP of the frame's ego motion added.
2. Each FPN level is flattened per camera, the camera and level embeddings are added, and the
   levels are concatenated: the encoder's ``(num_cams, num_keys, bs, C)`` value.
3. The previous frame's BEV, if any, is rotated per sample by the heading change about
   ``rotate_center`` (torchvision's ``rotate``, nearest), and the encoder shifts its reference
   points by the ego translation (:func:`ego_shift`).
4. The encoder builds the BEV map.

``img_metas[b]["can_bus"]`` is the 18-value CAN-bus vector with its translation ``[:3]`` and heading
``[-1]`` relative to the previous frame, as upstream's ``forward_test`` makes it (zero on a frame
without a previous BEV); ``can_bus[-2]`` is the absolute heading in radians.

The decoder half of upstream's ``PerceptionTransformer`` (``decoder`` and ``reference_points``) is
the head's, see ``reference/head.py``. Parameter names follow the checkpoint's
``pts_bbox_head.transformer``: ``encoder``, ``level_embeds``, ``cams_embeds`` and ``can_bus_mlp``
load unchanged. Upstream's defaults are kept: CAN bus, camera embeddings and the previous BEV's
rotation are on, ``can_bus_norm`` is on.
"""

import math

import torch
import torch.nn as nn
from torchvision.transforms.functional import rotate

CAN_BUS_DIMS = 18
# BEV cell size in metres, (y, x): upstream's ``grid_length`` default, 102.4 m over 200 cells.
GRID_LENGTH = (0.512, 0.512)
ROTATE_CENTER = (100, 100)


def ego_shift(img_metas, bev_h, bev_w, grid_length=GRID_LENGTH):
    """The ego translation in BEV-grid fractions, ``(bs, 2)`` as (x, y): upstream's ``shift``."""
    shifts = []
    for meta in img_metas:
        can_bus = meta["can_bus"]
        delta_x, delta_y = float(can_bus[0]), float(can_bus[1])
        ego_angle = float(can_bus[-2]) / math.pi * 180
        translation_length = math.sqrt(delta_x**2 + delta_y**2)
        translation_angle = math.atan2(delta_y, delta_x) / math.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * math.cos(bev_angle / 180 * math.pi) / grid_length[0] / bev_h
        shift_x = translation_length * math.sin(bev_angle / 180 * math.pi) / grid_length[1] / bev_w
        shifts.append([shift_x, shift_y])
    return torch.tensor(shifts, dtype=torch.float32)


def rotate_bev(bev, img_metas, bev_h, bev_w, rotate_center=ROTATE_CENTER):
    """``bev`` ``(bev_h * bev_w, bs, C)`` with each sample rotated by its heading change
    (``can_bus[-1]``, degrees) about ``rotate_center``; cells rotated in from outside the grid are
    zero."""
    rotated = []
    for b, meta in enumerate(img_metas):
        grid = bev[:, b].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
        grid = rotate(grid, float(meta["can_bus"][-1]), center=list(rotate_center))
        rotated.append(grid.permute(1, 2, 0).reshape(bev_h * bev_w, -1))
    return torch.stack(rotated, dim=1)


class PerceptionTransformer(nn.Module):
    """
    The encoder and its glue.

    Args:
        encoder (nn.Module): ``reference.encoder.BEVFormerEncoder``.
        embed_dims (int): Channels of the features, queries and BEV.
        num_feature_levels (int): FPN levels.
        num_cams (int): Cameras.
        rotate_center (tuple[int]): Pixel the previous BEV rotates about, (x, y).
    """

    def __init__(self, encoder, embed_dims=256, num_feature_levels=4, num_cams=6, rotate_center=ROTATE_CENTER):
        super().__init__()
        self.encoder = encoder
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.rotate_center = rotate_center
        self.level_embeds = nn.Parameter(torch.empty(num_feature_levels, embed_dims))
        self.cams_embeds = nn.Parameter(torch.empty(num_cams, embed_dims))
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(CAN_BUS_DIMS, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.ReLU(inplace=True),
        )
        self.can_bus_mlp.add_module("norm", nn.LayerNorm(embed_dims))
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)

    def flatten_features(self, mlvl_feats):
        """``mlvl_feats`` per level ``(bs, num_cams, C, h, w)`` -> the encoder's value
        ``(num_cams, num_keys, bs, C)`` with the camera and level embeddings, and the levels'
        ``(h, w)``."""
        flattened, spatial_shapes = [], []
        for lvl, feat in enumerate(mlvl_feats):
            _, _, _, h, w = feat.shape
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # (num_cams, bs, h * w, C)
            feat = feat + self.cams_embeds[:, None, None, :] + self.level_embeds[None, None, lvl : lvl + 1, :]
            flattened.append(feat)
            spatial_shapes.append((h, w))
        value = torch.cat(flattened, 2).permute(0, 2, 1, 3)
        return value, torch.tensor(spatial_shapes)

    def get_bev_features(self, mlvl_feats, bev_queries, bev_h, bev_w, bev_pos, img_metas, prev_bev=None):
        """``bev_queries`` ``(bev_h * bev_w, C)``, shared by the batch; ``bev_pos``
        ``(bev_h * bev_w, bs, C)``; ``prev_bev`` the previous frame's ``(bs, bev_h * bev_w, C)``
        BEV or None. Returns this frame's BEV, ``(bs, bev_h * bev_w, C)``."""
        can_bus = torch.tensor([list(map(float, meta["can_bus"])) for meta in img_metas], dtype=bev_queries.dtype)
        bev_queries = bev_queries[:, None, :] + self.can_bus_mlp(can_bus)[None]
        value, spatial_shapes = self.flatten_features(mlvl_feats)
        if prev_bev is not None:
            prev_bev = rotate_bev(prev_bev.permute(1, 0, 2), img_metas, bev_h, bev_w, self.rotate_center)
        return self.encoder(
            bev_queries,
            value,
            bev_h,
            bev_w,
            bev_pos,
            spatial_shapes,
            img_metas,
            prev_bev=prev_bev,
            shift=ego_shift(img_metas, bev_h, bev_w),
        )
