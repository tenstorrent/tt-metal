# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

##########################################################################
# Adapted from BEVFormer (https://github.com/fundamentalvision/BEVFormer).
# Original work Copyright (c) OpenMMLab.
# Modified by Zhiqi Li.
# Licensed under the Apache License, Version 2.0.
##########################################################################

import torch
import torch.nn as nn

from .resnet import ResNet
from .fpn import FPN
from .head import BEVFormerHead


class GridMask(nn.Module):
    def __init__(self, use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob <= 0:
            return x
        return x


class BEVFormerV2(nn.Module):
    def __init__(
        self,
        use_grid_mask=False,
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        num_levels=None,
        num_mono_levels=None,
        mono_loss_weight=1.0,
        frames=(0,),
        **kwargs,
    ):
        super(BEVFormerV2, self).__init__()
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.video_test_mode = video_test_mode
        self.mono_loss_weight = mono_loss_weight
        self.num_levels = num_levels
        self.num_mono_levels = num_mono_levels
        self.frames = frames

        if img_backbone:
            self.img_backbone = ResNet(
                depth=img_backbone.get("depth", 50),
                in_channels=img_backbone.get("in_channels", 3),
                out_indices=img_backbone.get("out_indices", (1, 2, 3)),
                style=img_backbone.get("style", "caffe"),
            )
        else:
            self.img_backbone = None

        if img_neck:
            self.img_neck = FPN(
                in_channels=img_neck.get("in_channels", [512, 1024, 2048]),
                out_channels=img_neck.get("out_channels", 256),
                num_outs=img_neck.get("num_outs", 5),
                start_level=img_neck.get("start_level", 0),
                add_extra_convs=img_neck.get("add_extra_convs", "on_output"),
                relu_before_extra_convs=img_neck.get("relu_before_extra_convs", True),
            )
        else:
            self.img_neck = None

        if pts_bbox_head:
            self.pts_bbox_head = BEVFormerHead(
                bev_h=pts_bbox_head.get("bev_h", 200),
                bev_w=pts_bbox_head.get("bev_w", 200),
                num_query=pts_bbox_head.get("num_query", 900),
                num_classes=pts_bbox_head.get("num_classes", 10),
                in_channels=pts_bbox_head.get("in_channels", 256),
                sync_cls_avg_factor=pts_bbox_head.get("sync_cls_avg_factor", True),
                with_box_refine=pts_bbox_head.get("with_box_refine", True),
                as_two_stage=pts_bbox_head.get("as_two_stage", False),
            )
        else:
            self.pts_bbox_head = None

    def extract_img_feat(self, img, img_metas=None):
        """Extract features of images."""
        if img is not None:
            if img.dim() == 5:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            else:
                B = 1
                BN, C, H, W = img.size()
                N = BN

            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        if self.img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas, len_queue=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        if (
            "aug_param" in img_metas[0]
            and img_metas[0]["aug_param"].get("CropResizeFlipImage_param", [None])[-1] is True
        ):
            img_feats = [
                torch.flip(
                    x,
                    dims=[
                        -1,
                    ],
                )
                for x in img_feats
            ]
        return img_feats

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether return_loss=True."""
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self, img_metas=None, **kwargs):
        """Forward function for training."""
        img = kwargs.get("img", None)
        img_feats = self.extract_feat(img, img_metas)
        outs = self.pts_bbox_head(img_feats, img_metas)
        return outs

    def forward_test(self, img_metas=None, **kwargs):
        img = kwargs.get("img", None)
        if isinstance(img, list) and len(img) > 0:
            img = img[0]
        while isinstance(img_metas, list) and len(img_metas) > 0 and isinstance(img_metas[0], list):
            img_metas = img_metas[0]
        img_feats = self.extract_feat(img, img_metas)
        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev=None)

        import os

        save_path = "models/experimental/BEVFormerV2/reference/dumps"
        os.makedirs(save_path, exist_ok=True)
        keys_to_save = ["bev_embed", "all_cls_scores", "all_bbox_preds"]
        for key in keys_to_save:
            if key in outs:
                tensor = outs[key]
                torch.save(tensor, os.path.join(save_path, f"{key}.pt"))

        bbox_list = self.pts_bbox_head.bbox_coder.decode(outs)
        bbox_results = [
            dict(
                pts_bbox=dict(
                    boxes_3d=bbox_list[i]["bboxes"],
                    scores_3d=bbox_list[i]["scores"],
                    labels_3d=bbox_list[i]["labels"],
                )
            )
            for i in range(len(bbox_list))
        ]
        return bbox_results

    def simple_test(self, img, img_metas, **kwargs):
        """Test function without augmentation."""
        return self.forward_test(img=img, img_metas=img_metas, **kwargs)
