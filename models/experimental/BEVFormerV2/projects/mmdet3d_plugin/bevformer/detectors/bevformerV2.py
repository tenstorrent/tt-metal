# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
# Source: Adapted from mmdet3d.models.detectors (BEVFormer/Detr3D pattern)
# Original reference: https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/models/detectors/mvx_two_stage.py

import copy
from collections import OrderedDict
import torch
from models.experimental.BEVFormerV2.projects.mmdet3d_plugin.dependency import (
    DETECTORS,
    bbox3d2result,
    MVXTwoStageDetector,
    build_head,
)
from models.experimental.BEVFormerV2.projects.mmdet3d_plugin.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class BEVFormerV2(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        fcos3d_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        num_levels=None,
        num_mono_levels=None,
        mono_loss_weight=1.0,
        frames=(0,),
    ):
        super(BEVFormerV2, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        assert not self.fp16_enabled  # not support fp16 yet
        # temporal
        self.video_test_mode = video_test_mode
        assert not self.video_test_mode  # not support video_test_mode yet

        # fcos3d head
        self.fcos3d_bbox_head = build_head(fcos3d_bbox_head) if fcos3d_bbox_head else None
        # loss weight
        self.mono_loss_weight = mono_loss_weight

        # levels of features
        self.num_levels = num_levels
        self.num_mono_levels = num_mono_levels
        self.frames = frames

    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        # Debug: Check backbone features before neck
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0
        if self._debug_count < 1:  # Only print first time
            if img_feats:
                print(f"\n[DEBUG] Backbone features before neck: {len(img_feats)} levels")
                for i, feat in enumerate(img_feats):
                    print(
                        f"  Level {i}: shape={feat.shape}, mean={feat.mean().item():.6f}, std={feat.std().item():.6f}"
                    )

        if self.with_img_neck:
            img_feats_before_neck = img_feats
            img_feats = self.img_neck(img_feats)
            # Debug: Check neck output
            if self._debug_count < 1:
                print(f"[DEBUG] After img_neck: {len(img_feats)} levels")
                for i, feat in enumerate(img_feats):
                    mean_val = feat.mean().item()
                    std_val = feat.std().item()
                    max_val = feat.max().item()
                    min_val = feat.min().item()
                    print(
                        f"  Level {i}: shape={feat.shape}, mean={mean_val:.6f}, std={std_val:.6f}, min={min_val:.6f}, max={max_val:.6f}"
                    )
                    # Warn if std is very high (potential numerical instability)
                    if std_val > 100:
                        print(
                            f"    ⚠ WARNING: Level {i} has very high std ({std_val:.2f}), possible numerical instability!"
                        )
                # Check if neck actually modified features
                if len(img_feats) == len(img_feats_before_neck):
                    for i in range(len(img_feats)):
                        if torch.equal(img_feats[i], img_feats_before_neck[i]):
                            print(f"  WARNING: Level {i} unchanged by neck!")
                self._debug_count += 1
        else:
            if self._debug_count < 1:
                print(f"[DEBUG] WARNING: with_img_neck is False, skipping neck!")
                self._debug_count += 1

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img)
        # Safely check for aug_param and CropResizeFlipImage_param
        # Handle cases where img_metas[0] or aug_param might be numpy arrays
        try:
            # Use try-except instead of 'in' operator to avoid numpy array boolean ambiguity
            aug_param = None
            try:
                if hasattr(img_metas[0], "get"):
                    aug_param = img_metas[0].get("aug_param")
                else:
                    # If it's not a dict-like object, try direct access
                    aug_param = img_metas[0]["aug_param"]
            except (KeyError, TypeError, AttributeError):
                pass

            if aug_param is not None:
                crop_param = None
                try:
                    if hasattr(aug_param, "get"):
                        crop_param = aug_param.get("CropResizeFlipImage_param")
                    else:
                        crop_param = aug_param["CropResizeFlipImage_param"]
                except (KeyError, TypeError, AttributeError):
                    pass

                if crop_param is not None:
                    flip_value = crop_param[-1] if isinstance(crop_param, (list, tuple)) else crop_param
                    # Handle numpy array or tensor by extracting scalar value
                    # This prevents "ValueError: The truth value of an array with more than one element is ambiguous"
                    if hasattr(flip_value, "item"):  # PyTorch tensor or numpy scalar array
                        flip_value = flip_value.item()
                    elif hasattr(flip_value, "__array__"):  # numpy array
                        import numpy as np

                        arr = np.asarray(flip_value)
                        flip_value = bool(arr.item() if arr.size == 1 else arr.any())
                    else:
                        flip_value = bool(flip_value)
                    # Check if flip is True
                    if flip_value:
                        # flip feature
                        img_feats = [
                            torch.flip(
                                x,
                                dims=[
                                    -1,
                                ],
                            )
                            for x in img_feats
                        ]
        except (KeyError, TypeError, ValueError, AttributeError, IndexError):
            # If aug_param doesn't exist or has unexpected structure, skip flip
            pass
        return img_feats

    def forward_pts_train(self, pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore=None, prev_bev=None):
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_mono_train(self, img_feats, mono_input_dict):
        """
        img_feats (list[Tensor]): 5-D tensor for each level, (B, N, C, H, W)
        gt_bboxes (list[list[Tensor]]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        gt_labels (list[list[Tensor]]): class indices corresponding to each box
        gt_bboxes_3d (list[list[[Tensor]]): 3D boxes ground truth with shape of
                (num_gts, code_size).
        gt_labels_3d (list[list[Tensor]]): same as gt_labels
        centers2d (list[list[Tensor]]): 2D centers on the image with shape of
                (num_gts, 2).
        depths (list[list[Tensor]]): Depth ground truth with shape of
                (num_gts, ).
        attr_labels (list[list[Tensor]]): Attributes indices of each box.
        img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        ann_idx (list[list[idx]]): indicate which image has mono annotation.
        """
        bsz = img_feats[0].shape[0]
        num_lvls = len(img_feats)

        img_feats_select = [[] for lvl in range(num_lvls)]
        for lvl, img_feat in enumerate(img_feats):
            for i in range(bsz):
                img_feats_select[lvl].append(img_feat[i, mono_input_dict["mono_ann_idx"][i]])
            img_feats_select[lvl] = torch.cat(img_feats_select[lvl], dim=0)
        bsz_new = img_feats_select[0].shape[0]
        assert bsz == len(mono_input_dict["mono_input_dict"])
        input_dict = []
        for i in range(bsz):
            input_dict.extend(mono_input_dict["mono_input_dict"][i])
        assert bsz_new == len(input_dict)
        losses = self.fcos3d_bbox_head.forward_train(img_feats_select, input_dict)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, img_dict, img_metas_dict):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated."""
        # Modify: roll back to previous version for single frame
        is_training = self.training
        self.eval()
        prev_bev = OrderedDict({i: None for i in self.frames})
        with torch.no_grad():
            for t in img_dict.keys():
                img = img_dict[t]
                img_metas = [
                    img_metas_dict[t],
                ]
                img_feats = self.extract_feat(img=img, img_metas=img_metas)
                if self.num_levels:
                    img_feats = img_feats[: self.num_levels]
                bev = self.pts_bbox_head(img_feats, img_metas, None, only_bev=True)
                prev_bev[t] = bev.detach()
        if is_training:
            self.train()
        return list(prev_bev.values())

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        img=None,
        gt_bboxes_ignore=None,
        **mono_input_dict,
    ):
        img_metas = OrderedDict(sorted(img_metas[0].items()))
        img_dict = {}
        # num_images = img.shape[1] if len(img.shape) > 1 else 1
        # img_metas_keys = list(img_metas.keys())
        for ind, t in enumerate(img_metas.keys()):
            # Use the first image if there are fewer images than timestamp keys
            # img_idx = min(ind, num_images - 1)
            # img_dict[t] = img[:, img_idx, ...]
            img_dict[t] = img[:, ind, ...]

        # Get the first key from sorted img_metas
        # first_key = img_metas_keys[0]
        # img = img_dict[first_key]
        # img_dict.pop(first_key)
        img = img_dict[0]
        img_dict.pop(0)

        prev_img_metas = copy.deepcopy(img_metas)
        # prev_img_metas.pop(first_key)
        prev_img_metas.pop(0)
        prev_bev = self.obtain_history_bev(img_dict, prev_img_metas)

        img_metas = [
            img_metas[0],
        ]

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats if self.num_levels is None else img_feats[: self.num_levels],
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
            prev_bev,
        )
        losses.update(losses_pts)

        if self.fcos3d_bbox_head:
            losses_mono = self.forward_mono_train(
                img_feats=img_feats if self.num_mono_levels is None else img_feats[: self.num_mono_levels],
                mono_input_dict=mono_input_dict,
            )
            for k, v in losses_mono.items():
                losses[f"{k}_mono"] = v * self.mono_loss_weight

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        if not isinstance(img, list):
            img = [img]

        # Reshape image tensor from [1, 1, 6, 640, 1600, 3] (NHWC) to [1, 6, 3, 640, 1600] (NCHW)
        if img[0] is not None and img[0].dim() == 6:
            # Shape: [B, 1, N, H, W, C] -> [B, N, C, H, W]
            img[0] = img[0].squeeze(1).permute(0, 1, 4, 2, 3)  # [B, N, H, W, C] -> [B, N, C, H, W]
        # print(img[0].shape)
        # print(img_metas[0])
        new_prev_bev, bbox_results = self.simple_test(img_metas[0], img[0], prev_bev=None, **kwargs)
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return outs["bev_embed"], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        img_metas = OrderedDict(sorted(img_metas[0].items()))
        img_dict = {}
        num_images = img.shape[1] if len(img.shape) > 1 else 1
        img_metas_keys = list(img_metas.keys())
        for ind, t in enumerate(img_metas_keys):
            # Use the first image if there are fewer images than timestamp keys
            img_idx = min(ind, num_images - 1)
            img_dict[t] = img[:, img_idx, ...]
        # Get the first key from sorted img_metas
        first_key = img_metas_keys[0]
        img = img_dict[first_key]
        img_dict.pop(first_key)

        prev_img_metas = copy.deepcopy(img_metas)
        prev_img_metas.pop(first_key)
        prev_bev = self.obtain_history_bev(img_dict, prev_img_metas)

        img_metas = [
            img_metas[first_key],
        ]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if self.num_levels:
            img_feats = img_feats[: self.num_levels]

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return new_prev_bev, bbox_list
