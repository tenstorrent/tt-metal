# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch
from torch import nn
from models.experimental.functional_petr.reference.petr_head import PETRHead
from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP
from models.experimental.functional_petr.reference.cp_fpn import CPFPN
from models.experimental.functional_petr.reference.utils import bbox3d2result
from .grid_mask import GridMask


class PETR(nn.Module):
    """PETR."""

    def __init__(
        self,
        use_grid_mask=False,
    ):
        super(PETR, self).__init__()
        self.with_img_neck = True
        self.pts_bbox_head = PETRHead(
            num_classes=10,
            in_channels=256,
            num_query=900,
            LID=True,
            with_position=True,
            with_multiview=True,
            position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            normedlinear=False,
            test_cfg=None,
        )
        self.img_backbone = VoVNetCP(
            spec_name="V-99-eSE", norm_eval=True, frozen_stages=-1, input_ch=3, out_features=("stage4", "stage5")
        )
        self.img_neck = CPFPN(in_channels=[768, 1024], out_channels=256, num_outs=2)

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.head_outs = None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    # @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def predict(self, inputs=None, data_samples=None, mode=None, skip_post_processing=False, **kwargs):
        img = inputs["imgs"]
        batch_img_metas = data_samples  # the above is preprocessed outside
        for var, name in [(batch_img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        batch_img_metas = self.add_lidar2img(img, batch_img_metas)

        if skip_post_processing:
            return self.simple_test(batch_img_metas, img, skip_post_processing, **kwargs)

        results_list_3d = self.simple_test(batch_img_metas, img, **kwargs)

        return results_list_3d

    def simple_test_pts(self, x, img_metas, skip_post_processing=False, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        # print("outs", outs)
        if skip_post_processing:
            return outs
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return bbox_results

    def simple_test(self, img_metas, img=None, skip_post_processing=False, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if skip_post_processing:
            return self.simple_test_pts(
                img_feats, img_metas, skip_post_processing=skip_post_processing, rescale=rescale
            )
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        # print("bbox_pts", bbox_pts)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return bbox_list

    # may need speed-up
    def add_lidar2img(self, img, batch_input_metas):
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        if isinstance(img, list):
            img = img[0]
        else:
            img = img

        for meta in batch_input_metas:
            lidar2img_rts = []
            # obtain lidar to image transformation matrix
            for i in range(len(meta["cam2img"])):
                lidar2cam_rt = torch.tensor(meta["lidar2cam"][i]).double()
                intrinsic = torch.tensor(meta["cam2img"][i]).double()
                viewpad = torch.eye(4).double()
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt
                # The extrinsics mean the transformation from lidar to camera.
                # If anyone want to use the extrinsics as sensor to lidar,
                # please use np.linalg.inv(lidar2cam_rt.T)
                # and modify the ResizeCropFlipImage
                # and LoadMultiViewImageFromMultiSweepsFiles.
                lidar2img_rts.append(lidar2img_rt)
            meta["lidar2img"] = lidar2img_rts
            img_shape = meta["img_shape"][:3]
            meta["img_shape"] = [img_shape] * len(img[0])

        return batch_input_metas
