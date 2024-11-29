# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

# from mmengine.structures import InstanceData
from torch import nn

# from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
from models.experimental.functional_petr.reference.petr_head import PETRHead
from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP
from models.experimental.functional_petr.reference.cp_fpn import CPFPN

# from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
# from mmdet3d.registry import MODELS
# from mmdet3d.structures.ops import bbox3d2result
from models.experimental.functional_petr.reference.utils import bbox3d2result

from .grid_mask import GridMask


# @MODELS.register_module()
class PETR(nn.Module):
    """PETR."""

    def __init__(
        self,
        use_grid_mask=False,
    ):
        super(PETR, self).__init__()
        # super(PETR, self).__init__(
        #     pts_voxel_layer,
        #     pts_middle_encoder,
        #     pts_fusion_layer,
        #     img_backbone,
        #     pts_backbone,
        #     img_neck,
        #     pts_neck,
        #     pts_bbox_head,
        #     img_roi_head,
        #     img_rpn_head,
        #     train_cfg,
        #     test_cfg,
        #     init_cfg,
        #     data_preprocessor,
        # )
        self.with_img_neck = True
        # self.data_preprocessor = Det3DDataPreprocessor(
        #     mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], bgr_to_rgb=False, pad_size_divisor=32
        # )
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

    def predict(self, inputs=None, data_samples=None, mode=None, **kwargs):
        img = inputs["imgs"]
        # batch_img_metas = [ds.metainfo for ds in data_samples]
        batch_img_metas = data_samples  # the above is preprocessed outside
        for var, name in [(batch_img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        batch_img_metas = self.add_lidar2img(img, batch_img_metas)

        results_list_3d = self.simple_test(batch_img_metas, img, **kwargs)

        # for i, data_sample in enumerate(data_samples):
        #     results_list_3d_i = InstanceData(metainfo=results_list_3d[i]["pts_bbox"])
        #     data_sample.pred_instances_3d = results_list_3d_i
        #     data_sample.pred_instances = InstanceData()

        return results_list_3d
        # return data_sample

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return bbox_results

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
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
