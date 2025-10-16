# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.functional_petr.tt.ttnn_petr_head import ttnn_PETRHead
from models.experimental.functional_petr.tt.ttnn_vovnetcp import ttnn_VoVNetCP
from models.experimental.functional_petr.tt.ttnn_cp_fpn import ttnn_CPFPN
from models.experimental.functional_petr.tt.ttnn_grid_mask import ttnn_GridMask

from models.experimental.functional_petr.reference.utils import bbox3d2result


class ttnn_PETR:
    """PETR."""

    def __init__(
        self,
        use_grid_mask=False,
        parameters=None,
        query_embedding_input=None,
        device=None,
    ):
        self.with_img_neck = True
        self.pts_bbox_head = ttnn_PETRHead(
            num_classes=10,
            in_channels=256,
            num_query=900,
            LID=True,
            with_position=True,
            with_multiview=True,
            position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            parameters=parameters["pts_bbox_head"],
            device=device,
            query_embedding_input=query_embedding_input,
        )
        self.img_backbone = ttnn_VoVNetCP(
            parameters=parameters["img_backbone"], stem_parameters=parameters["stem_parameters"], device=device
        )
        self.img_neck = ttnn_CPFPN(
            in_channels=[768, 1024], out_channels=256, num_outs=2, parameters=parameters["img_neck"]
        )

        self.grid_mask = ttnn_GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.device = device
        self.head_outs = None

    # def extract_img_feat(self, img, img_metas):
    #     """Extract features of images."""
    #     if isinstance(img, list):
    #         img = torch.stack(img, dim=0)

    #     B = img.shape[0]
    #     if img is not None:
    #         input_shape = tuple((img.shape[-2], img.shape[-1]))

    #         # update real input shape of each single img
    #         for img_meta in img_metas:
    #             img_meta.update(input_shape=input_shape)
    #         if len(img.shape) == 5:
    #             B, N, C, H, W = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]

    #             if B == 1 and N != 1:
    #                 img = ttnn.reshape(img, (N, C, H, W))
    #             else:
    #                 B, N, C, H, W = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]
    #                 img = ttnn.reshape(img, (B * N, C, H, W))
    #         if self.use_grid_mask:
    #             img = self.grid_mask(img)

    #         img_nhwc = ttnn.permute(img, (0, 2, 3, 1))

    #         img_feats = self.img_backbone(
    #             device=self.device, x=ttnn.permute(img, (0, 2, 3, 1))
    #         )  # permute is done to change the input from NCHW to NHWC
    #         if isinstance(img_feats, dict):
    #             img_feats = list(img_feats.values())
    #     else:
    #         return None
    #     if self.with_img_neck:
    #         img_feats = self.img_neck(device=self.device, inputs=img_feats)

    #     img_feats_reshaped = []
    #     for img_feat in img_feats:
    #         img_feat = ttnn.permute(img_feat, (0, 3, 1, 2))  # converting img_neck output from NHWC to NCHW
    #         BN, C, H, W = img_feat.shape[0], img_feat.shape[1], img_feat.shape[2], img_feat.shape[3]
    #         img_feats_reshaped.append(ttnn.reshape(img_feat, (B, int(BN / B), C, H, W)))
    #     return img_feats_reshaped

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.shape[0]
        if img is not None:
            input_shape = tuple((img.shape[-2], img.shape[-1]))

            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if len(img.shape) == 5:
                B, N, C, H, W = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]
                img = ttnn.reshape(img, (B * N, C, H, W))

            if self.use_grid_mask:
                img = self.grid_mask(img)

            # Process each camera separately to avoid L1 memory issues
            img_feats_list = []
            num_cameras = img.shape[0]

            for cam_idx in range(num_cameras):
                # Extract single camera image
                single_img = img[cam_idx : cam_idx + 1]  # Keep batch dim

                # Convert to NHWC
                single_img_nhwc = ttnn.permute(single_img, (0, 2, 3, 1))

                # Process through backbone
                single_feats = self.img_backbone(device=self.device, x=single_img_nhwc)

                img_feats_list.append(single_feats)
                # ttnn.device.dump_device_profiler(self.device)

            # Combine features from all cameras
            img_feats = []
            num_stages = len(img_feats_list[0])

            for stage_idx in range(num_stages):
                # Collect this stage's features from all cameras
                stage_feats = [img_feats_list[cam][stage_idx] for cam in range(num_cameras)]
                # Stack
                stacked_feat = ttnn.concat(stage_feats, dim=0)
                img_feats.append(stacked_feat)

        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(device=self.device, inputs=img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = ttnn.permute(img_feat, (0, 3, 1, 2))  # NHWC → NCHW
            BN, C, H, W = img_feat.shape[0], img_feat.shape[1], img_feat.shape[2], img_feat.shape[3]
            # Reshape
            img_feats_reshaped.append(ttnn.reshape(img_feat, (B, int(BN / B), C, H, W)))

        return img_feats_reshaped

    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def predict(self, inputs=None, data_samples=None, skip_post_processing=False, **kwargs):
        img = inputs["imgs"]
        batch_img_metas = data_samples
        for var, name in [(batch_img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        batch_img_metas = self.add_lidar2img(img, batch_img_metas)

        if skip_post_processing:
            return self.simple_test(batch_img_metas, img, skip_post_processing, **kwargs)

        results_list_3d = self.simple_test(batch_img_metas, img, skip_post_processing, **kwargs)

        return results_list_3d

    def simple_test_pts(self, x, img_metas, skip_post_processing=False, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas, device=self.device)
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
                lidar2img_rts.append(lidar2img_rt)
            meta["lidar2img"] = lidar2img_rts
            img_shape = meta["img_shape"][0] if isinstance(meta["img_shape"], list) else meta["img_shape"]
            num_cameras = len(meta["cam2img"])
            meta["img_shape"] = [img_shape for _ in range(num_cameras)]

        return batch_input_metas
