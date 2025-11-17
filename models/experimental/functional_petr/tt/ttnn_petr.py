# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

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

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.shape[0]
        if img is not None:
            # input_shape = img.shape[-2:]
            input_shape = tuple((img.shape[-2], img.shape[-1]))

            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if len(img.shape) == 5:
                if img.shape[0] == 1 and img.shape[1] != 1:
                    img = ttnn.reshape(img, (img.shape[1], img.shape[2], img.shape[3], img.shape[4]))
                else:
                    # This is not invoked in our run
                    B, N, C, H, W = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]
                    img = ttnn.reshape(img, (B * N, C, H, W))
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(
                device=self.device, x=ttnn.permute(img, (0, 2, 3, 1))
            )  # permute is done to change the input from NCHW to NHWC
            if isinstance(img_feats, dict):
                # This is not invoked in our run
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(device=self.device, inputs=img_feats)

        # for i in range(len(img_feats)): #converting img_neck output from NHWC to NCHW
        #     img_feats[i]=ttnn.permute(img_feats[i],(0,3,1,2))
        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = ttnn.permute(img_feat, (0, 3, 1, 2))  # converting img_neck output from NHWC to NCHW
            BN, C, H, W = img_feat.shape[0], img_feat.shape[1], img_feat.shape[2], img_feat.shape[3]
            img_feats_reshaped.append(ttnn.reshape(img_feat, (B, int(BN / B), C, H, W)))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def predict(self, inputs=None, data_samples=None, **kwargs):
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
        outs = self.pts_bbox_head(x, img_metas, device=self.device)
        for i in outs.keys():
            if i in ["all_cls_scores", "all_bbox_preds"]:
                outs[i] = ttnn.to_torch(outs[i])
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
                lidar2img_rts.append(lidar2img_rt)
            meta["lidar2img"] = lidar2img_rts
            img_shape = meta["img_shape"][:3]
            meta["img_shape"] = img_shape * img.shape[1]

        return batch_input_metas
