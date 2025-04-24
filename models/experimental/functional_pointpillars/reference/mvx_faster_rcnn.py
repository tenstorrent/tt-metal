# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Sequence, Dict

from torch import Tensor
from torch import nn as nn

from models.experimental.functional_pointpillars.reference.hard_vfe import HardVFE
from models.experimental.functional_pointpillars.reference.point_pillars_scatter import PointPillarsScatter
from models.experimental.functional_pointpillars.reference.second import SECOND
from models.experimental.functional_pointpillars.reference.fpn import FPN
from models.experimental.functional_pointpillars.reference.anchor3d_head import Anchor3DHead
from mmengine.structures import InstanceData


class MVXFasterRCNN(nn.Module):
    def __init__(
        self,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        pts_bbox_head: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        img_bbox_head=None,
        img_backbone=None,
    ):
        super(MVXFasterRCNN, self).__init__()
        # init_cfg=init_cfg, data_preprocessor=data_preprocessor, **kwargs)

        if pts_voxel_encoder:
            self.pts_voxel_encoder = HardVFE(
                in_channels=4,
                feat_channels=[64, 64],
                point_cloud_range=[-50, -50, -5, 50, 50, 3],
                with_distance=False,
                voxel_size=[0.25, 0.25, 8],
                with_cluster_center=True,
                with_voxel_center=True,
                norm_cfg={"type": "BN1d", "eps": 0.001, "momentum": 0.01},
            )
        if pts_middle_encoder:
            self.pts_middle_encoder = PointPillarsScatter(in_channels=64, output_shape=[400, 400])
        if pts_backbone:
            self.pts_backbone = SECOND(
                in_channels=64,
                norm_cfg={"type": "BN2d", "eps": 0.001, "momentum": 0.01},
                layer_nums=[3, 5, 5],
                layer_strides=[2, 2, 2],
                out_channels=[64, 128, 256],
            )

        if pts_neck is not None:
            self.pts_neck = FPN(
                norm_cfg={"type": "BN2d", "eps": 0.001, "momentum": 0.01},
                act_cfg={"type": "ReLU"},
                in_channels=[64, 128, 256],
                out_channels=256,
                start_level=0,
                num_outs=3,
            )
        if pts_bbox_head:
            # pts_train_cfg = train_cfg.pts if train_cfg else None
            # pts_bbox_head.update(train_cfg=pts_train_cfg)
            # pts_test_cfg = test_cfg.pts if test_cfg else None
            # pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = Anchor3DHead(
                num_classes=10,
                in_channels=256,
                feat_channels=256,
                use_direction_classifier=True,
                anchor_generator={
                    "type": "AlignedAnchor3DRangeGenerator",
                    "ranges": [[-50, -50, -1.8, 50, 50, -1.8]],
                    "scales": [1, 2, 4],
                    "sizes": [[2.5981, 0.866, 1.0], [1.7321, 0.5774, 1.0], [1.0, 1.0, 1.0], [0.4, 0.4, 1]],
                    "custom_values": [0, 0],
                    "rotations": [0, 1.57],
                    "reshape_out": True,
                },
                assigner_per_size=False,
                diff_rad_by_sin=True,
                dir_offset=-0.7854,
                bbox_coder={"type": "DeltaXYZWLHRBBoxCoder", "code_size": 9},
                loss_cls={
                    "type": "mmdet.FocalLoss",
                    "use_sigmoid": True,
                    "gamma": 2.0,
                    "alpha": 0.25,
                    "loss_weight": 1.0,
                },
                loss_bbox={"type": "mmdet.SmoothL1Loss", "beta": 0.1111111111111111, "loss_weight": 1.0},
                loss_dir={"type": "mmdet.CrossEntropyLoss", "use_sigmoid": False, "loss_weight": 0.2},
                test_cfg={
                    "use_rotate_nms": True,
                    "nms_across_levels": False,
                    "nms_pre": 1000,
                    "nms_thr": 0.2,
                    "score_thr": 0.05,
                    "min_bbox_size": 0,
                    "max_num": 500,
                },
            )

        self.train_cfg = train_cfg
        self.img_backbone = None
        self.img_neck = None
        self.test_cfg = {
            "use_rotate_nms": True,
            "nms_across_levels": False,
            "nms_pre": 1000,
            "nms_thr": 0.2,
            "score_thr": 0.05,
            "min_bbox_size": 0,
            "max_num": 500,
        }

    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self, "img_shared_head") and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self, "pts_bbox_head") and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self, "img_bbox_head") and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, "img_backbone") and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, "pts_backbone") and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self, "pts_fusion_layer") and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, "img_neck") and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, "pts_neck") and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, "img_rpn_head") and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, "img_roi_head") and self.img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self, "voxel_encoder") and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(self, "middle_encoder") and self.middle_encoder is not None

    def _forward(self):
        pass

    def extract_img_feat(self, img: Tensor, input_metas: List[dict]) -> dict:
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in input_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(
        self,
        voxel_dict: Dict[str, Tensor],
        points: Optional[List[Tensor]] = None,
        img_feats: Optional[Sequence[Tensor]] = None,
        batch_input_metas: Optional[List[dict]] = None,
    ) -> Sequence[Tensor]:
        """Extract features of points.

        Args:
            voxel_dict(Dict[str, Tensor]): Dict of voxelization infos.
            points (List[tensor], optional):  Point cloud of multiple inputs.
            img_feats (list[Tensor], tuple[tensor], optional): Features from
                image backbone.
            batch_input_metas (list[dict], optional): The meta information
                of multiple samples. Defaults to True.

        Returns:
            Sequence[tensor]: points features of multiple inputs
            from backbone or neck.
        """
        if not self.with_pts_bbox:
            return None
        voxel_features = self.pts_voxel_encoder(
            voxel_dict["voxels"], voxel_dict["num_points"], voxel_dict["coors"], img_feats, batch_input_metas
        )
        batch_size = voxel_dict["coors"][-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, voxel_dict["coors"], batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, batch_inputs_dict: dict, batch_input_metas: List[dict]) -> tuple:
        """Extract features from images and points.

        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains

                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """
        voxel_dict = batch_inputs_dict.get("voxels", None)
        imgs = batch_inputs_dict.get("imgs", None)
        points = batch_inputs_dict.get("points", None)
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        pts_feats = self.extract_pts_feat(
            voxel_dict, points=points, img_feats=img_feats, batch_input_metas=batch_input_metas
        )
        return (img_feats, pts_feats)

    # def loss(self, batch_inputs_dict: Dict[List, torch.Tensor],
    #          batch_data_samples: List[Det3DDataSample],
    #          **kwargs) -> List[Det3DDataSample]:
    #     """
    #     Args:
    #         batch_inputs_dict (dict): The model input dict which include
    #             'points' and `imgs` keys.

    #             - points (list[torch.Tensor]): Point cloud of each sample.
    #             - imgs (torch.Tensor): Tensor of batch images, has shape
    #               (B, C, H ,W)
    #         batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
    #             Samples. It usually includes information such as
    #             `gt_instance_3d`, .

    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.

    #     """

    #     batch_input_metas = [item.metainfo for item in batch_data_samples]
    #     img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
    #                                              batch_input_metas)
    #     losses = dict()
    #     if pts_feats:
    #         losses_pts = self.pts_bbox_head.loss(pts_feats, batch_data_samples,
    #                                              **kwargs)
    #         losses.update(losses_pts)
    #     if img_feats:
    #         losses_img = self.loss_imgs(img_feats, batch_data_samples)
    #         losses.update(losses_img)
    #     return losses

    # def loss_imgs(self, x: List[Tensor],
    #               batch_data_samples: List[Det3DDataSample], **kwargs):
    #     """Forward function for image branch.

    #     This function works similar to the forward function of Faster R-CNN.

    #     Args:
    #         x (list[torch.Tensor]): Image features of shape (B, C, H, W)
    #             of multiple levels.
    #         batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
    #             Samples. It usually includes information such as
    #             `gt_instance_3d`, .

    #     Returns:
    #         dict: Losses of each branch.
    #     """
    #     losses = dict()
    #     # RPN forward and loss
    #     if self.with_img_rpn:
    #         proposal_cfg = self.test_cfg.rpn
    #         rpn_data_samples = copy.deepcopy(batch_data_samples)
    #         # set cat_id of gt_labels to 0 in RPN
    #         for data_sample in rpn_data_samples:
    #             data_sample.gt_instances.labels = \
    #                 torch.zeros_like(data_sample.gt_instances.labels)
    #         rpn_losses, rpn_results_list = self.img_rpn_head.loss_and_predict(
    #             x, rpn_data_samples, proposal_cfg=proposal_cfg, **kwargs)
    #         # avoid get same name with roi_head loss
    #         keys = rpn_losses.keys()
    #         for key in keys:
    #             if 'loss' in key and 'rpn' not in key:
    #                 rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
    #         losses.update(rpn_losses)

    #     else:
    #         if 'proposals' in batch_data_samples[0]:
    #             # use pre-defined proposals in InstanceData
    #             # for the second stage
    #             # to extract ROI features.
    #             rpn_results_list = [
    #                 data_sample.proposals for data_sample in batch_data_samples
    #             ]
    #         else:
    #             rpn_results_list = None
    #     # bbox head forward and loss
    #     if self.with_img_bbox:
    #         roi_losses = self.img_roi_head.loss(x, rpn_results_list,
    #                                             batch_data_samples, **kwargs)
    #         losses.update(roi_losses)
    #     return losses

    def add_pred_to_datasample(
        self,
        data_samples,
        data_instances_3d=None,
        data_instances_2d=None,
    ):
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or (
            data_instances_3d is not None
        ), "please pass at least one type of data_samples"

        if data_instances_2d is None:
            data_instances_2d = [InstanceData() for _ in range(len(data_instances_3d))]
        if data_instances_3d is None:
            data_instances_3d = [InstanceData() for _ in range(len(data_instances_2d))]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples

    def predict_imgs(self, x: List[Tensor], batch_data_samples, rescale: bool = True, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            x (List[Tensor]): Image features from FPN.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.
        """

        if batch_data_samples[0].get("proposals", None) is None:
            rpn_results_list = self.img_rpn_head.predict(x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]
        results_list = self.img_roi_head.predict(x, rpn_results_list, batch_data_samples, rescale=rescale, **kwargs)
        return results_list

    def forward(self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples, **kwargs):
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        # batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_metas = batch_data_samples  # modified and passed
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        if pts_feats and self.with_pts_bbox:
            outs = self.pts_bbox_head(pts_feats)
        else:
            outs = None

        if img_feats and self.with_img_bbox:
            # TODO check this for camera modality
            results_list_2d = self.predict_imgs(img_feats, batch_data_samples, **kwargs)
        else:
            results_list_2d = None

        # print("results_list_3d", results_list_3d)
        # detsamples = self.add_pred_to_datasample(batch_data_samples, results_list_3d, results_list_2d)
        return outs
