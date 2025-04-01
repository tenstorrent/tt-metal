# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch import nn as nn

from models.experimental.functional_pointpillars.reference.point_pillars_utils import (
    Anchor3DRangeGenerator,
    DeltaXYZWLHRBBoxCoder,
    multi_apply,
    Base3DDenseHead,
)


class AlignedAnchor3DRangeGenerator(Anchor3DRangeGenerator):
    def __init__(self, align_corner: bool = False, **kwargs) -> None:
        super(AlignedAnchor3DRangeGenerator, self).__init__(**kwargs)
        self.align_corner = align_corner

    def anchors_single_range(
        self,
        feature_size: List[int],
        anchor_range: List[float],
        scale: int,
        sizes: Union[List[List[float]], List[float]] = [[3.9, 1.6, 1.56]],
        rotations: List[float] = [0, 1.5707963],
        device: Union[str, torch.device] = "cuda",
    ) -> Tensor:
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = torch.tensor(anchor_range, device=device)
        z_centers = torch.linspace(anchor_range[2], anchor_range[5], feature_size[0] + 1, device=device)
        y_centers = torch.linspace(anchor_range[1], anchor_range[4], feature_size[1] + 1, device=device)
        x_centers = torch.linspace(anchor_range[0], anchor_range[3], feature_size[2] + 1, device=device)
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3) * scale
        rotations = torch.tensor(rotations, device=device)

        # shift the anchor center
        if not self.align_corner:
            z_shift = (z_centers[1] - z_centers[0]) / 2
            y_shift = (y_centers[1] - y_centers[0]) / 2
            x_shift = (x_centers[1] - x_centers[0]) / 2
            z_centers += z_shift
            y_centers += y_shift
            x_centers += x_shift

        # torch.meshgrid default behavior is 'id', np's default is 'xy'
        rets = torch.meshgrid(
            x_centers[: feature_size[2]], y_centers[: feature_size[1]], z_centers[: feature_size[0]], rotations
        )

        # torch.meshgrid returns a tuple rather than list
        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)

        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])

        if len(self.custom_values) > 0:
            custom_ndim = len(self.custom_values)
            custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
            # TODO: check the support of custom values
            # custom[:] = self.custom_values
            ret = torch.cat([ret, custom], dim=-1)
        return ret


class Anchor3DHead(Base3DDenseHead):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        use_direction_classifier: bool = True,
        anchor_generator=dict(
            type="Anchor3DRangeGenerator",
            range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
            strides=[2],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            custom_values=[],
            reshape_out=False,
        ),
        assigner_per_size: bool = False,
        assign_per_class: bool = False,
        diff_rad_by_sin: bool = True,
        dir_offset: float = -np.pi / 2,
        dir_limit_offset: int = 0,
        bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder"),
        loss_cls=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="mmdet.SmoothL1Loss", beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type="mmdet.CrossEntropyLoss", loss_weight=0.2),
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        warnings.warn(
            "dir_offset and dir_limit_offset will be depressed and be " "incorporated into box coder in the future"
        )

        # build anchor generator
        self.prior_generator = AlignedAnchor3DRangeGenerator(
            ranges=anchor_generator["ranges"],
            scales=anchor_generator["scales"],
            sizes=anchor_generator["sizes"],
            custom_values=anchor_generator["custom_values"],
            rotations=anchor_generator["rotations"],
            reshape_out=anchor_generator["reshape_out"],
        )
        # In 3D detection, the anchor stride is connected with anchor size
        self.num_anchors = self.prior_generator.num_base_anchors
        # build box coder
        self.bbox_coder = DeltaXYZWLHRBBoxCoder(code_size=bbox_coder["code_size"])
        self.box_code_size = self.bbox_coder.code_size

        # build loss function
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        self.sampling = loss_cls["type"] not in ["mmdet.FocalLoss", "mmdet.GHMC"]
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        # self.loss_cls = MODELS.build(loss_cls) #Not used should check
        # self.loss_bbox = MODELS.build(loss_bbox)  #Not used should check
        # self.loss_dir = MODELS.build(loss_dir) #Not used should check

        self._init_layers()
        self._init_assigner_sampler()

        if init_cfg is None:
            self.init_cfg = dict(
                type="Normal",
                layer="Conv2d",
                std=0.01,
                override=dict(type="Normal", name="conv_cls", std=0.01, bias_prob=0.01),
            )

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:  # This is invoked
            return

        # if self.sampling:
        #     self.bbox_sampler = TASK_UTILS.build(self.train_cfg.sampler)
        # else:
        #     self.bbox_sampler = PseudoSampler()
        # if isinstance(self.train_cfg.assigner, dict):
        #     self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
        # elif isinstance(self.train_cfg.assigner, list):
        #     self.bbox_assigner = [
        #         TASK_UTILS.build(res) for res in self.train_cfg.assigner
        #     ]

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * self.box_code_size, 1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(self.feat_channels, self.num_anchors * 2, 1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_pred = None
        if self.use_direction_classifier:
            dir_cls_pred = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_pred

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        return multi_apply(self.forward_single, x)

    # # TODO: Support augmentation test
    # def aug_test(self,
    #              aug_batch_feats,
    #              aug_batch_input_metas,
    #              rescale=False,
    #              **kwargs):
    #     aug_bboxes = []
    #     # only support aug_test for one sample
    #     for x, input_meta in zip(aug_batch_feats, aug_batch_input_metas):
    #         outs = self.forward(x)
    #         bbox_list = self.get_results(*outs, [input_meta], rescale=rescale)
    #         bbox_dict = dict(
    #             bboxes_3d=bbox_list[0].bboxes_3d,
    #             scores_3d=bbox_list[0].scores_3d,
    #             labels_3d=bbox_list[0].labels_3d)
    #         aug_bboxes.append(bbox_dict)
    #     # after merging, bboxes will be rescaled to the original image size
    #     merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, aug_batch_input_metas,
    #                                         self.test_cfg)
    #     return [merged_bboxes]

    # def get_anchors(self,
    #                 featmap_sizes: List[tuple],
    #                 input_metas: List[dict],
    #                 device: str = 'cuda') -> list:
    #     """Get anchors according to feature map sizes.

    #     Args:
    #         featmap_sizes (list[tuple]): Multi-level feature map sizes.
    #         input_metas (list[dict]): contain pcd and img's meta info.
    #         device (str): device of current module.

    #     Returns:
    #         list[list[torch.Tensor]]: Anchors of each image, valid flags
    #             of each image.
    #     """
    #     num_imgs = len(input_metas)
    #     # since feature map sizes of all images are the same, we only compute
    #     # anchors for one time
    #     multi_level_anchors = self.prior_generator.grid_anchors(
    #         featmap_sizes, device=device)
    #     anchor_list = [multi_level_anchors for _ in range(num_imgs)]
    #     return anchor_list
