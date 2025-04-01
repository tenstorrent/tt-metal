# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np

from models.experimental.functional_pointpillars.tt.ttnn_point_pillars_utils import (
    TtDeltaXYZWLHRBBoxCoder,
    multi_apply,
    TtBase3DDenseHead,
    TtAlignedAnchor3DRangeGenerator,
)

from models.experimental.functional_pointpillars.tt.common import TtConv


class TtAnchor3DHead(TtBase3DDenseHead):
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
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        parameters=None,
        device=None,
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

        self.parameters = parameters
        self.device = device
        warnings.warn(
            "dir_offset and dir_limit_offset will be depressed and be " "incorporated into box coder in the future"
        )

        # build anchor generator
        self.prior_generator = TtAlignedAnchor3DRangeGenerator(
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
        self.bbox_coder = TtDeltaXYZWLHRBBoxCoder(code_size=bbox_coder["code_size"])
        self.box_code_size = self.bbox_coder.code_size

        # build loss function
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        self.sampling = loss_cls["type"] not in ["mmdet.FocalLoss", "mmdet.GHMC"]
        if not self.use_sigmoid_cls:
            self.num_classes += 1

        self._init_layers()
        # self._init_assigner_sampler()

        if init_cfg is None:
            self.init_cfg = dict(
                type="Normal",
                layer="Conv2d",
                std=0.01,
                override=dict(type="Normal", name="conv_cls", std=0.01, bias_prob=0.01),
            )

    # def _init_assigner_sampler(self):
    #     """Initialize the target assigner and sampler of the head."""
    #     if self.train_cfg is None:  # This is invoked
    #         return

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = TtConv(
            parameters=self.parameters["conv_cls"],
            device=self.device,
            input_params=[1, 1, 0, self.cls_out_channels, self.feat_channels],
            reshape_tensor=True,
        )
        self.conv_reg = TtConv(
            parameters=self.parameters["conv_reg"],
            device=self.device,
            input_params=[1, 1, 0, self.num_anchors * self.box_code_size, self.feat_channels],
            reshape_tensor=True,
        )
        if self.use_direction_classifier:
            self.conv_dir_cls = TtConv(
                parameters=self.parameters["conv_dir_cls"],
                device=self.device,
                input_params=[1, 1, 0, self.num_anchors * 2, self.feat_channels],
                reshape_tensor=True,
            )

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_pred = None
        if self.use_direction_classifier:
            dir_cls_pred = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_pred

    def forward(self, x):
        return multi_apply(self.forward_single, x)
