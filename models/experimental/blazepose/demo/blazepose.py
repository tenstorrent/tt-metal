# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn

from models.experimental.blazepose.demo.blazebase import BlazeDetector, BlazeBlock


class BlazePose(BlazeDetector):
    """The BlazePose pose detection model from MediaPipe.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/google/mediapipe/
    """

    def __init__(self):
        super(BlazePose, self).__init__()

        # These are the settings from the MediaPipe example graph
        # mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 12
        self.score_clipping_thresh = 100.0
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
        self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3
        self.num_keypoints = 4

        # These settings are for converting detections to ROIs which can then
        # be extracted and feed into the landmark network
        # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
        self.detection2roi_method = "alignment"
        # mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
        self.kp1 = 2
        self.kp2 = 3
        self.theta0 = 90 * np.pi / 180
        self.dscale = 1.5
        self.dy = 0.0

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=48,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 64, 5, 2, skip_proj=True),
            BlazeBlock(64, 64, 5),
            BlazeBlock(64, 64, 5),
            BlazeBlock(64, 64, 5),
            BlazeBlock(64, 96, 5, 2, skip_proj=True),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 128, 5, 2, skip_proj=True),
            BlazeBlock(128, 128, 5),
            BlazeBlock(128, 128, 5),
            BlazeBlock(128, 128, 5),
            BlazeBlock(128, 128, 5),
            BlazeBlock(128, 128, 5),
            BlazeBlock(128, 128, 5),
            BlazeBlock(128, 128, 5),
        )

        self.backbone2 = nn.Sequential(
            BlazeBlock(128, 256, 5, 2, skip_proj=True),
            BlazeBlock(256, 256, 5),
            BlazeBlock(256, 256, 5),
            BlazeBlock(256, 256, 5),
            BlazeBlock(256, 256, 5),
            BlazeBlock(256, 256, 5),
            BlazeBlock(256, 256, 5),
        )

        self.classifier_8 = nn.Conv2d(128, 2, 1, bias=True)
        self.classifier_16 = nn.Conv2d(256, 6, 1, bias=True)

        self.regressor_8 = nn.Conv2d(128, 24, 1, bias=True)
        self.regressor_16 = nn.Conv2d(256, 72, 1, bias=True)

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        # x = F.pad(x, (1, 2, 1, 2), "constant", 0)

        b = x.shape[0]  # batch size, needed for reshaping later

        x = self.backbone1(x)
        h = self.backbone2(x)

        # Note: Because PyTorch is NCHW but TFLite is NHWC, we need to
        # permute the output from the conv layers before reshaping it.

        c1 = self.classifier_8(x)
        # print(c1)
        # print(c1.shape)

        c1 = c1.permute(0, 2, 3, 1)
        c1 = c1.reshape(b, -1, 1)

        c2 = self.classifier_16(h)
        c2 = c2.permute(0, 2, 3, 1)
        c2 = c2.reshape(b, -1, 1)

        c = torch.cat((c1, c2), dim=1)

        r1 = self.regressor_8(x)
        r1 = r1.permute(0, 2, 3, 1)
        r1 = r1.reshape(b, -1, 12)

        r2 = self.regressor_16(h)
        r2 = r2.permute(0, 2, 3, 1)
        r2 = r2.reshape(b, -1, 12)

        r = torch.cat((r1, r2), dim=1)
        return [r, c]
