# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO11 Pose with RAW keypoint output (no decoding)

This version outputs raw keypoint values WITHOUT decoding them to pixel coordinates.
Used for testing TTNN implementation (which also outputs raw values).

The only difference from yolov11_pose_correct.py is that the PoseHead
does NOT decode keypoints - it outputs raw cv4 values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as f

from models.demos.yolov11.reference.yolov11 import DFL, Conv, make_anchors


class DWConv(nn.Module):
    """Depthwise Convolution"""

    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel, stride=stride, padding=padding, groups=in_channel, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PoseHeadRaw(nn.Module):
    """
    Pose Head that outputs RAW keypoints (no decoding)

    This version is for testing TTNN implementation.
    Keypoint decoding is done in postprocessing on CPU.
    """

    def __init__(self):
        super().__init__()

        # cv2: Bounding box regression head (3 scales)
        self.cv2 = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(64, 64, kernel=3, stride=1, padding=1),
                    Conv(64, 64, kernel=3, stride=1, padding=1),
                    nn.Conv2d(64, 64, 1, 1),
                ),
                nn.Sequential(
                    Conv(128, 64, kernel=3, stride=1, padding=1),
                    Conv(64, 64, kernel=3, stride=1, padding=1),
                    nn.Conv2d(64, 64, 1, 1),
                ),
                nn.Sequential(
                    Conv(256, 64, kernel=3, stride=1, padding=1),
                    Conv(64, 64, kernel=3, stride=1, padding=1),
                    nn.Conv2d(64, 64, 1, 1),
                ),
            ]
        )

        # cv3: Person confidence head (3 scales) - uses DWConv
        self.cv3 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Sequential(
                        DWConv(64, 64, kernel=3, stride=1, padding=1),
                        Conv(64, 64, kernel=1, stride=1, padding=0, enable_act=True),
                    ),
                    nn.Sequential(
                        DWConv(64, 64, kernel=3, stride=1, padding=1),
                        Conv(64, 64, kernel=1, stride=1, padding=0, enable_act=True),
                    ),
                    nn.Conv2d(64, 1, 1, 1),
                ),
                nn.Sequential(
                    nn.Sequential(
                        DWConv(128, 128, kernel=3, stride=1, padding=1),
                        Conv(128, 64, kernel=1, stride=1, padding=0, enable_act=True),
                    ),
                    nn.Sequential(
                        DWConv(64, 64, kernel=3, stride=1, padding=1),
                        Conv(64, 64, kernel=1, stride=1, padding=0, enable_act=True),
                    ),
                    nn.Conv2d(64, 1, 1, 1),
                ),
                nn.Sequential(
                    nn.Sequential(
                        DWConv(256, 256, kernel=3, stride=1, padding=1),
                        Conv(256, 64, kernel=1, stride=1, padding=0, enable_act=True),
                    ),
                    nn.Sequential(
                        DWConv(64, 64, kernel=3, stride=1, padding=1),
                        Conv(64, 64, kernel=1, stride=1, padding=0, enable_act=True),
                    ),
                    nn.Conv2d(64, 1, 1, 1),
                ),
            ]
        )

        # cv4: Keypoint prediction head (3 scales)
        self.cv4 = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(64, 51, kernel=3, stride=1, padding=1),
                    Conv(51, 51, kernel=3, stride=1, padding=1),
                    nn.Conv2d(51, 51, 1, 1),
                ),
                nn.Sequential(
                    Conv(128, 51, kernel=3, stride=1, padding=1),
                    Conv(51, 51, kernel=3, stride=1, padding=1),
                    nn.Conv2d(51, 51, 1, 1),
                ),
                nn.Sequential(
                    Conv(256, 51, kernel=3, stride=1, padding=1),
                    Conv(51, 51, kernel=3, stride=1, padding=1),
                    nn.Conv2d(51, 51, 1, 1),
                ),
            ]
        )

        self.dfl = DFL()

    def forward(self, y1, y2, y3):
        """
        Returns raw outputs WITHOUT keypoint decoding

        Output: [batch, 56, num_anchors]
        - Channels 0-3: Decoded bbox (x,y,w,h)
        - Channel 4: Sigmoid confidence
        - Channels 5-55: RAW keypoints (NOT decoded!)
        """
        # Bbox regression
        x1_bbox = self.cv2[0](y1)
        x2_bbox = self.cv2[1](y2)
        x3_bbox = self.cv2[2](y3)

        # Confidence
        x1_conf = self.cv3[0](y1)
        x2_conf = self.cv3[1](y2)
        x3_conf = self.cv3[2](y3)

        # Keypoints (raw output from cv4)
        x1_kpts = self.cv4[0](y1)
        x2_kpts = self.cv4[1](y2)
        x3_kpts = self.cv4[2](y3)

        # Concatenate
        y1 = torch.cat((x1_bbox, x1_conf, x1_kpts), 1)
        y2 = torch.cat((x2_bbox, x2_conf, x2_kpts), 1)
        y3 = torch.cat((x3_bbox, x3_conf, x3_kpts), 1)
        y_all = [y1, y2, y3]

        # Reshape to [B, C, num_anchors]
        y1 = torch.reshape(y1, (y1.shape[0], y1.shape[1], y1.shape[2] * y1.shape[3]))
        y2 = torch.reshape(y2, (y2.shape[0], y2.shape[1], y2.shape[2] * y2.shape[3]))
        y3 = torch.reshape(y3, (y3.shape[0], y3.shape[1], y3.shape[2] * y3.shape[3]))

        # Concatenate all scales
        y = torch.cat((y1, y2, y3), 2)

        # Split into bbox (64), conf (1), keypoints (51)
        ya, yb, yc = y.split((64, 1, 51), 1)

        # ===== Decode bounding boxes with DFL =====
        ya = torch.reshape(ya, (ya.shape[0], 4, 16, ya.shape[2]))
        ya = torch.permute(ya, (0, 2, 1, 3))
        ya = f.softmax(ya, dim=1)
        c = self.dfl(ya)
        c1 = torch.reshape(c, (c.shape[0], c.shape[1] * c.shape[2], c.shape[3]))
        c1_lt = c1[:, 0:2, :]
        c1_rb = c1[:, 2:4, :]

        # Generate anchors
        anchor, strides = (y_all.transpose(0, 1) for y_all in make_anchors(y_all, [8, 16, 32], 0.5))
        anchor.unsqueeze(0)

        # Decode bbox
        lt = anchor - c1_lt
        rb = anchor + c1_rb
        wh = rb - lt
        cxy = (lt + rb) / 2
        bbox = torch.concat((cxy, wh), 1) * strides

        # ===== Process confidence =====
        yb = torch.sigmoid(yb)

        # ===== Keypoints: OUTPUT RAW (NO DECODING) =====
        # Just return the raw cv4 output without decoding
        # Decoding will be done in CPU postprocessing

        # Final output: [bbox(4) + conf(1) + raw_keypoints(51)] = 56 channels
        out = torch.concat((bbox, yb, yc), 1)
        return out


# Import the rest from yolov11_pose_correct.py


class YoloV11PoseRaw(nn.Module):
    """YOLO11 Pose that outputs RAW keypoints (for TTNN comparison)"""

    def __init__(self):
        super().__init__()
        # Use the same backbone/neck structure
        from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose

        full_model = YoloV11Pose()

        # Copy all layers except the pose head
        self.model = nn.ModuleList()
        for i in range(23):
            self.model.append(full_model.model[i])

        # Use raw pose head instead
        self.model.append(PoseHeadRaw())

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.model[0](x)  # 0
        x = self.model[1](x)  # 1
        x = self.model[2](x)  # 2
        x = self.model[3](x)  # 3
        x = self.model[4](x)  # 4
        x4 = x
        x = self.model[5](x)  # 5
        x = self.model[6](x)  # 6
        x6 = x
        x = self.model[7](x)  # 7
        x = self.model[8](x)  # 8
        x = self.model[9](x)  # 9
        x = self.model[10](x)  # 10
        x10 = x
        x = f.interpolate(x, scale_factor=2.0, mode="nearest")  # 11
        x = torch.cat((x, x6), 1)  # 12
        x = self.model[13](x)  # 13
        x13 = x
        x = f.interpolate(x, scale_factor=2.0, mode="nearest")  # 14
        x = torch.cat((x, x4), 1)  # 15
        x = self.model[16](x)  # 16
        x16 = x
        x = self.model[17](x)  # 17
        x = torch.cat((x, x13), 1)  # 18
        x = self.model[19](x)  # 19
        x19 = x
        x = self.model[20](x)  # 20
        x = torch.cat((x, x10), 1)  # 21
        x = self.model[22](x)  # 22
        x22 = x
        x = self.model[23](x16, x19, x22)  # 23 - PoseHeadRaw
        return x
