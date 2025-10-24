# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO11 Pose Estimation Model - Correct Implementation matching Ultralytics

This implementation exactly matches the Ultralytics YOLO11n-pose architecture
to enable proper weight loading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as f

# Import backbone/neck components from original yolov11.py
from yolov11 import C2PSA, DFL, SPPF, C3k2, Concat, Conv, make_anchors


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


class PoseHead(nn.Module):
    """
    YOLO11 Pose Head - Exact match to Ultralytics implementation

    Outputs:
    - cv2: Bounding box regression (64 channels)
    - cv3: Person confidence (1 channel)
    - cv4: Keypoints (51 channels = 17 keypoints × 3)
    """

    def __init__(self):
        super().__init__()

        # cv2: Bounding box regression head (3 scales)
        self.cv2 = nn.ModuleList(
            [
                # Scale 0: 64 -> 64 -> 64 -> 64
                nn.Sequential(
                    Conv(64, 64, kernel=3, stride=1, padding=1),
                    Conv(64, 64, kernel=3, stride=1, padding=1),
                    nn.Conv2d(64, 64, 1, 1),  # Final layer with bias
                ),
                # Scale 1: 128 -> 64 -> 64 -> 64
                nn.Sequential(
                    Conv(128, 64, kernel=3, stride=1, padding=1),
                    Conv(64, 64, kernel=3, stride=1, padding=1),
                    nn.Conv2d(64, 64, 1, 1),
                ),
                # Scale 2: 256 -> 64 -> 64 -> 64
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
                # Scale 0: 64 -> 1
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
                # Scale 1: 128 -> 1
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
                # Scale 2: 256 -> 1
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

        # cv4: Keypoint prediction head (3 scales) - 51 channels (17 kpts × 3)
        self.cv4 = nn.ModuleList(
            [
                # Scale 0: 64 -> 51
                nn.Sequential(
                    Conv(64, 51, kernel=3, stride=1, padding=1),
                    Conv(51, 51, kernel=3, stride=1, padding=1),
                    nn.Conv2d(51, 51, 1, 1),
                ),
                # Scale 1: 128 -> 51
                nn.Sequential(
                    Conv(128, 51, kernel=3, stride=1, padding=1),
                    Conv(51, 51, kernel=3, stride=1, padding=1),
                    nn.Conv2d(51, 51, 1, 1),
                ),
                # Scale 2: 256 -> 51
                nn.Sequential(
                    Conv(256, 51, kernel=3, stride=1, padding=1),
                    Conv(51, 51, kernel=3, stride=1, padding=1),
                    nn.Conv2d(51, 51, 1, 1),
                ),
            ]
        )

        # DFL layer for bounding box regression
        self.dfl = DFL()

    def forward(self, y1, y2, y3):
        """
        Args:
            y1, y2, y3: Feature maps from 3 scales
        Returns:
            Decoded predictions: [batch, 56, num_anchors]
                - 4: bbox (x, y, w, h)
                - 1: person confidence
                - 51: keypoints (17 × 3)
        """
        # Bbox regression for each scale
        x1_bbox = self.cv2[0](y1)  # [B, 64, H1, W1]
        x2_bbox = self.cv2[1](y2)  # [B, 64, H2, W2]
        x3_bbox = self.cv2[2](y3)  # [B, 64, H3, W3]

        # Confidence for each scale
        x1_conf = self.cv3[0](y1)  # [B, 1, H1, W1]
        x2_conf = self.cv3[1](y2)  # [B, 1, H2, W2]
        x3_conf = self.cv3[2](y3)  # [B, 1, H3, W3]

        # Keypoints for each scale
        x1_kpts = self.cv4[0](y1)  # [B, 51, H1, W1]
        x2_kpts = self.cv4[1](y2)  # [B, 51, H2, W2]
        x3_kpts = self.cv4[2](y3)  # [B, 51, H3, W3]

        # Concatenate bbox + conf for each scale
        y1 = torch.cat((x1_bbox, x1_conf, x1_kpts), 1)  # [B, 116, H1, W1]
        y2 = torch.cat((x2_bbox, x2_conf, x2_kpts), 1)  # [B, 116, H2, W2]
        y3 = torch.cat((x3_bbox, x3_conf, x3_kpts), 1)  # [B, 116, H3, W3]
        y_all = [y1, y2, y3]

        # Reshape to [B, C, num_anchors]
        y1 = torch.reshape(y1, (y1.shape[0], y1.shape[1], y1.shape[2] * y1.shape[3]))
        y2 = torch.reshape(y2, (y2.shape[0], y2.shape[1], y2.shape[2] * y2.shape[3]))
        y3 = torch.reshape(y3, (y3.shape[0], y3.shape[1], y3.shape[2] * y3.shape[3]))

        # Concatenate all scales
        y = torch.cat((y1, y2, y3), 2)  # [B, 116, total_anchors]

        # Split into bbox (64), conf (1), keypoints (51)
        ya, yb, yc = y.split((64, 1, 51), 1)

        # Process bounding boxes with DFL
        ya = torch.reshape(ya, (ya.shape[0], 4, 16, ya.shape[2]))
        ya = torch.permute(ya, (0, 2, 1, 3))
        ya = f.softmax(ya, dim=1)
        c = self.dfl(ya)
        c1 = torch.reshape(c, (c.shape[0], c.shape[1] * c.shape[2], c.shape[3]))
        c1_lt = c1[:, 0:2, :]  # left-top
        c1_rb = c1[:, 2:4, :]  # right-bottom

        # Generate anchors
        anchor, strides = (y_all.transpose(0, 1) for y_all in make_anchors(y_all, [8, 16, 32], 0.5))
        anchor.unsqueeze(0)

        # Decode bounding boxes
        lt = anchor - c1_lt
        rb = anchor + c1_rb
        wh = rb - lt
        cxy = (lt + rb) / 2
        bbox = torch.concat((cxy, wh), 1) * strides

        # Process confidence
        yb = torch.sigmoid(yb)

        # Process keypoints - decode to pixel coordinates
        # Reshape to [batch, 17, 3, num_anchors] for processing
        batch_size = yc.shape[0]
        num_anchors = yc.shape[2]
        yc = yc.reshape(batch_size, 17, 3, num_anchors)

        # Extract x, y coordinates and visibility
        kpt_x = yc[:, :, 0, :]  # [batch, 17, anchors]
        kpt_y = yc[:, :, 1, :]  # [batch, 17, anchors]
        kpt_v = yc[:, :, 2, :]  # [batch, 17, anchors]

        # Apply sigmoid ONLY to visibility, NOT to x,y coordinates
        kpt_v = torch.sigmoid(kpt_v)

        # Keypoint decoding formula from Ultralytics
        # Reference: https://community.ultralytics.com/t/understanding-keypoint-decode/357
        # NO sigmoid on x,y! Just: (x * 2 - 0.5 + anchor) * stride
        # anchor shape after transpose: [2, num_anchors], strides shape: [1, num_anchors]
        anchor_x = anchor[0, :].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]
        anchor_y = anchor[1, :].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]
        strides_val = strides[0, :].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]

        # Decode: (x * 2 - 0.5 + anchor) * stride (NO sigmoid on x,y!)
        kpt_x = (kpt_x * 2.0 - 0.5 + anchor_x) * strides_val
        kpt_y = (kpt_y * 2.0 - 0.5 + anchor_y) * strides_val

        # Stack back to [batch, 17, 3, anchors]
        keypoints_decoded = torch.stack([kpt_x, kpt_y, kpt_v], dim=2)

        # Reshape back to [batch, 51, anchors]
        keypoints_decoded = keypoints_decoded.reshape(batch_size, 51, num_anchors)

        # Final output: [bbox(4) + conf(1) + keypoints(51)] = 56 channels
        out = torch.concat((bbox, yb, keypoints_decoded), 1)
        return out


class YoloV11Pose(nn.Module):
    """YOLO11 Pose Estimation Model - Exact Ultralytics architecture"""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Conv(3, 16, kernel=3, stride=2, padding=1),  # 0
            Conv(16, 32, kernel=3, stride=2, padding=1),  # 1
            C3k2(  # 2
                [32, 48, 16, 8],
                [32, 64, 8, 16],
                [1, 1, 3, 3],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                is_bk_enabled=True,
            ),
            Conv(64, 64, kernel=3, stride=2, padding=1),  # 3
            C3k2(  # 4
                [64, 96, 32, 16],
                [64, 128, 16, 32],
                [1, 1, 3, 3],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                is_bk_enabled=True,
            ),
            Conv(128, 128, kernel=3, stride=2, padding=1),  # 5
            C3k2(
                [128, 192, 64, 64, 64, 32, 32, 32, 32],  # 6
                [128, 128, 32, 32, 64, 32, 32, 32, 32],
                [1, 1, 1, 1, 1, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            Conv(128, 256, kernel=3, stride=2, padding=1),  # 7
            C3k2(
                [256, 384, 128, 128, 128, 64, 64, 64, 64],  # 8
                [256, 256, 64, 64, 128, 64, 64, 64, 64],
                [1, 1, 1, 1, 1, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            SPPF([256, 512], [128, 256], [1, 1], [1, 1]),  # 9
            C2PSA(
                [256, 256, 128, 128, 128, 128, 256],  # 10
                [256, 256, 256, 128, 128, 256, 128],
                [1, 1, 1, 1, 3, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 128, 1, 1],
            ),
            nn.Upsample(scale_factor=2.0, mode="nearest"),  # 11
            Concat(),  # 12
            C3k2(  # 13
                [384, 192, 64, 32],
                [128, 128, 32, 64],
                [1, 1, 3, 3],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                is_bk_enabled=True,
            ),
            nn.Upsample(scale_factor=2.0, mode="nearest"),  # 14
            Concat(),  # 15
            C3k2(  # 16
                [256, 96, 32, 16],
                [64, 64, 16, 32],
                [1, 1, 3, 3],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                is_bk_enabled=True,
            ),
            Conv(64, 64, kernel=3, stride=2, padding=1),  # 17
            Concat(),  # 18
            C3k2(  # 19
                [192, 192, 64, 32],
                [128, 128, 32, 64],
                [1, 1, 3, 3],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                is_bk_enabled=True,
            ),
            Conv(128, 128, kernel=3, stride=2, padding=1),  # 20
            Concat(),  # 21
            C3k2(
                [384, 384, 128, 128, 128, 64, 64, 64, 64],  # 22
                [256, 256, 64, 64, 128, 64, 64, 64, 64],
                [1, 1, 1, 1, 1, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            PoseHead(),  # 23
        )

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
        x = self.model[23](x16, x19, x22)  # 23
        return x


# Backward compatibility
YoloV11 = YoloV11Pose
