# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of YOLO11 Pose Head

This implements the pose estimation head that predicts:
- Bounding boxes (cv2)
- Person confidence (cv3) - uses regular Conv (optimized)
- Keypoints (cv4) - 17 keypoints × 3 values
"""

import ttnn
from models.demos.yolov11.tt.common import TtnnConv, Yolov11Conv2D, deallocate_tensors


class TtnnPoseHead:
    """
    TTNN Pose Head for YOLO11 Pose Estimation

    Architecture:
    - cv2: Bounding box regression (3 scales, 64 channels each)
    - cv3: Person confidence (3 scales, 1 channel each, uses regular Conv)
    - cv4: Keypoints (3 scales, 51 channels each = 17 keypoints × 3)
    """

    def __init__(self, device, parameter, conv_pt):
        """
        Initialize TTNN Pose Head

        Args:
            device: TT device
            parameter: Parameter configuration from model
            conv_pt: PyTorch pose head with pretrained weights
        """
        self.device = device

        # cv2: Bounding box regression head (3 scales)
        # Scale 0: 64 -> 64 -> 64
        self.cv2_0_0 = TtnnConv(device, parameter.cv2[0][0], conv_pt.cv2[0][0], is_detect=True)
        self.cv2_0_1 = TtnnConv(device, parameter.cv2[0][1], conv_pt.cv2[0][1], is_detect=True)
        self.cv2_0_2 = Yolov11Conv2D(parameter.cv2[0][2], conv_pt.cv2[0][2], device=device, is_detect=True)

        # Scale 1: 128 -> 64 -> 64
        self.cv2_1_0 = TtnnConv(device, parameter.cv2[1][0], conv_pt.cv2[1][0], is_detect=True)
        self.cv2_1_1 = TtnnConv(device, parameter.cv2[1][1], conv_pt.cv2[1][1], is_detect=True)
        self.cv2_1_2 = Yolov11Conv2D(parameter.cv2[1][2], conv_pt.cv2[1][2], device=device, is_detect=True)

        # Scale 2: 256 -> 64 -> 64
        self.cv2_2_0 = TtnnConv(device, parameter.cv2[2][0], conv_pt.cv2[2][0], is_detect=True)
        self.cv2_2_1 = TtnnConv(device, parameter.cv2[2][1], conv_pt.cv2[2][1], is_detect=True)
        self.cv2_2_2 = Yolov11Conv2D(parameter.cv2[2][2], conv_pt.cv2[2][2], device=device, is_detect=True)

        # cv3: Person confidence head (3 scales, uses regular Conv for performance)
        # Scale 0: 64 -> 1 (Conv -> Conv -> Conv -> Conv -> Conv2d)
        self.cv3_0_0_0 = TtnnConv(device, parameter.cv3[0][0][0], conv_pt.cv3[0][0][0], is_detect=True)
        self.cv3_0_0_1 = TtnnConv(device, parameter.cv3[0][0][1], conv_pt.cv3[0][0][1], is_detect=True)
        self.cv3_0_1_0 = TtnnConv(device, parameter.cv3[0][1][0], conv_pt.cv3[0][1][0], is_detect=True)
        self.cv3_0_1_1 = TtnnConv(device, parameter.cv3[0][1][1], conv_pt.cv3[0][1][1], is_detect=True)
        self.cv3_0_2 = Yolov11Conv2D(parameter.cv3[0][2], conv_pt.cv3[0][2], device=device, is_detect=True)

        # Scale 1: 128 -> 1
        self.cv3_1_0_0 = TtnnConv(device, parameter.cv3[1][0][0], conv_pt.cv3[1][0][0], is_detect=True)
        self.cv3_1_0_1 = TtnnConv(device, parameter.cv3[1][0][1], conv_pt.cv3[1][0][1], is_detect=True)
        self.cv3_1_1_0 = TtnnConv(device, parameter.cv3[1][1][0], conv_pt.cv3[1][1][0], is_detect=True)
        self.cv3_1_1_1 = TtnnConv(device, parameter.cv3[1][1][1], conv_pt.cv3[1][1][1], is_detect=True)
        self.cv3_1_2 = Yolov11Conv2D(parameter.cv3[1][2], conv_pt.cv3[1][2], device=device, is_detect=True)

        # Scale 2: 256 -> 1
        self.cv3_2_0_0 = TtnnConv(device, parameter.cv3[2][0][0], conv_pt.cv3[2][0][0], is_detect=True)
        self.cv3_2_0_1 = TtnnConv(device, parameter.cv3[2][0][1], conv_pt.cv3[2][0][1], is_detect=True)
        self.cv3_2_1_0 = TtnnConv(device, parameter.cv3[2][1][0], conv_pt.cv3[2][1][0], is_detect=True)
        self.cv3_2_1_1 = TtnnConv(device, parameter.cv3[2][1][1], conv_pt.cv3[2][1][1], is_detect=True)
        self.cv3_2_2 = Yolov11Conv2D(parameter.cv3[2][2], conv_pt.cv3[2][2], device=device, is_detect=True)

        # cv4: Keypoint prediction head (3 scales)
        # Scale 0: 64 -> 51
        self.cv4_0_0 = TtnnConv(device, parameter.cv4[0][0], conv_pt.cv4[0][0], is_detect=True)
        self.cv4_0_1 = TtnnConv(device, parameter.cv4[0][1], conv_pt.cv4[0][1], is_detect=True)
        self.cv4_0_2 = Yolov11Conv2D(parameter.cv4[0][2], conv_pt.cv4[0][2], device=device, is_detect=True)

        # Scale 1: 128 -> 51
        self.cv4_1_0 = TtnnConv(device, parameter.cv4[1][0], conv_pt.cv4[1][0], is_detect=True)
        self.cv4_1_1 = TtnnConv(device, parameter.cv4[1][1], conv_pt.cv4[1][1], is_detect=True)
        self.cv4_1_2 = Yolov11Conv2D(parameter.cv4[1][2], conv_pt.cv4[1][2], device=device, is_detect=True)

        # Scale 2: 256 -> 51
        self.cv4_2_0 = TtnnConv(device, parameter.cv4[2][0], conv_pt.cv4[2][0], is_detect=True)
        self.cv4_2_1 = TtnnConv(device, parameter.cv4[2][1], conv_pt.cv4[2][1], is_detect=True)
        self.cv4_2_2 = Yolov11Conv2D(parameter.cv4[2][2], conv_pt.cv4[2][2], device=device, is_detect=True)

        # DFL layer for bounding box regression
        self.dfl = Yolov11Conv2D(parameter.dfl.conv, conv_pt.dfl.conv, device=device, is_dfl=True)

        # Anchors and strides for decoding
        self.anchors = conv_pt.anchors
        self.strides = conv_pt.strides

    def __call__(self, device, y1, y2, y3, tile_size=32):
        """
        Forward pass through pose head

        Args:
            device: TT device
            y1: Feature map from scale 0 (smallest)
            y2: Feature map from scale 1 (medium)
            y3: Feature map from scale 2 (largest)
            tile_size: Tile size for TTNN operations

        Returns:
            Decoded pose predictions: [batch, 56, num_anchors]
                - 4 channels: bbox (x, y, w, h)
                - 1 channel: person confidence
                - 51 channels: keypoints (17 × 3)
        """

        # ===== cv2: Bounding box regression =====
        x1_bbox = self.cv2_0_0(device, y1)
        x1_bbox = self.cv2_0_1(device, x1_bbox)
        x1_bbox = self.cv2_0_2(x1_bbox)

        x2_bbox = self.cv2_1_0(device, y2)
        x2_bbox = self.cv2_1_1(device, x2_bbox)
        x2_bbox = self.cv2_1_2(x2_bbox)

        x3_bbox = self.cv2_2_0(device, y3)
        x3_bbox = self.cv2_2_1(device, x3_bbox)
        x3_bbox = self.cv2_2_2(x3_bbox)

        # ===== cv3: Person confidence (uses regular Conv) =====
        x1_conf = self.cv3_0_0_0(device, y1)
        x1_conf = self.cv3_0_0_1(device, x1_conf)
        x1_conf = self.cv3_0_1_0(device, x1_conf)
        x1_conf = self.cv3_0_1_1(device, x1_conf)
        x1_conf = self.cv3_0_2(x1_conf)

        x2_conf = self.cv3_1_0_0(device, y2)
        x2_conf = self.cv3_1_0_1(device, x2_conf)
        x2_conf = self.cv3_1_1_0(device, x2_conf)
        x2_conf = self.cv3_1_1_1(device, x2_conf)
        x2_conf = self.cv3_1_2(x2_conf)

        x3_conf = self.cv3_2_0_0(device, y3)
        x3_conf = self.cv3_2_0_1(device, x3_conf)
        x3_conf = self.cv3_2_1_0(device, x3_conf)
        x3_conf = self.cv3_2_1_1(device, x3_conf)
        x3_conf = self.cv3_2_2(x3_conf)

        # ===== cv4: Keypoint prediction =====
        x1_kpts = self.cv4_0_0(device, y1)
        x1_kpts = self.cv4_0_1(device, x1_kpts)
        x1_kpts = self.cv4_0_2(x1_kpts)

        x2_kpts = self.cv4_1_0(device, y2)
        x2_kpts = self.cv4_1_1(device, x2_kpts)
        x2_kpts = self.cv4_1_2(x2_kpts)

        x3_kpts = self.cv4_2_0(device, y3)
        x3_kpts = self.cv4_2_1(device, x3_kpts)
        x3_kpts = self.cv4_2_2(x3_kpts)

        # ===== Concatenate bbox + conf + keypoints for each scale =====
        # Convert to interleaved first to avoid circular buffer alignment issues
        # (116 channels = 64 bbox + 1 conf + 51 keypoints)
        x1_bbox = ttnn.sharded_to_interleaved(x1_bbox, memory_config=ttnn.L1_MEMORY_CONFIG)
        x1_conf = ttnn.sharded_to_interleaved(x1_conf, memory_config=ttnn.L1_MEMORY_CONFIG)
        x1_kpts = ttnn.sharded_to_interleaved(x1_kpts, memory_config=ttnn.L1_MEMORY_CONFIG)

        x2_bbox = ttnn.sharded_to_interleaved(x2_bbox, memory_config=ttnn.L1_MEMORY_CONFIG)
        x2_conf = ttnn.sharded_to_interleaved(x2_conf, memory_config=ttnn.L1_MEMORY_CONFIG)
        x2_kpts = ttnn.sharded_to_interleaved(x2_kpts, memory_config=ttnn.L1_MEMORY_CONFIG)

        x3_bbox = ttnn.sharded_to_interleaved(x3_bbox, memory_config=ttnn.L1_MEMORY_CONFIG)
        x3_conf = ttnn.sharded_to_interleaved(x3_conf, memory_config=ttnn.L1_MEMORY_CONFIG)
        x3_kpts = ttnn.sharded_to_interleaved(x3_kpts, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Concatenate all three at once (bbox + conf + keypoints = 116 channels)
        # Already converted to interleaved above, so use interleaved concat
        y1 = ttnn.concat((x1_bbox, x1_conf, x1_kpts), dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y2 = ttnn.concat((x2_bbox, x2_conf, x2_kpts), dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y3 = ttnn.concat((x3_bbox, x3_conf, x3_kpts), dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Concatenate all scales (y1, y2, y3 are now interleaved)
        y = ttnn.concat((y1, y2, y3), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        y = ttnn.to_layout(y, layout=ttnn.TILE_LAYOUT)
        # Keep batch dimension for proper splitting: y shape is [B, 116, total_anchors]

        # Split into bbox (64), conf (1), keypoints (51) along channel dimension
        ya = y[:, :64, :]  # Bbox regression (DFL input) - [B, 64, total_anchors]
        yb = y[:, 64:65, :]  # Person confidence - [B, 1, total_anchors]
        yc = y[:, 65:116, :]  # Keypoints (51 channels) - [B, 51, total_anchors]

        deallocate_tensors(y1, y2, y3, x1_bbox, x2_bbox, x3_bbox, x1_conf, x2_conf, x3_conf, x1_kpts, x2_kpts, x3_kpts)

        # ===== SKIP DFL PROCESSING =====
        # Return raw outputs directly like YoloV11PoseRaw does
        # DFL processing will be done in PyTorch CPU post-processing

        # Process confidence (simple sigmoid)
        yb = ttnn.sigmoid(yb)  # [B, 1, total_anchors]

        # Keep bbox and keypoints raw - no DFL decoding in TTNN

        # Return raw concatenated output - post-processing done in PyTorch CPU
        dram_memory_config = ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.concat((ya, yb, yc), dim=1, memory_config=dram_memory_config)  # [B, 116, total_anchors]

        deallocate_tensors(ya, yb, yc)

        return out
