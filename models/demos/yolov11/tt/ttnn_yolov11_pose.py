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
        y = ttnn.squeeze(y, dim=0)

        # Split concatenated output into bbox logits, confidence, and keypoints
        bbox_raw = y[:, :, :64]
        conf_raw = y[:, :, 64:65]
        keypoints_raw = y[:, :, 65:]

        # ===== Decode bounding boxes (DFL + anchor decode) =====
        bbox_logits = ttnn.reshape(bbox_raw, (bbox_raw.shape[0], y.shape[1], 4, 16))
        bbox_logits = ttnn.softmax_in_place(bbox_logits, dim=-1, numeric_stable=False)
        bbox_logits = ttnn.permute(bbox_logits, (0, 2, 1, 3))
        bbox_offsets = self.dfl(bbox_logits)
        ttnn.deallocate(bbox_logits)

        bbox_offsets = ttnn.sharded_to_interleaved(bbox_offsets, memory_config=ttnn.L1_MEMORY_CONFIG)
        bbox_offsets = ttnn.permute(bbox_offsets, (0, 3, 1, 2))
        bbox_offsets = ttnn.reshape(
            bbox_offsets,
            (bbox_offsets.shape[0], 1, 4, int(bbox_offsets.shape[3] / 4)),
        )
        bbox_offsets = ttnn.reshape(
            bbox_offsets,
            (bbox_offsets.shape[0], bbox_offsets.shape[1] * bbox_offsets.shape[2], bbox_offsets.shape[3]),
        )
        bbox_left_top, bbox_right_bottom = bbox_offsets[:, :2, :], bbox_offsets[:, 2:4, :]

        anchor = ttnn.to_memory_config(self.anchors, memory_config=ttnn.L1_MEMORY_CONFIG)
        strides = ttnn.to_memory_config(self.strides, memory_config=ttnn.L1_MEMORY_CONFIG)

        bbox_left_top = anchor - bbox_left_top
        bbox_right_bottom = anchor + bbox_right_bottom
        bbox_wh = bbox_right_bottom - bbox_left_top
        bbox_center = ttnn.add(bbox_left_top, bbox_right_bottom)
        bbox_center = ttnn.div(bbox_center, 2)
        bbox = ttnn.concat((bbox_center, bbox_wh), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        bbox = ttnn.multiply(bbox, strides)
        bbox = ttnn.to_layout(bbox, layout=ttnn.ROW_MAJOR_LAYOUT)

        # ===== Confidence processing =====
        conf = ttnn.permute(conf_raw, (0, 2, 1))
        conf = ttnn.sigmoid(conf)
        conf = ttnn.to_layout(conf, layout=ttnn.ROW_MAJOR_LAYOUT)

        # ===== Keypoint decoding =====
        keypoints = ttnn.permute(keypoints_raw, (0, 2, 1))
        keypoints = ttnn.reshape(
            keypoints,
            (keypoints.shape[0], keypoints.shape[1] // 3, 3, keypoints.shape[2]),
        )
        kpt_x = keypoints[:, :, 0, :]
        kpt_y = keypoints[:, :, 1, :]
        kpt_v = keypoints[:, :, 2, :]

        kpt_v = ttnn.sigmoid(kpt_v)

        kpt_x = ttnn.multiply(kpt_x, 2.0)
        kpt_y = ttnn.multiply(kpt_y, 2.0)
        kpt_x = ttnn.subtract(kpt_x, 0.5)
        kpt_y = ttnn.subtract(kpt_y, 0.5)

        anchor_x = anchor[:, 0:1, :]
        anchor_y = anchor[:, 1:2, :]
        kpt_x = ttnn.add(kpt_x, anchor_x)
        kpt_y = ttnn.add(kpt_y, anchor_y)

        strides_kp = strides
        if len(strides_kp.shape) == 2:
            strides_kp = ttnn.reshape(strides_kp, (strides_kp.shape[0], 1, strides_kp.shape[1]))

        kpt_x = ttnn.multiply(kpt_x, strides_kp)
        kpt_y = ttnn.multiply(kpt_y, strides_kp)

        kpt_x = ttnn.unsqueeze(kpt_x, dim=2)
        kpt_y = ttnn.unsqueeze(kpt_y, dim=2)
        kpt_v = ttnn.unsqueeze(kpt_v, dim=2)

        keypoints_decoded = ttnn.concat((kpt_x, kpt_y, kpt_v), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        keypoints_decoded = ttnn.reshape(
            keypoints_decoded,
            (
                keypoints_decoded.shape[0],
                keypoints_decoded.shape[1] * keypoints_decoded.shape[2],
                keypoints_decoded.shape[3],
            ),
        )
        keypoints_decoded = ttnn.to_layout(keypoints_decoded, layout=ttnn.ROW_MAJOR_LAYOUT)

        # ===== Final concatenation to match PyTorch PoseHead output =====
        out = ttnn.concat((bbox, conf, keypoints_decoded), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

        deallocate_tensors(
            y1,
            y2,
            y3,
            x1_bbox,
            x2_bbox,
            x3_bbox,
            x1_conf,
            x2_conf,
            x3_conf,
            x1_kpts,
            x2_kpts,
            x3_kpts,
            bbox_offsets,
            bbox_left_top,
            bbox_right_bottom,
            bbox_wh,
            bbox_center,
            keypoints,
            kpt_x,
            kpt_y,
            kpt_v,
            anchor,
            strides,
            bbox,
            conf,
            keypoints_decoded,
        )

        return out
