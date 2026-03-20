# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YUNetTool: Face detection using YUNet on TTNN.

Detects faces in images with bounding boxes and facial keypoints.
Uses the YUNet architecture optimized for TTNN.
"""

import os
from typing import Dict, List

import cv2
import torch
from loguru import logger

import ttnn
from models.experimental.yunet.common import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_NMS_IOU_THRESHOLD,
    STRIDES,
    YUNET_INPUT_SIZE,
    get_default_weights_path,
    load_torch_model,
    setup_yunet_reference,
)
from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model


class YUNetTool:
    """
    TTNN-accelerated YUNet face detection tool.

    Detects faces in images and returns bounding boxes with facial keypoints
    (eyes, nose, mouth corners).
    """

    def __init__(self, mesh_device, input_size: int = YUNET_INPUT_SIZE):
        self.mesh_device = mesh_device
        self.input_size = input_size
        self._init_model(mesh_device)

    def _init_model(self, mesh_device):
        """Load YUNet model."""
        logger.info("Loading YUNet face detection model...")

        # Use chip0 submesh for YUNet (single-device model)
        if hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() > 1:
            self.device = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        else:
            self.device = mesh_device

        # Setup reference model if needed
        setup_yunet_reference()

        # Load weights
        weights_path = get_default_weights_path()
        if not os.path.exists(weights_path):
            logger.info("Downloading YUNet weights...")
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            import urllib.request

            urllib.request.urlretrieve(
                "https://github.com/jahongir7174/YUNet/releases/download/v0.0.1/best.pt",
                weights_path,
            )

        torch_model = load_torch_model(weights_path).to(torch.bfloat16)
        self.tt_model = create_yunet_model(self.device, torch_model)

        logger.info("YUNet ready.")

    def detect(
        self,
        image_path: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    ) -> List[Dict]:
        """
        Detect faces in an image.

        Args:
            image_path: Path to the input image.
            confidence_threshold: Minimum confidence score for detections.
            nms_threshold: IoU threshold for non-maximum suppression.

        Returns:
            List of detections, each containing:
                - box: (x1, y1, x2, y2) bounding box coordinates
                - confidence: Detection confidence score
                - keypoints: List of (x, y) facial keypoints
        """
        logger.info(f"Detecting faces in: {image_path}")

        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        orig_h, orig_w = image.shape[:2]

        # Resize to model input size
        image_resized = cv2.resize(image, (self.input_size, self.input_size))
        image_rgb = image_resized[:, :, ::-1]  # BGR -> RGB

        # Convert to NHWC tensor
        tensor = torch.from_numpy(image_rgb.copy()).float()
        tensor_nhwc = tensor.unsqueeze(0).to(torch.bfloat16)

        # Run inference
        tt_input = ttnn.from_torch(tensor_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        cls_outs, box_outs, obj_outs, kpt_outs = self.tt_model(tt_input)

        # Decode detections
        detections = self._decode_outputs(cls_outs, box_outs, obj_outs, kpt_outs, orig_w, orig_h, confidence_threshold)

        # Apply NMS
        detections = self._nms(detections, nms_threshold)

        logger.info(f"Found {len(detections)} face(s)")
        return detections

    def _decode_outputs(self, cls_outs, box_outs, obj_outs, kpt_outs, orig_w, orig_h, threshold) -> List[Dict]:
        """Decode raw model outputs to detections."""
        detections = []

        for scale_idx in range(3):
            cls_out = ttnn.to_torch(cls_outs[scale_idx]).float().permute(0, 3, 1, 2)
            box_out = ttnn.to_torch(box_outs[scale_idx]).float().permute(0, 3, 1, 2)
            obj_out = ttnn.to_torch(obj_outs[scale_idx]).float().permute(0, 3, 1, 2)
            kpt_out = ttnn.to_torch(kpt_outs[scale_idx]).float().permute(0, 3, 1, 2)

            stride = STRIDES[scale_idx]
            score = cls_out.sigmoid() * obj_out.sigmoid()

            high_conf = score > threshold
            if not high_conf.any():
                continue

            indices = torch.where(high_conf)
            for b, c, h, w in zip(*indices):
                conf = score[b, c, h, w].item()
                anchor_x = w.item() * stride
                anchor_y = h.item() * stride

                # Decode box
                dx = box_out[b, 0, h, w].item()
                dy = box_out[b, 1, h, w].item()
                dw = box_out[b, 2, h, w].item()
                dh = box_out[b, 3, h, w].item()

                cx = (dx * stride + anchor_x) * orig_w / self.input_size
                cy = (dy * stride + anchor_y) * orig_h / self.input_size
                bw = (dw * stride) * orig_w / self.input_size
                bh = (dh * stride) * orig_h / self.input_size

                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2)
                y2 = int(cy + bh / 2)

                # Decode keypoints (5 points: eyes, nose, mouth corners)
                keypoints = []
                for k in range(5):
                    kpt_dx = kpt_out[b, k * 2, h, w].item()
                    kpt_dy = kpt_out[b, k * 2 + 1, h, w].item()
                    kx = int((kpt_dx * stride + anchor_x) * orig_w / self.input_size)
                    ky = int((kpt_dy * stride + anchor_y) * orig_h / self.input_size)
                    keypoints.append((kx, ky))

                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "confidence": conf,
                        "keypoints": keypoints,
                    }
                )

        return detections

    def _nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """Apply non-maximum suppression."""
        if not detections:
            return []

        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)

            remaining = []
            for det in detections:
                x1 = max(best["box"][0], det["box"][0])
                y1 = max(best["box"][1], det["box"][1])
                x2 = min(best["box"][2], det["box"][2])
                y2 = min(best["box"][3], det["box"][3])

                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (best["box"][2] - best["box"][0]) * (best["box"][3] - best["box"][1])
                area2 = (det["box"][2] - det["box"][0]) * (det["box"][3] - det["box"][1])

                iou = inter / max(area1 + area2 - inter, 1e-6)
                if iou < iou_threshold:
                    remaining.append(det)

            detections = remaining

        return keep

    def close(self):
        """Release resources."""
        self.tt_model = None
        logger.info("YUNetTool closed.")
