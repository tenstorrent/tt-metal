# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Custom extension loader that provides CPU-compatible implementations
using torchvision.ops instead of MMCV's CUDA extensions.
"""

import torch
import torchvision.ops as ops


class CPUNMSWrapper:
    """Wrapper providing NMS functions using torchvision (CPU compatible)."""

    @staticmethod
    def nms(boxes, scores, iou_threshold, offset=0):
        """
        Non-maximum suppression using torchvision.
        Args:
            boxes: (N, 4) tensor of boxes in (x1, y1, x2, y2) format
            scores: (N,) tensor of scores
            iou_threshold: IoU threshold for NMS
            offset: offset for box coordinates (ignored, for compatibility)
        Returns:
            indices of kept boxes
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        return ops.nms(boxes, scores, iou_threshold)

    @staticmethod
    def softnms(boxes, scores, iou_threshold, sigma=0.5, min_score=0.001, method=1, offset=0):
        """
        Soft-NMS fallback - uses regular NMS as approximation.
        For true soft-NMS, a custom implementation would be needed.
        """
        return CPUNMSWrapper.nms(boxes, scores, iou_threshold, offset)

    @staticmethod
    def nms_match(dets, iou_threshold):
        """NMS match - fallback to regular NMS."""
        if dets.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=dets.device)
        boxes = dets[:, :4]
        scores = dets[:, 4]
        return ops.nms(boxes, scores, iou_threshold)

    @staticmethod
    def nms_rotated(dets, scores, iou_threshold, labels=None):
        """
        Rotated NMS fallback - uses regular NMS as approximation.
        For true rotated NMS, boxes should be converted to axis-aligned first.
        """
        if dets.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=dets.device)
        # For rotated boxes, we approximate with axis-aligned bounding boxes
        # dets format: (x_center, y_center, width, height, angle)
        if dets.shape[1] == 5:
            # Convert rotated boxes to axis-aligned for approximation
            x_c, y_c, w, h = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
            half_w, half_h = w / 2, h / 2
            boxes = torch.stack([x_c - half_w, y_c - half_h, x_c + half_w, y_c + half_h], dim=1)
        else:
            boxes = dets[:, :4]
        return ops.nms(boxes, scores, iou_threshold)


class ExtWrapper:
    """Wrapper to provide attribute access to extension functions."""

    def __init__(self, ext):
        self._ext = ext

    def __getattr__(self, name):
        return getattr(self._ext, name)


def load_ext(name, funcs):
    """
    Load extension module with CPU-compatible implementations.
    Uses torchvision.ops for NMS operations instead of MMCV CUDA extensions.
    """
    if name == "_ext":
        # Return CPU-compatible NMS wrapper
        return ExtWrapper(CPUNMSWrapper)
    else:
        # For other extensions, try to import from mmcv
        # but this may fail if CUDA is required
        try:
            import importlib

            ext = importlib.import_module("mmcv." + name)
            return ExtWrapper(ext)
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(
                f"Cannot load extension '{name}'. " f"MMCV CUDA extensions are not available. " f"Error: {e}"
            )
