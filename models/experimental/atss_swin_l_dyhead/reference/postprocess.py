# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Post-processing utilities for ATSS detection: anchor generation,
# bounding-box decoding, and NMS.  All pure PyTorch / NumPy (CPU).

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor

try:
    from torchvision.ops import batched_nms

    HAS_TV_NMS = True
except ImportError:
    HAS_TV_NMS = False


# ---------------------------------------------------------------------------
# Anchor generation
# ---------------------------------------------------------------------------


def generate_anchors_single_level(
    feat_h: int,
    feat_w: int,
    stride: int,
    base_size: int = 8,
    center_offset: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Generate anchors for one FPN level.

    For the ATSS config, each level has a single anchor per position
    (ratio=1, scale=1, octave_base_scale=8), so each anchor is a square
    of side ``stride * base_size`` centred at each grid cell.

    Returns:
        Tensor of shape (feat_h * feat_w, 4) in (x1, y1, x2, y2) format.
    """
    shift_x = (torch.arange(0, feat_w, device=device).float() + center_offset) * stride
    shift_y = (torch.arange(0, feat_h, device=device).float() + center_offset) * stride
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)

    half = stride * base_size / 2.0
    anchors = torch.stack([shift_x - half, shift_y - half, shift_x + half, shift_y + half], dim=1)
    return anchors


def generate_all_anchors(
    feat_shapes: List[Tuple[int, int]],
    strides: Sequence[int] = (8, 16, 32, 64, 128),
    base_size: int = 8,
    center_offset: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> List[Tensor]:
    """Generate anchors for all FPN levels."""
    assert len(feat_shapes) == len(strides)
    anchors = []
    for (h, w), s in zip(feat_shapes, strides):
        anchors.append(generate_anchors_single_level(h, w, s, base_size, center_offset, device))
    return anchors


# ---------------------------------------------------------------------------
# Bounding-box decoding  (DeltaXYWH)
# ---------------------------------------------------------------------------


def delta_xywh_decode(
    anchors: Tensor,
    deltas: Tensor,
    means: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    stds: Tuple[float, ...] = (0.1, 0.1, 0.2, 0.2),
    max_shape: Tuple[int, int] | None = None,
) -> Tensor:
    """Decode predicted deltas w.r.t. anchors into bounding boxes.

    Args:
        anchors: (N, 4) in (x1, y1, x2, y2).
        deltas: (N, 4) predicted (dx, dy, dw, dh).
        means / stds: normalization parameters from the config.
        max_shape: optional (H, W) to clamp decoded boxes.

    Returns:
        Tensor: (N, 4) decoded boxes in (x1, y1, x2, y2).
    """
    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    denorm_deltas = deltas * stds + means

    dx, dy, dw, dh = denorm_deltas.unbind(dim=-1)

    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    px = ax + aw * dx
    py = ay + ah * dy
    pw = aw * torch.exp(dw)
    ph = ah * torch.exp(dh)

    x1 = px - pw * 0.5
    y1 = py - ph * 0.5
    x2 = px + pw * 0.5
    y2 = py + ph * 0.5

    bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
    if max_shape is not None:
        bboxes[:, 0].clamp_(min=0, max=max_shape[1])
        bboxes[:, 1].clamp_(min=0, max=max_shape[0])
        bboxes[:, 2].clamp_(min=0, max=max_shape[1])
        bboxes[:, 3].clamp_(min=0, max=max_shape[0])
    return bboxes


# ---------------------------------------------------------------------------
# Post-processing (score filter + NMS)
# ---------------------------------------------------------------------------


@torch.no_grad()
def atss_postprocess(
    cls_scores: List[Tensor],
    bbox_preds: List[Tensor],
    centernesses: List[Tensor],
    img_shape: Tuple[int, int],
    strides: Sequence[int] = (8, 16, 32, 64, 128),
    score_thr: float = 0.05,
    nms_iou_thr: float = 0.6,
    nms_pre: int = 1000,
    max_per_img: int = 100,
) -> Dict[str, Tensor]:
    """Full ATSS post-processing pipeline.

    Args:
        cls_scores: Per-level classification logits, each (1, num_classes, H, W).
        bbox_preds: Per-level regression deltas, each (1, 4, H, W).
        centernesses: Per-level centerness logits, each (1, 1, H, W).
        img_shape: (H, W) of the original image (for box clamping).
        strides: Per-level strides.
        score_thr: Minimum score to keep.
        nms_iou_thr: NMS IoU threshold.
        nms_pre: Keep top-k before NMS per level.
        max_per_img: Keep top-k after NMS.

    Returns:
        Dict with keys 'bboxes' (N, 4), 'scores' (N,), 'labels' (N,).
    """
    assert HAS_TV_NMS, "torchvision is required for NMS"

    feat_shapes = [(s.shape[2], s.shape[3]) for s in cls_scores]
    all_anchors = generate_all_anchors(feat_shapes, strides, device=cls_scores[0].device)

    all_bboxes = []
    all_scores = []
    all_labels = []

    for level_idx in range(len(cls_scores)):
        cls_logits = cls_scores[level_idx][0]  # (num_classes, H, W)
        bbox_delta = bbox_preds[level_idx][0]  # (4, H, W)
        ctr_logits = centernesses[level_idx][0]  # (1, H, W)

        num_classes = cls_logits.shape[0]
        H, W = cls_logits.shape[1], cls_logits.shape[2]

        cls_logits = cls_logits.permute(1, 2, 0).reshape(-1, num_classes)
        bbox_delta = bbox_delta.permute(1, 2, 0).reshape(-1, 4)
        ctr_logits = ctr_logits.permute(1, 2, 0).reshape(-1)

        scores = cls_logits.sigmoid() * ctr_logits.sigmoid().unsqueeze(-1)

        max_scores, _ = scores.max(dim=1)
        if nms_pre > 0 and max_scores.shape[0] > nms_pre:
            _, topk_inds = max_scores.topk(nms_pre)
            scores = scores[topk_inds]
            bbox_delta = bbox_delta[topk_inds]
            anchors_level = all_anchors[level_idx][topk_inds]
        else:
            anchors_level = all_anchors[level_idx]

        bboxes = delta_xywh_decode(anchors_level, bbox_delta, max_shape=img_shape)

        # Flatten multi-class scores
        labels = torch.arange(num_classes, device=scores.device)
        labels = labels.unsqueeze(0).expand_as(scores).reshape(-1)
        scores_flat = scores.reshape(-1)
        bboxes_flat = bboxes.unsqueeze(1).expand(-1, num_classes, -1).reshape(-1, 4)

        # Filter by score
        valid = scores_flat > score_thr
        scores_flat = scores_flat[valid]
        bboxes_flat = bboxes_flat[valid]
        labels_flat = labels[valid]

        all_bboxes.append(bboxes_flat)
        all_scores.append(scores_flat)
        all_labels.append(labels_flat)

    bboxes = torch.cat(all_bboxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if bboxes.numel() == 0:
        return {
            "bboxes": bboxes.new_zeros((0, 4)),
            "scores": scores.new_zeros((0,)),
            "labels": labels.new_zeros((0,), dtype=torch.long),
        }

    keep = batched_nms(bboxes, scores, labels, nms_iou_thr)
    if max_per_img > 0:
        keep = keep[:max_per_img]

    return {
        "bboxes": bboxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
    }
