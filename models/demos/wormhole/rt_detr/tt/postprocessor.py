# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RT-DETR postprocessor.

import torch

import ttnn


def _cx_cy_wh_to_xyxy(boxes):
    """(cx, cy, w, h) normalised -> (x1, y1, x2, y2) normalised."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def postprocess(pred_logits_tt, pred_boxes_tt, orig_sizes, score_threshold=0.3):
    """Convert raw decoder outputs to detection results."""

    device = pred_logits_tt.device()
    mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0) if hasattr(device, "get_num_devices") else None

    # Transfer raw logits to CPU with mesh composer
    logits = ttnn.to_torch(pred_logits_tt, mesh_composer=mesh_composer).squeeze(1).float()  # (B, 300, num_classes)
    boxes = ttnn.to_torch(pred_boxes_tt, mesh_composer=mesh_composer).squeeze(1).float()  # (B, 300, 4)

    # PyTorch Post-processing
    # We do sigmoid here because hardware sigmoid introduces microscopic approximations
    # that randomly shift the argmax for identical low-confidence background boxes.
    scores = logits.sigmoid()  # (B, 300, num_classes)
    scores, labels = scores.max(dim=-1)  # (B, 300)

    boxes_xyxy = _cx_cy_wh_to_xyxy(boxes)  # (B, 300, 4) in [0,1]

    results = []
    for i in range(scores.shape[0]):
        img_h, img_w = orig_sizes[i].tolist()
        scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

        keep = scores[i] > score_threshold
        results.append(
            {
                "scores": scores[i][keep],
                "labels": labels[i][keep],
                "boxes": boxes_xyxy[i][keep] * scale,  # absolute pixel coords
            }
        )

    return results
