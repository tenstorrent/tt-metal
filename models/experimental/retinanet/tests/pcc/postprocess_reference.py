# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""torchvision's candidate selection and box decoding, to check TTDetectionDecoder against."""

import torch
import ttnn
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops

from models.experimental.retinanet.tt.tt_postprocess import NUM_CLASSES

# The device decodes in float32 like torchvision; what is left is its exp() and the order of
# the float32 operations, a small fraction of a pixel.
BOX_ATOL = 0.05
# Scores are the sigmoid of the same bfloat16 logits, computed in float32 on both sides.
SCORE_ATOL = 1e-2


def reference_candidates(logits, deltas, anchors, topk_candidates, image_size):
    """torchvision's per-level selection and decoding of the head outputs ``logits`` [1, A, 91] and
    ``deltas`` [1, A, 4], without the score threshold: per level, boxes [k, 4], scores [k] and
    labels [k], best first."""
    box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
    boxes, scores, labels = [], [], []
    start = 0
    for level_anchors in anchors:
        end = start + len(level_anchors)
        level_logits = logits[0, start:end].flatten()
        values, indices = level_logits.topk(min(topk_candidates, len(level_logits)))
        anchor_index = torch.div(indices, NUM_CLASSES, rounding_mode="floor")
        level_boxes = box_coder.decode_single(deltas[0, start:end][anchor_index], level_anchors[anchor_index])
        boxes.append(box_ops.clip_boxes_to_image(level_boxes, image_size))
        scores.append(torch.sigmoid(values))
        labels.append(indices % NUM_CLASSES)
        start = end
    return boxes, scores, labels


def device_candidates(tt_boxes, tt_scores, tt_labels):
    """TTDetectionDecoder's per-level output on the host, in reference_candidates' format."""
    boxes = [ttnn.to_torch(level).reshape(4, -1).T.float() for level in tt_boxes]
    scores = [ttnn.to_torch(level).reshape(-1).float() for level in tt_scores]
    labels = [ttnn.to_torch(level).reshape(-1).long() for level in tt_labels]
    return boxes, scores, labels


def assert_candidates_match(device, reference, level):
    """Asserts one level's device candidates match the reference ones, which may differ only in
    candidates tied with the last one selected: the two top-k break ties differently. The rest are
    compared in the order of (score, label, box)."""
    (boxes, scores, labels), (ref_boxes, ref_scores, ref_labels) = device, reference
    assert len(scores) == len(ref_scores), f"Level {level}: {len(scores)} candidates, expected {len(ref_scores)}"
    # Above the last selected score by less than any two distinct bfloat16 logits' scores differ.
    cutoff = ref_scores.min().item() + 1e-5

    def untied(boxes, scores, labels):
        keep = scores > cutoff
        rows = sorted(zip(scores[keep].tolist(), labels[keep].tolist(), boxes[keep].tolist()))
        return (
            torch.tensor([row[0] for row in rows]),
            torch.tensor([row[1] for row in rows], dtype=torch.long),
            torch.tensor([row[2] for row in rows]).reshape(-1, 4),
        )

    scores, labels, boxes = untied(boxes, scores, labels)
    ref_scores, ref_labels, ref_boxes = untied(ref_boxes, ref_scores, ref_labels)
    assert len(scores) == len(ref_scores), f"Level {level}: candidates above the last tie differ"
    assert torch.equal(labels, ref_labels), f"Level {level}: labels differ"
    score_error = (scores - ref_scores).abs().max().item() if len(scores) else 0.0
    box_error = (boxes - ref_boxes).abs().max().item() if len(boxes) else 0.0
    assert score_error < SCORE_ATOL, f"Level {level}: max score error {score_error}"
    assert box_error < BOX_ATOL, f"Level {level}: max box error {box_error} px"
    return box_error, score_error
