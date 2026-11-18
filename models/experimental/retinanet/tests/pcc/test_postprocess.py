# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from loguru import logger
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops

from models.experimental.retinanet.tests.pcc.synthetic_input import IMAGE_SIZE
from models.experimental.retinanet.tt.tt_postprocess import (
    NUM_CLASSES,
    TTDetectionDecoder,
    retinanet_anchors,
    select_detections,
)

# Every level gets more distinct high logits than the top-k keeps, so the selected set and its
# order are unique; every other logit is far below them.
NUM_HIGH_LOGITS = 1200
LOW_LOGIT = -8.0
# The device decodes in float32 like torchvision; what is left is its exp() and the order of
# the float32 operations, a small fraction of a pixel.
BOX_ATOL = 0.05
# Scores are the sigmoid of the same bfloat16 logits, computed in float32 on both sides.
SCORE_ATOL = 1e-2


def synthetic_head_outputs(anchors_per_level, seed=0):
    """Head outputs whose top-k has no ties in bfloat16, as bfloat16-valued float32 tensors."""
    generator = torch.Generator().manual_seed(seed)
    distinct = torch.linspace(-4.0, 4.0, 100000).to(torch.bfloat16).unique().float()
    logits = []
    for count in anchors_per_level:
        level = torch.full((count * NUM_CLASSES,), LOW_LOGIT)
        positions = torch.randperm(count * NUM_CLASSES, generator=generator)[:NUM_HIGH_LOGITS]
        values = distinct[torch.randperm(len(distinct), generator=generator)[:NUM_HIGH_LOGITS]]
        level[positions] = values
        logits.append(level.reshape(1, count, NUM_CLASSES))
    total = sum(anchors_per_level)
    deltas = (0.5 * torch.randn(1, total, 4, generator=generator)).to(torch.bfloat16).float()
    return torch.cat(logits, dim=1), deltas


def reference_candidates(logits, deltas, anchors, topk_candidates):
    """torchvision's per-level selection and decoding, without the score threshold."""
    box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
    boxes, scores, labels = [], [], []
    start = 0
    for level_anchors in anchors:
        end = start + len(level_anchors)
        level_logits = logits[0, start:end].flatten()
        values, indices = level_logits.topk(min(topk_candidates, len(level_logits)))
        anchor_index = torch.div(indices, NUM_CLASSES, rounding_mode="floor")
        level_boxes = box_coder.decode_single(deltas[0, start:end][anchor_index], level_anchors[anchor_index])
        boxes.append(box_ops.clip_boxes_to_image(level_boxes, (IMAGE_SIZE, IMAGE_SIZE)))
        scores.append(torch.sigmoid(values))
        labels.append(indices % NUM_CLASSES)
        start = end
    return boxes, scores, labels


def test_detection_decoder(device):
    image_size = (IMAGE_SIZE, IMAGE_SIZE)
    anchors = retinanet_anchors(image_size)
    logits, deltas = synthetic_head_outputs([len(level) for level in anchors])
    # The postprocessing settings are torchvision's, read from its model.
    retinanet = retinanet_resnet50_fpn_v2(weights=None, weights_backbone=None).eval()

    def to_device(tensor):
        return ttnn.from_torch(
            tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    decoder = TTDetectionDecoder(device, image_size, topk_candidates=retinanet.topk_candidates)
    tt_boxes, tt_scores, tt_labels = decoder(to_device(logits), to_device(deltas))

    ref_boxes, ref_scores, ref_labels = reference_candidates(logits, deltas, anchors, retinanet.topk_candidates)
    for level in range(len(anchors)):
        boxes = ttnn.to_torch(tt_boxes[level]).reshape(4, -1).T.float()
        scores = ttnn.to_torch(tt_scores[level]).reshape(-1).float()
        labels = ttnn.to_torch(tt_labels[level]).reshape(-1).long()
        box_error = (boxes - ref_boxes[level]).abs().max().item()
        score_error = (scores - ref_scores[level]).abs().max().item()
        logger.info(f"Level {level}: max box error {box_error:.4f} px, max score error {score_error:.2e}")
        assert torch.equal(labels, ref_labels[level]), f"Level {level}: labels differ"
        assert score_error < SCORE_ATOL, f"Level {level}: max score error {score_error}"
        assert box_error < BOX_ATOL, f"Level {level}: max box error {box_error} px"

    # The host's threshold and NMS on the device candidates against torchvision's postprocessing.
    detections = select_detections(
        tt_boxes,
        tt_scores,
        tt_labels,
        score_thresh=retinanet.score_thresh,
        nms_thresh=retinanet.nms_thresh,
        detections_per_img=retinanet.detections_per_img,
    )
    counts = [len(level) for level in anchors]
    head_outputs = {
        "cls_logits": list(logits.split(counts, dim=1)),
        "bbox_regression": list(deltas.split(counts, dim=1)),
    }
    expected = retinanet.postprocess_detections(head_outputs, [anchors], [image_size])[0]
    logger.info(f"Detections: {len(detections['boxes'])} on device and host, {len(expected['boxes'])} in torchvision")
    assert len(expected["boxes"]) > 0, "No candidate is above the score threshold; the check is empty"
    assert torch.equal(detections["labels"], expected["labels"])
    assert torch.allclose(detections["boxes"], expected["boxes"], atol=BOX_ATOL)
    assert torch.allclose(detections["scores"], expected["scores"], atol=SCORE_ATOL)
