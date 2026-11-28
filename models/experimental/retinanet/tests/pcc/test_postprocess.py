# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import inspect

import torch
import ttnn
from loguru import logger
from torchvision.models.detection import retinanet_resnet50_fpn_v2

from models.experimental.retinanet.tests.pcc.postprocess_reference import (
    BOX_ATOL,
    SCORE_ATOL,
    device_candidates,
    reference_candidates,
)
from models.experimental.retinanet.tests.pcc.synthetic_input import IMAGE_SIZE
from models.experimental.retinanet.tt.tt_postprocess import (
    BBOX_XFORM_CLIP,
    NUM_CLASSES,
    TTDetectionDecoder,
    retinanet_anchors,
    select_detections,
)

# Every level gets more distinct high logits than the top-k keeps, so the selected set and its
# order are unique; every other logit is far below them.
NUM_HIGH_LOGITS = 1200
LOW_LOGIT = -8.0
# Every this many anchors gets width and height deltas above torchvision's clamp.
CLAMPED_ANCHOR_STRIDE = 10


def torchvision_retinanet():
    """An untrained torchvision RetinaNet, for its postprocessing settings and functions."""
    return retinanet_resnet50_fpn_v2(weights=None, weights_backbone=None).eval()


def synthetic_head_outputs(anchors_per_level, score_thresh, seed=0):
    """Head outputs whose top-k has no ties in bfloat16, as bfloat16-valued float32 tensors. No
    high logit's score is within 2 * SCORE_ATOL of ``score_thresh``, so device and host agree on
    which candidates pass it; some anchors' size deltas exceed the clamp."""
    generator = torch.Generator().manual_seed(seed)
    distinct = torch.linspace(-4.0, 4.0, 100000).to(torch.bfloat16).unique().float()
    distinct = distinct[(torch.sigmoid(distinct) - score_thresh).abs() > 2 * SCORE_ATOL]
    logits = []
    for count in anchors_per_level:
        level = torch.full((count * NUM_CLASSES,), LOW_LOGIT)
        positions = torch.randperm(count * NUM_CLASSES, generator=generator)[:NUM_HIGH_LOGITS]
        values = distinct[torch.randperm(len(distinct), generator=generator)[:NUM_HIGH_LOGITS]]
        level[positions] = values
        logits.append(level.reshape(1, count, NUM_CLASSES))
    total = sum(anchors_per_level)
    deltas = 0.5 * torch.randn(1, total, 4, generator=generator)
    deltas[0, ::CLAMPED_ANCHOR_STRIDE, 2:] = BBOX_XFORM_CLIP + 1.0
    return torch.cat(logits, dim=1), deltas.to(torch.bfloat16).float()


def to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def test_postprocess_defaults_are_torchvisions():
    """The decoder's and select_detections' defaults, which the demo relies on, are torchvision's."""
    retinanet = torchvision_retinanet()
    decoder_defaults = inspect.signature(TTDetectionDecoder.__init__).parameters
    select_defaults = inspect.signature(select_detections).parameters
    assert decoder_defaults["topk_candidates"].default == retinanet.topk_candidates
    assert select_defaults["score_thresh"].default == retinanet.score_thresh
    assert select_defaults["nms_thresh"].default == retinanet.nms_thresh
    assert select_defaults["detections_per_img"].default == retinanet.detections_per_img


def test_detection_decoder(device):
    image_size = (IMAGE_SIZE, IMAGE_SIZE)
    anchors = retinanet_anchors(image_size)
    retinanet = torchvision_retinanet()
    logits, deltas = synthetic_head_outputs([len(level) for level in anchors], retinanet.score_thresh)

    decoder = TTDetectionDecoder(device, image_size, topk_candidates=retinanet.topk_candidates)
    tt_boxes, tt_scores, tt_labels = decoder(to_device(logits, device), to_device(deltas, device))

    reference = reference_candidates(logits, deltas, anchors, retinanet.topk_candidates, image_size)
    candidates = device_candidates(tt_boxes, tt_scores, tt_labels)
    for level, (boxes, scores, labels, ref_boxes, ref_scores, ref_labels) in enumerate(zip(*candidates, *reference)):
        box_error = (boxes - ref_boxes).abs().max().item()
        score_error = (scores - ref_scores).abs().max().item()
        logger.info(f"Level {level}: max box error {box_error:.4f} px, max score error {score_error:.2e}")
        assert torch.equal(labels, ref_labels), f"Level {level}: labels differ"
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


def test_detection_decoder_rejects_other_image_size(device, expect_error):
    """Head outputs of another image size have another anchor count, which would misplace every
    level's slice."""
    decoder = TTDetectionDecoder(device, (IMAGE_SIZE, IMAGE_SIZE))
    anchor_count = sum(len(level) for level in retinanet_anchors((IMAGE_SIZE // 2, IMAGE_SIZE // 2)))
    logits = to_device(torch.zeros(1, anchor_count, NUM_CLASSES), device)
    deltas = to_device(torch.zeros(1, anchor_count, 4), device)
    with expect_error(AssertionError, "anchors"):
        decoder(logits, deltas)
