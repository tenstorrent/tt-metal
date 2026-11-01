# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The detector end to end: camera images through the backbone, FPN, perception transformer and
head, over two consecutive frames, with dummy weights and random inputs.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.bevformer.model_config import BEV_H, BEV_W, IMAGE_HEIGHT, IMAGE_WIDTH
from models.experimental.bevformer.tests.common import (
    assert_finite,
    build_reference_bevformer,
    check_detector_outputs,
    frame_metas,
    random_image_batch,
    to_conv_layout,
)
from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_parameters
from models.experimental.bevformer.tt.tt_bevformer import TTBEVFormer

NUM_FRAMES = 2
PCC_THRESHOLD = 0.99


@torch.no_grad()
# The reference runs the backbone on six images on the CPU three times, once to record the conv
# shapes and once per frame: base's 928x1600 take most of this.
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_bevformer(device, reset_seeds):
    """Each frame's BEV, class logits and boxes; the second frame takes each side's own first-frame
    BEV as its previous BEV, with the ego's rotation and shift."""
    generator = torch.Generator().manual_seed(0)
    logger.info(f"{IMAGE_WIDTH}x{IMAGE_HEIGHT} images, {BEV_H}x{BEV_W} BEV")
    torch_model = build_reference_bevformer((BEV_H, BEV_W))
    img = random_image_batch()[None]
    frames = frame_metas(1, NUM_FRAMES, generator)

    tt_model = TTBEVFormer(create_bevformer_parameters(torch_model, img, device), device)
    tt_img = to_conv_layout(img.flatten(0, 1), device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    torch_prev = tt_prev = frame = None
    for i, metas in enumerate(frames):
        cls_scores, bbox_preds, torch_prev = torch_model(img, metas, prev_bev=torch_prev)
        frame = tt_model.prepare_frame(metas, frame)
        tt_outputs = tt_model(tt_img, frame, prev_bev=tt_prev)
        *_, tt_prev = tt_outputs
        tt_detections = tt_model.detect(*tt_outputs[:2])

        logger.info(f"frame {i}")
        # The outputs and the detections on device against the reference's, then the host compaction.
        scores, labels, query_index, boxes, valid = tt_detections
        tt_bev, tt_cls_scores, tt_bbox_preds = (ttnn.to_torch(t).float() for t in (tt_prev, *tt_outputs[:2]))
        tt_topk = tuple(ttnn.to_torch(t) for t in (scores, labels, query_index, boxes))
        check_detector_outputs(
            (cls_scores, bbox_preds, torch_prev), (tt_cls_scores, tt_bbox_preds, tt_bev, tt_topk), PCC_THRESHOLD
        )
        for (tt_boxes, _, _), (ref_boxes, _, _) in zip(
            tt_model.get_bboxes(scores, labels, boxes, valid),
            torch_model.get_bboxes(cls_scores, bbox_preds),
            strict=True,
        ):
            assert tt_boxes.shape[-1] == ref_boxes.shape[-1]
            assert_finite("the 3D boxes", tt_boxes)
            logger.info(f"3D boxes in range: {len(tt_boxes)} on device, {len(ref_boxes)} in the reference")
