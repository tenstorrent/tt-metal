# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The detector end to end: camera images through the backbone, FPN, perception transformer and
head, over two consecutive frames.

Dummy weights by default; with ``BEVFORMER_CHECKPOINT`` set to a BEVFormer-base checkpoint
(``bevformer_r101_dcn_24ep.pth``) the reference and the port load its weights instead. The
inputs are random either way.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.bevformer.reference.bevformer import build_bevformer_base, load_bevformer_checkpoint
from models.experimental.bevformer.tests.common import (
    BEV_SHAPES,
    assert_finite,
    build_reference_bevformer,
    check_detector_outputs,
    frame_metas,
    random_image_batch,
    to_conv_layout,
)
from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_parameters
from models.experimental.bevformer.tt.tt_bevformer import TtBEVFormer

CHECKPOINT = os.environ.get("BEVFORMER_CHECKPOINT")
NUM_FRAMES = 2
# The trained weights' box velocities and yaw set this floor: the backbone's bfloat8_b weights and
# bfloat16 accumulation, carried through the encoder and the decoder, leave them at PCC 0.97 to
# 0.99, the other outputs above 0.99. The dummy weights stay above 0.99 throughout.
PCC_THRESHOLD = 0.95


def build_reference_model():
    """The dummy-weight detector, or BEVFormer-base with ``BEVFORMER_CHECKPOINT``'s weights when set."""
    if CHECKPOINT is None:
        return build_reference_bevformer(BEV_SHAPES["base"])
    model = build_bevformer_base(*BEV_SHAPES["base"])
    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]
    load_bevformer_checkpoint(model, state_dict)
    return model.eval().requires_grad_(False)


@torch.no_grad()
# The reference runs the backbone on six 928x1600 images on the CPU three times: once to record
# the conv shapes and once per frame.
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_bevformer(device, reset_seeds):
    """Each frame's BEV, class logits and boxes; the second frame takes each side's own first-frame
    BEV as its previous BEV, with the ego's rotation and shift."""
    generator = torch.Generator().manual_seed(0)
    torch_model = build_reference_model()
    img = random_image_batch()[None]
    frames = frame_metas(1, NUM_FRAMES, generator)

    tt_model = TtBEVFormer(create_bevformer_parameters(torch_model, img, device), device)
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
