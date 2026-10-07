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
    assert_channels_close,
    assert_pcc,
    build_reference_bevformer,
    center_channels,
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
    bev_h, bev_w = BEV_SHAPES["base"]
    if CHECKPOINT is None:
        return build_reference_bevformer((bev_h, bev_w))
    model = build_bevformer_base(bev_h, bev_w)
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
        tt_cls_scores, tt_bbox_preds, tt_prev = tt_model(tt_img, frame, prev_bev=tt_prev)

        tt_bev = ttnn.to_torch(tt_prev).float().reshape(torch_prev.shape)
        tt_cls_scores = ttnn.to_torch(tt_cls_scores).float()
        tt_bbox_preds = ttnn.to_torch(tt_bbox_preds).float()
        for name, tensor in (("BEV", tt_bev), ("class logits", tt_cls_scores), ("boxes", tt_bbox_preds)):
            # comp_pcc zeroes NaN and Inf before correlating, so they must be ruled out here.
            assert torch.isfinite(tensor).all(), f"frame {i}: non-finite values in the {name}"
        logger.info(f"frame {i}")
        center_error = (bbox_preds[-1][..., center_channels()] - tt_bbox_preds[-1][..., center_channels()]).abs()
        logger.info(f"last layer's box center error: mean {center_error.mean():.4f} m, max {center_error.max():.4f} m")
        assert_pcc(torch_prev, tt_bev, PCC_THRESHOLD)
        # The last decoder layer's, which the coder reads.
        assert_pcc(cls_scores[-1], tt_cls_scores[-1], PCC_THRESHOLD)
        assert_channels_close(bbox_preds[-1], tt_bbox_preds[-1], PCC_THRESHOLD)
