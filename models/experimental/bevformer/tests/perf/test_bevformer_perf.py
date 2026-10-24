# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the detector end to end: images to class logits, boxes, the frame's BEV and the
3D detections, on device; dropping the invalid boxes on the host after it is timed apart.

Dummy weights and nuScenes' six-camera rig, three frames, at the selected variant's image size and
BEV grid (base's 1600x928 and 200x200 by default).
The first frame, which has no previous BEV, runs eagerly; the second is the warmup and the PCC
gate, then runs once more between Tracy's ``start`` and ``stop``, so the stage signposts and the
host dispatch are in the report. The detector runs live, not under a trace: the encoder projects
the cameras and reads their visibility on the host inside its forward. The third frame refills
the frame's inputs with the ego driving backwards and twice as far, turning the other way, and
checks the outputs once more.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.bevformer.model_config import BEV_H, BEV_W
from models.experimental.bevformer.tests.common import (
    build_reference_bevformer,
    check_detector_outputs,
    frame_metas,
    random_image_batch,
    signposted_forward,
    to_conv_layout,
)
from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_parameters
from models.experimental.bevformer.tt.tt_bevformer import TTBEVFormer

PCC_THRESHOLD = 0.99


def _check(expected, tt_outputs):
    """``tt_outputs`` are the detector's outputs followed by its detections."""
    tt_cls_scores, tt_bbox_preds, tt_bev = (ttnn.to_torch(t).float() for t in tt_outputs[:3])
    tt_topk = tuple(ttnn.to_torch(t) for t in tt_outputs[3][:4])
    check_detector_outputs(expected, (tt_cls_scores, tt_bbox_preds, tt_bev, tt_topk), PCC_THRESHOLD)


@torch.no_grad()
# The reference runs the backbone on six images on the CPU four times, once to record the conv
# shapes and once per frame: base's 928x1600 take most of this.
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_bevformer_perf(device, reset_seeds):
    torch_model = build_reference_bevformer((BEV_H, BEV_W))
    img = random_image_batch()[None]
    frames = frame_metas(1, 3, torch.Generator().manual_seed(0), motion=(1, -2))

    expected, torch_prev = [], None
    for metas in frames:
        outputs = torch_model(img, metas, prev_bev=torch_prev)
        expected.append(outputs)
        *_, torch_prev = outputs

    tt_model = TTBEVFormer(create_bevformer_parameters(torch_model, img, device), device)
    # Uploaded once so the profiled run measures the detector, not the transfer.
    tt_img = to_conv_layout(img.flatten(0, 1), device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    frame = tt_model.prepare_frame(frames[0])
    *_, prev_bev = tt_model(tt_img, frame)
    tt_model.prepare_frame(frames[1], frame)

    def run():
        """The detector and its detections: images to 3D boxes, all but the host compaction."""
        outputs = tt_model(tt_img, frame, prev_bev=prev_bev)
        return (*outputs, tt_model.detect(*outputs[:2]))

    # Doubles as the warmup of the path with a previous BEV.
    _check(expected[1], run())
    tt_outputs = signposted_forward(device, run)
    _check(expected[1], tt_outputs)

    start = time.perf_counter()
    tt_model.prepare_frame(frames[2], frame)
    logger.info(f"prepare_frame refill on the host: {(time.perf_counter() - start) * 1e3:.1f} ms")
    prev_bev = tt_outputs[2]
    tt_outputs = run()
    _check(expected[2], tt_outputs)
    scores, labels, _, boxes, valid = tt_outputs[3]
    start = time.perf_counter()
    tt_model.get_bboxes(scores, labels, boxes, valid)
    logger.info(f"get_bboxes' readback and compaction on the host: {(time.perf_counter() - start) * 1e3:.1f} ms")
