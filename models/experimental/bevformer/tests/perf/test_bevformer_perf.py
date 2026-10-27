# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the detector end to end: images to class logits, boxes, the frame's BEV and the
3D detections, on device; dropping the invalid boxes on the host after it is timed apart.

Dummy weights and nuScenes' six-camera rig, at the selected variant's image size and BEV grid
(base's 1600x928 and 200x200 by default). ``BEVFORMER_PERF_FRAMES`` selects 1 or 10 consecutive
profiled frames and defaults to 1. One preceding frame, which has no previous BEV, establishes the
temporal state outside the measured region. The first profiled frame also runs once as a warmup and
PCC gate, then the complete sequence runs between Tracy's ``start`` and ``stop``. The detector runs
live, not under a trace: the encoder projects the cameras and reads their visibility on the host
inside its forward.
"""

import os
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
NUM_PROFILED_FRAMES = int(os.environ.get("BEVFORMER_PERF_FRAMES", "1"))
if NUM_PROFILED_FRAMES not in (1, 10):
    raise ValueError(f"BEVFORMER_PERF_FRAMES is {NUM_PROFILED_FRAMES}, expected 1 or 10")


def _check(expected, tt_outputs):
    """``tt_outputs`` are the detector's outputs followed by its detections."""
    tt_cls_scores, tt_bbox_preds, tt_bev = (ttnn.to_torch(t).float() for t in tt_outputs[:3])
    tt_topk = tuple(ttnn.to_torch(t) for t in tt_outputs[3][:4])
    check_detector_outputs(expected, (tt_cls_scores, tt_bbox_preds, tt_bev, tt_topk), PCC_THRESHOLD)


@torch.no_grad()
# The reference runs the backbone on six images on the CPU once to record the conv shapes and once
# per logical frame: base's 928x1600 take most of this.
@pytest.mark.timeout(1800 if NUM_PROFILED_FRAMES == 1 else 5400)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_bevformer_perf(device, reset_seeds):
    torch_model = build_reference_bevformer((BEV_H, BEV_W))
    img = random_image_batch()[None]
    frames = frame_metas(1, NUM_PROFILED_FRAMES + 1, torch.Generator().manual_seed(0))

    expected_first = expected_last = torch_prev = None
    for frame_index, metas in enumerate(frames):
        outputs = torch_model(img, metas, prev_bev=torch_prev)
        if frame_index == 1:
            expected_first = outputs
        if frame_index == NUM_PROFILED_FRAMES:
            expected_last = outputs
        *_, torch_prev = outputs

    tt_model = TTBEVFormer(create_bevformer_parameters(torch_model, img, device), device)
    # Uploaded once so the profiled run measures the detector, not the transfer.
    tt_img = to_conv_layout(img.flatten(0, 1), device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    # Each measured frame is prepared up front so the signposted region contains only detector
    # execution. Separate buffers let the measured loop advance frames without host-side refills.
    tt_frames = [tt_model.prepare_frame(metas) for metas in frames]

    *_, initial_prev_bev = tt_model(tt_img, tt_frames[0])

    def run_frame(frame, prev_bev):
        """One detector frame and its detections, all but the host compaction."""
        outputs = tt_model(tt_img, frame, prev_bev=prev_bev)
        return (*outputs, tt_model.detect(*outputs[:2]))

    # Warm the path with a previous BEV without advancing the state used by the measured sequence.
    _check(expected_first, run_frame(tt_frames[1], initial_prev_bev))

    def run_sequence():
        prev_bev = initial_prev_bev
        for frame in tt_frames[1:]:
            outputs = run_frame(frame, prev_bev)
            prev_bev = outputs[2]
        return outputs

    tt_outputs = signposted_forward(device, run_sequence)
    _check(expected_last, tt_outputs)
    scores, labels, _, boxes, valid = tt_outputs[3]
    start = time.perf_counter()
    tt_model.get_bboxes(scores, labels, boxes, valid)
    logger.info(f"get_bboxes' readback and compaction on the host: {(time.perf_counter() - start) * 1e3:.1f} ms")
