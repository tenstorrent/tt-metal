# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the detector end to end: images to class logits, boxes, the frame's BEV and the
3D detections, on device; dropping the invalid boxes on the host after it is timed apart.

Dummy weights and nuScenes' six-camera rig, at the selected variant's image size and BEV grid
(base's 1600x928 and 200x200 by default). ``BEVFORMER_PERF_FRAMES`` selects 1 or 2 consecutive
profiled frames and defaults to 1. One preceding frame, which has no previous BEV, establishes the
temporal state outside the measured region. The first profiled frame also runs once as a warmup and
PCC gate, then the complete sequence runs between Tracy's ``start`` and ``stop``.

``live`` dispatches the detector op by op. ``trace`` replays the same sequence from
``TTBEVFormerTrace``'s traces, from the same state; each frame's images are uploaded inside the
window, as the runner takes one upload per frame, and its other host inputs are prepared before
it. Report a ``trace`` run with ``tt-perf-report --tracing-mode``.
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
from models.experimental.bevformer.tt.tt_bevformer_trace import TTBEVFormerTrace
from models.experimental.bevformer.tt.tt_head import box_preds_to_torch

PCC_THRESHOLD = 0.99
NUM_PROFILED_FRAMES = int(os.environ.get("BEVFORMER_PERF_FRAMES", "1"))
if NUM_PROFILED_FRAMES not in (1, 2):
    raise ValueError(f"BEVFORMER_PERF_FRAMES is {NUM_PROFILED_FRAMES}, expected 1 or 2")


def _check(expected, tt_outputs):
    """``tt_outputs`` are the detector's outputs followed by its detections."""
    tt_cls_scores, tt_bev = (ttnn.to_torch(t).float() for t in (tt_outputs[0], tt_outputs[2]))
    tt_bbox_preds = box_preds_to_torch(tt_outputs[1]).float()
    tt_topk = tuple(ttnn.to_torch(t) for t in tt_outputs[3][:4])
    check_detector_outputs(expected, (tt_cls_scores, tt_bbox_preds, tt_bev, tt_topk), PCC_THRESHOLD)


@torch.no_grad()
# The reference runs the backbone on six images on the CPU for the initial temporal state and the
# first profiled frame's PCC gate. The remaining frames are performance-only: base's 928x1600 take
# most of the test setup time, and correctness of the temporal sequence is covered separately.
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params, execution",
    [
        pytest.param({"l1_small_size": 32 * 1024}, "live", id="live"),
        pytest.param({"l1_small_size": 32 * 1024, "trace_region_size": 256 << 20}, "trace", id="trace"),
    ],
    indirect=["device_params"],
)
def test_bevformer_perf(device, execution, reset_seeds):
    torch_model = build_reference_bevformer((BEV_H, BEV_W))
    img = random_image_batch()[None]
    frames = frame_metas(1, NUM_PROFILED_FRAMES + 1, torch.Generator().manual_seed(0))

    expected_first = torch_prev = None
    for frame_index, metas in enumerate(frames[:2]):
        outputs = torch_model(img, metas, prev_bev=torch_prev)
        logger.info(f"Reference frame {frame_index}/{NUM_PROFILED_FRAMES} completed outside measurements")
        if frame_index == 1:
            expected_first = outputs
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
    warm = run_frame(tt_frames[1], initial_prev_bev)
    _check(expected_first, warm)
    scores, labels, _, boxes, valid = (ttnn.to_torch(t) for t in warm[3])
    warm_detections = (scores, labels, boxes, valid)
    # Every measured frame once, so the planner is sized for all of them before the window.
    for frame in tt_frames[2:]:
        run_frame(frame, initial_prev_bev)
    generation = tt_model.planner.generation

    def run_sequence():
        prev_bev = initial_prev_bev
        for frame in tt_frames[1:]:
            outputs = run_frame(frame, prev_bev)
            prev_bev = outputs[2]
        return outputs

    if execution == "live":
        scores, labels, _, boxes, valid = signposted_forward(device, run_sequence)[3]
        assert tt_model.planner.generation == generation, "the plan grew inside the measured sequence"
    else:
        host_img = to_conv_layout(img.flatten(0, 1), None, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        host_frames = [tt_model.host_frame(metas) for metas in frames[1:]]
        # The traces refill their previous BEV, a copy: the measured sequence starts from the live one's state.
        runner = TTBEVFormerTrace(tt_model, device, host_img, tt_frames[1], ttnn.clone(initial_prev_bev))

        def run_traced_sequence():
            for metas, host_frame in zip(frames[1:], host_frames):
                runner.upload(host_img)
                outputs = runner(metas, host_frame)
            return outputs

        try:
            scores, labels, boxes, valid = signposted_forward(device, run_traced_sequence)
            assert not runner.stale and tt_model.planner.generation == generation, "a frame in the window ran live"
        finally:
            runner.release()
        if NUM_PROFILED_FRAMES == 1:
            # The warmup's inputs and previous BEV, on one command queue: the same detections bit for bit.
            for name, want, have in zip(
                ("scores", "labels", "boxes", "valid"), warm_detections, (scores, labels, boxes, valid)
            ):
                assert torch.equal(want, ttnn.to_torch(have)), f"traced {name} differ from the live warmup's"
    start = time.perf_counter()
    tt_model.get_bboxes(scores, labels, boxes, valid)
    logger.info(f"get_bboxes' readback and compaction on the host: {(time.perf_counter() - start) * 1e3:.1f} ms")
