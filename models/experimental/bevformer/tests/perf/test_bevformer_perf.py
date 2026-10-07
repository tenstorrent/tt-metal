# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the detector end to end: images to class logits, boxes and the frame's BEV.

Dummy weights and ``test_bevformer``'s inputs over three frames. The first frame, which has no
previous BEV, runs eagerly; the second is the warmup and the PCC gate, then is captured as a trace
and replayed once in a signposted region, so the report covers already-compiled programs with no
host dispatch in between. The capture is also the check that the forward has no host operations:
any read or write between host and device inside a capture region fails it.

The third frame replays the same trace, as a deployment would: ``prepare_frame`` refills the
frame's buffers in place on the host and the second frame's BEV is copied into the captured
previous-BEV input. Its ego turns and moves the other way, so a replay that kept the second
frame's shift or rotation would fail the check.
"""

import subprocess
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.experimental.bevformer.tests.common import (
    BEV_SHAPES,
    assert_channels_close,
    assert_pcc,
    build_reference_bevformer,
    frame_metas,
    random_image_batch,
    to_conv_layout,
)
from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_parameters
from models.experimental.bevformer.tt.tt_bevformer import TtBEVFormer

# test_bevformer's floor for dummy weights is above 0.99; this only catches a broken replay.
PCC_THRESHOLD = 0.99


def _head_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _check(expected, tt_outputs):
    cls_scores, bbox_preds, bev = expected
    tt_cls_scores, tt_bbox_preds, tt_bev = (ttnn.to_torch(t).float() for t in tt_outputs)
    assert_pcc(bev, tt_bev.reshape(bev.shape), PCC_THRESHOLD)
    assert_pcc(cls_scores[-1], tt_cls_scores[-1], PCC_THRESHOLD)
    assert_channels_close(bbox_preds[-1], tt_bbox_preds[-1], PCC_THRESHOLD)


@torch.no_grad()
# The reference runs the backbone on six 928x1600 images on the CPU four times: once to record
# the conv shapes and once per frame.
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params",
    # Headroom for the detector's recorded commands, not a measured size.
    [{"l1_small_size": 32 * 1024, "trace_region_size": 128 * 1024 * 1024}],
    indirect=True,
)
def test_bevformer_perf(device, reset_seeds):
    logger.info(f"device-perf run of commit {_head_sha()}")
    torch_model = build_reference_bevformer(BEV_SHAPES["base"])
    img = random_image_batch()[None]
    frames = frame_metas(1, 3, torch.Generator().manual_seed(0))
    for meta in frames[2]:
        meta["can_bus"][:2] *= -2
        meta["can_bus"][-1] *= -2

    expected, torch_prev = [], None
    for metas in frames:
        outputs = torch_model(img, metas, prev_bev=torch_prev)
        expected.append(outputs)
        torch_prev = outputs[2]

    tt_model = TtBEVFormer(create_bevformer_parameters(torch_model, img, device), device)
    # Uploaded once so the profiled replay measures the detector, not the transfer.
    tt_img = to_conv_layout(img.flatten(0, 1), device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    frame = tt_model.prepare_frame(frames[0])
    *_, prev_bev = tt_model(tt_img, frame)
    frame = tt_model.prepare_frame(frames[1], frame)
    # Doubles as the warmup of the path with a previous BEV, the one the trace records.
    _check(expected[1], tt_model(tt_img, frame, prev_bev=prev_bev))

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    tt_outputs = tt_model(tt_img, frame, prev_bev=prev_bev)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    ttnn.synchronize_device(device)
    # Drains and resets the device profiler buffers so the signposted region starts from
    # empty; the warmup's markers would otherwise eat into the same budget.
    ttnn.ReadDeviceProfiler(device)
    signpost("start")
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    signpost("stop")

    try:
        _check(expected[1], tt_outputs)
        start = time.perf_counter()
        tt_model.prepare_frame(frames[2], frame)
        logger.info(f"prepare_frame refill on the host: {(time.perf_counter() - start) * 1e3:.1f} ms")
        ttnn.copy(tt_outputs[2], prev_bev)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        _check(expected[2], tt_outputs)
    finally:
        ttnn.release_trace(device, trace_id)
