# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The detector replayed from traces against the same detector run live, frame by frame.

``test_bevformer`` pins the live detector to the reference; this pins the traces to the live
detector, which is cheaper than another reference run and isolates what tracing changes.
"""

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.bevformer.model_config import BEV_H, BEV_W
from models.experimental.bevformer.tests.common import (
    assert_pcc,
    build_reference_bevformer,
    frame_metas,
    random_image_batch,
    to_conv_layout,
)
from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_parameters
from models.experimental.bevformer.tt.tt_bevformer import TTBEVFormer
from models.experimental.bevformer.tt.tt_bevformer_trace import TTBEVFormerTrace

# Two live frames start every run, the first without a previous BEV. The traced frames then cover
# a replay, a replay at different counts, a frame no camera sees, a frame whose counts outgrow the
# plan between the readback and the plan, a planner grown between two frames, and a replay after
# each growth.
LIVE_FRAMES = 2
EMPTY_FRAME = 4
GROWTH_FRAME = 5
RESERVE_FRAME = 7
NUM_FRAMES = 9
# Per frame, the rig turns by this much, so each camera sees a different number of BEV cells.
RIG_YAW_STEP_DEG = 5.0
# The same programs on the same inputs: anything but bit-exact would be a replay reading a buffer
# the live run does not. PCC rather than equality only keeps the failure readable.
PCC_THRESHOLD = 0.99999
# Measured: base's three traces take well under this.
TRACE_REGION_SIZE = 256 << 20


def _turn_rig(meta, degrees):
    yaw = math.radians(degrees)
    turn = torch.eye(4)
    turn[:2, :2] = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    meta["lidar2img"] = (torch.tensor(meta["lidar2img"]) @ turn).tolist()


def _sequence(generator):
    frames = frame_metas(1, NUM_FRAMES, generator)
    for i, (meta,) in enumerate(frames):
        _turn_rig(meta, i * RIG_YAW_STEP_DEG)
    # Negating lidar2img puts every point behind every camera.
    (empty,) = frames[EMPTY_FRAME]
    empty["lidar2img"] = (-torch.tensor(empty["lidar2img"])).tolist()
    return frames


def _host_image(images):
    return to_conv_layout(images, None, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def _to_torch(tensors):
    return [ttnn.to_torch(t, cq_id=0).float() for t in tensors]


@torch.no_grad()
# Measured on base: about 170 s, or about 900 s with a cold kernel cache, as two command queues
# leave a smaller grid every kernel is compiled for again.
@pytest.mark.timeout(2400)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32 * 1024, "trace_region_size": TRACE_REGION_SIZE, "num_command_queues": 2}],
    indirect=True,
)
def test_bevformer_trace_matches_live(device, reset_seeds, expect_error):
    """The demo's pipelined use: two command queues, each frame's host inputs prepared ahead, and
    its images uploaded before the previous frame's detections are read, on the ops queue alone.
    The upload thus overlaps the previous frame and races the next frame's copy out of the staging
    buffer, which only the runner's events order. A race test cannot force the losing
    interleaving, so a missing event shows as a flaky failure, not a certain one. One command
    queue's replays are pinned by test_bevformer_perf's [trace] variant."""
    generator = torch.Generator().manual_seed(0)
    torch_model = build_reference_bevformer((BEV_H, BEV_W))
    images = [random_image_batch(generator=generator) for _ in range(NUM_FRAMES)]
    frames = _sequence(generator)
    tt_model = TTBEVFormer(create_bevformer_parameters(torch_model, images[0][None], device), device)

    def live_frame(i, frame, prev_bev):
        frame = tt_model.prepare_frame(frames[i], frame)
        cls_scores, bbox_preds, bev = tt_model(ttnn.to_device(_host_image(images[i]), device), frame, prev_bev)
        scores, labels, _, boxes, valid = tt_model.detect(cls_scores, bbox_preds)
        return frame, bev, _to_torch((scores, labels, boxes, valid, bev))

    expected, frame, prev_bev = [], None, None
    for i in range(NUM_FRAMES):
        frame, prev_bev, outputs = live_frame(i, frame, prev_bev)
        expected.append(outputs)

    frame = prev_bev = None
    for i in range(LIVE_FRAMES):
        frame, prev_bev, _ = live_frame(i, frame, prev_bev)
    # The live pass sized the planner for every frame, so only GROWTH_FRAME and RESERVE_FRAME grow it.
    planner = tt_model.planner
    runner = TTBEVFormerTrace(tt_model, device, _host_image(images[0]), frame, prev_bev, num_command_queues=2)
    size = planner.size

    def grow():
        # As counts with more queries per camera, and more cameras per query, would: a longer plan,
        # one more slot's ops and new constants.
        rows, slots = planner._constants_key[-1], planner.num_slots
        planner.reserve(planner.capacity + ttnn.TILE_SIZE, slots + 1)
        assert planner._constants_key[-1] > rows and planner.num_slots == slots + 1

    def size_and_grow(visibility):
        visible = size(visibility)
        grow()
        return visible

    replayed_addresses = set()
    try:
        runner.upload(_host_image(images[LIVE_FRAMES]))
        for i in range(LIVE_FRAMES, NUM_FRAMES):
            # GROWTH_FRAME grows between its counts' readback and its plan, RESERVE_FRAME before it starts.
            planner.size = size_and_grow if i == GROWTH_FRAME else size
            if i == RESERVE_FRAME:
                grow()
            outputs = runner(frames[i], tt_model.host_frame(frames[i]))
            if i + 1 < NUM_FRAMES:
                runner.upload(_host_image(images[i + 1]))
            # Blocking reads on the ops queue; the input queue's upload may still run.
            got = _to_torch((*outputs, runner.prev_bev))
            if i not in (GROWTH_FRAME, RESERVE_FRAME):
                replayed_addresses.add(tuple(t.buffer_address() for t in outputs))
            runner.capture_if_stale()
            logger.info(f"frame {i}: {len(runner.captures)} captures")
            assert len(runner.captures) == 1 + (i >= GROWTH_FRAME) + (i >= RESERVE_FRAME), f"frame {i}"

            for name, want, have in zip(("scores", "labels", "boxes", "valid", "BEV"), expected[i], got):
                if name in ("labels", "valid"):
                    assert torch.equal(want, have), f"frame {i}: {name} differ"
                else:
                    assert_pcc(want, have, PCC_THRESHOLD)
        # One set of output buffers per capture: replays keep their buffers.
        assert len(replayed_addresses) == 3, replayed_addresses
        # A frame's host inputs refill only that frame, before anything reaches the device.
        with expect_error(ValueError, "other metas"):
            runner(frames[0], tt_model.host_frame(frames[1]))
    finally:
        planner.size = size
        runner.release()
