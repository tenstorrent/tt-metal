# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the encoder device path.

Inputs shaped like ``test_encoder``'s second frame, a smooth random previous BEV and an ego
shift: a PCC gate that doubles as the warmup, then the forward captured as a trace and one
signposted replay, so the report covers already-compiled programs with no host dispatch in
between. ``num_layers=1`` measures a single layer.

The camera geometry (``prepare_frame``) runs before the capture, once per frame as in the
detector, which refills the same plan for every later frame. The capture is also the check that
the forward has no host operations: any read or write between host and device inside a capture
region fails it. After the profiled replay, the trace replays once more on the next frame of the
same rig: a new previous BEV and shift written into the captured buffers and the plan refilled
in place with the same geometry, checked against the reference.
"""

import subprocess

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.experimental.bevformer.tests.backbone_common import assert_pcc
from models.experimental.bevformer.tests.encoder_common import (
    BEV_SHAPES,
    NUM_LAYERS,
    SPATIAL_SHAPES,
    build_reference_encoder,
    ego_shift,
    random_bev,
    random_encoder_inputs,
)
from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_encoder_parameters
from models.experimental.bevformer.tt.tt_common import GRID_DTYPE
from models.experimental.bevformer.tt.tt_encoder import TTBEVFormerEncoder


def _head_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _to_device(tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device)


@torch.no_grad()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("num_layers", [NUM_LAYERS, 1])
@pytest.mark.parametrize(
    "device_params",
    # Headroom for the encoder's recorded commands, not a measured size.
    [{"l1_small_size": 32 * 1024, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
def test_encoder_perf(device, reset_seeds, num_layers):
    logger.info(f"device-perf run of commit {_head_sha()}")
    bev_h, bev_w = BEV_SHAPES["base"]
    torch_model = build_reference_encoder(num_layers)
    inputs = random_encoder_inputs((bev_h, bev_w), 1)
    prev_bev = random_bev((bev_h, bev_w), 1, torch.Generator().manual_seed(1))
    shift = ego_shift(1)

    def reference(frame_prev_bev, frame_shift):
        return torch_model(
            inputs["bev_query"],
            inputs["value"],
            bev_h,
            bev_w,
            inputs["bev_pos"],
            torch.tensor(SPATIAL_SHAPES),
            inputs["img_metas"],
            prev_bev=frame_prev_bev,
            shift=frame_shift,
        )

    torch_output = reference(prev_bev, shift)

    tt_model = TTBEVFormerEncoder(
        create_bevformer_encoder_parameters(torch_model, device),
        device,
        bev_h=bev_h,
        bev_w=bev_w,
        spatial_shapes=SPATIAL_SHAPES,
    )
    plan = tt_model.prepare_frame(inputs["img_metas"])
    # Uploaded once so the profiled replay measures the encoder, not the transfers.
    tt_inputs = dict(
        bev_query=_to_device(inputs["bev_query"].permute(1, 0, 2), device),
        value=_to_device(inputs["value"], device),
        bev_pos=_to_device(inputs["bev_pos"].permute(1, 0, 2), device),
        plan=plan,
        prev_bev=_to_device(prev_bev.permute(1, 0, 2), device),
        shift=_to_device(shift.view(1, 1, 1, 2), device, GRID_DTYPE, ttnn.ROW_MAJOR_LAYOUT),
    )

    def check(expected, tt_output):
        assert_pcc(expected, ttnn.to_torch(tt_output).float().reshape(expected.shape), 0.997)

    # Doubles as the warmup: this call compiles the kernels and fills the program cache.
    check(torch_output, tt_model(**tt_inputs))

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    tt_output = tt_model(**tt_inputs)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    ttnn.synchronize_device(device)
    # Drains and resets the device profiler buffers so the signposted region starts from
    # empty; the warmup's markers would otherwise eat into the same budget.
    ttnn.ReadDeviceProfiler(device)
    signpost("start")
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    signpost("stop")

    try:
        check(torch_output, tt_output)
        # The next frame of the same rig through the same trace: the replay must read the new
        # previous BEV and shift from the captured buffers; the refilled plan keeps its geometry.
        next_prev_bev = random_bev((bev_h, bev_w), 1, torch.Generator().manual_seed(2))
        next_shift = -2 * shift
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(next_prev_bev.permute(1, 0, 2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            tt_inputs["prev_bev"],
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(next_shift.view(1, 1, 1, 2), dtype=GRID_DTYPE, layout=ttnn.ROW_MAJOR_LAYOUT),
            tt_inputs["shift"],
        )
        tt_model.prepare_frame(inputs["img_metas"], plan=plan)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        check(reference(next_prev_bev, next_shift), tt_output)
    finally:
        ttnn.release_trace(device, trace_id)
