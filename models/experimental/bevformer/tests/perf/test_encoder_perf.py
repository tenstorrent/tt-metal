# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the encoder device path.

Same inputs as ``test_encoder``'s second frame (the previous BEV and an ego shift): a PCC gate
that doubles as the warmup, then the forward captured as a trace and one signposted replay, so
the report covers already-compiled programs with no host dispatch in between. ``num_layers=1``
measures a single layer.

The camera geometry (``prepare_frame``) runs before the capture, once per frame as in the
detector. The capture is also the check that the forward has no host operations: any read or
write between host and device inside a capture region fails it.
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
from models.experimental.bevformer.tt.tt_encoder import GRID_DTYPE, TTBEVFormerEncoder


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
    torch_output = torch_model(
        inputs["bev_query"],
        inputs["value"],
        bev_h,
        bev_w,
        inputs["bev_pos"],
        torch.tensor(SPATIAL_SHAPES),
        inputs["img_metas"],
        prev_bev=prev_bev,
        shift=shift,
    )

    tt_model = TTBEVFormerEncoder(
        create_bevformer_encoder_parameters(torch_model, device),
        device,
        bev_h=bev_h,
        bev_w=bev_w,
        spatial_shapes=SPATIAL_SHAPES,
    )
    frame = tt_model.prepare_frame(inputs["img_metas"])
    # Uploaded once so the profiled replay measures the encoder, not the transfers.
    tt_inputs = dict(
        bev_query=_to_device(inputs["bev_query"].permute(1, 0, 2), device),
        value=_to_device(inputs["value"], device),
        bev_pos=_to_device(inputs["bev_pos"].permute(1, 0, 2), device),
        frame=frame,
        prev_bev=_to_device(prev_bev.permute(1, 0, 2), device),
        shift=_to_device(shift.view(1, 1, 1, 2), device, GRID_DTYPE, ttnn.ROW_MAJOR_LAYOUT),
    )

    def check(tt_output):
        assert_pcc(torch_output, ttnn.to_torch(tt_output).float().reshape(torch_output.shape), 0.99)

    # Doubles as the warmup: this call compiles the kernels and fills the program cache.
    check(tt_model(**tt_inputs))

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
        check(tt_output)
    finally:
        ttnn.release_trace(device, trace_id)
