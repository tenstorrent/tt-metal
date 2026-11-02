# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the detection head and the coder's top-k over the base 200x200 BEV map.

``test_head``'s base inputs: a PCC gate that doubles as the warmup, then the head and the
coder's device top-k captured as a trace and one signposted replay, so the report covers
already-compiled programs with no host dispatch in between. ``test_head_detections`` checks the
top-k itself.

The capture is also the check that the forward has no host operations: any read or write
between host and device inside a capture region fails it. The replay's head outputs are checked
against the reference after the signposted region.
"""

import pytest
import torch
from tracy import signpost

import ttnn
from models.experimental.bevformer.tests.common import (
    BEV_SHAPES,
    build_reference_head,
    check_head_outputs,
    random_bev,
    to_device,
)
from models.experimental.bevformer.tt.model_preprocessing import create_head_parameters
from models.experimental.bevformer.tt.tt_head import TTBEVFormerHead
from models.experimental.bevformer.tt.tt_nms_free_coder import TTNMSFreeCoder


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    # Headroom for the head's and coder's recorded commands, not a measured size.
    [{"trace_region_size": 32 * 1024 * 1024}],
    indirect=True,
)
def test_head_perf(device, reset_seeds):
    bev_shape = BEV_SHAPES["base"]
    torch_model = build_reference_head(bev_shape)
    tt_model = TTBEVFormerHead(create_head_parameters(torch_model, device), device)
    coder = TTNMSFreeCoder(device)

    generator = torch.Generator().manual_seed(0)
    bev_embed = random_bev(bev_shape, 1, generator).permute(1, 0, 2).contiguous()
    torch_outputs = torch_model(bev_embed)
    # Uploaded once so the profiled replay measures the head, not the transfer.
    tt_bev_embed = to_device(bev_embed, device)

    def run():
        cls_scores, bbox_preds = tt_model(tt_bev_embed)
        coder.topk(cls_scores[-1], bbox_preds[-1])
        return cls_scores, bbox_preds

    check_head_outputs(torch_outputs, run())

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    tt_outputs = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    ttnn.synchronize_device(device)
    # Drains and resets the device profiler buffers so the signposted region starts from
    # empty; the warmup's markers would otherwise eat into the same budget.
    ttnn.ReadDeviceProfiler(device)
    signpost("start")
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    signpost("stop")

    try:
        check_head_outputs(torch_outputs, tt_outputs)
    finally:
        ttnn.release_trace(device, trace_id)
