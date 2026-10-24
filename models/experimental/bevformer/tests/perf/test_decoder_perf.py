# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the six-layer detection decoder over the base 200x200 BEV map.

``test_decoder``'s base inputs: a PCC gate that doubles as the warmup, then the forward
captured as a trace and one signposted replay, so the report covers already-compiled programs
with no host dispatch in between.

The capture is also the check that the forward has no host operations: any read or write
between host and device inside a capture region fails it. The replay's outputs are checked
against the reference after the signposted region.
"""

import pytest
import torch
from tracy import signpost

import ttnn
from models.experimental.bevformer.model_config import GRID_DTYPE
from models.experimental.bevformer.tests.common import (
    BEV_SHAPES,
    build_reference_decoder,
    build_reg_branches,
    check_decoder_outputs,
    random_decoder_inputs,
    reference_box_codes,
    to_device,
)
from models.experimental.bevformer.tt.model_preprocessing import (
    create_decoder_parameters,
    create_reg_branch_parameters,
)
from models.experimental.bevformer.tt.tt_decoder import TTDetectionTransformerDecoder


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    # Headroom for the six layers' recorded commands, not a measured size.
    [{"trace_region_size": 32 * 1024 * 1024}],
    indirect=True,
)
def test_decoder_perf(device, reset_seeds):
    bev_shape = BEV_SHAPES["base"]
    torch_model = build_reference_decoder()
    reg_branches = build_reg_branches()
    inputs = random_decoder_inputs(bev_shape, 1, seed=0)
    outputs, points = torch_model(**inputs, spatial_shapes=torch.tensor([bev_shape]), reg_branches=reg_branches)
    torch_outputs = (outputs, points, reference_box_codes(outputs, reg_branches))

    tt_model = TTDetectionTransformerDecoder(create_decoder_parameters(torch_model, device), device, bev_shape)
    tt_reg_branches = create_reg_branch_parameters(reg_branches, device)
    # Uploaded once so the profiled replay measures the decoder, not the transfers.
    tt_inputs = {
        key: to_device(tensor, device, GRID_DTYPE if key == "reference_points" else ttnn.bfloat16)
        for key, tensor in inputs.items()
    }

    def run():
        return tt_model(**tt_inputs, reg_branches=tt_reg_branches)

    def check(tt_outputs):
        tt_outputs = tuple(ttnn.to_torch(t).float() for t in tt_outputs)
        check_decoder_outputs(torch_outputs, tt_outputs, inputs["reference_points"], bev_shape)

    check(run())

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
        check(tt_outputs)
    finally:
        ttnn.release_trace(device, trace_id)
