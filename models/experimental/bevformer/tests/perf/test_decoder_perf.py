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
    assert_channels_close,
    assert_pcc,
    build_reference_decoder,
    build_reg_branches,
    random_decoder_inputs,
)
from models.experimental.bevformer.tt.model_preprocessing import (
    create_decoder_parameters,
    create_reg_branch_parameters,
)
from models.experimental.bevformer.tt.tt_decoder import TtDetectionTransformerDecoder


def _check(torch_outputs, tt_outputs):
    """``torch_outputs`` and ``tt_outputs`` are (layer outputs, refined points, box codes)."""
    tt_outputs = tuple(ttnn.to_torch(t).float() for t in tt_outputs)
    for tensor in tt_outputs:
        assert torch.isfinite(tensor).all(), "non-finite values in the decoder outputs"
    for torch_output, tt_output in zip(torch_outputs[:2], tt_outputs[:2], strict=True):
        for expected_layer, actual_layer in zip(torch_output, tt_output, strict=True):
            assert_pcc(expected_layer, actual_layer, 0.99)
    assert_channels_close(torch_outputs[2], tt_outputs[2])


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
    # The TT decoder also returns each layer's reg branch on its output.
    box_codes = torch.stack([branch(out.permute(1, 0, 2)) for branch, out in zip(reg_branches, outputs)])
    torch_outputs = (outputs, points, box_codes)

    tt_model = TtDetectionTransformerDecoder(create_decoder_parameters(torch_model, device), device, bev_shape)
    tt_reg_branches = create_reg_branch_parameters(reg_branches, device)
    # Uploaded once so the profiled replay measures the decoder, not the transfers.
    tt_inputs = {
        key: ttnn.from_torch(
            tensor,
            dtype=GRID_DTYPE if key == "reference_points" else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        for key, tensor in inputs.items()
    }

    def run():
        return tt_model(**tt_inputs, reg_branches=tt_reg_branches)

    # Doubles as the warmup: this call compiles the kernels and fills the program cache.
    _check(torch_outputs, run())

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
        _check(torch_outputs, tt_outputs)
    finally:
        ttnn.release_trace(device, trace_id)
