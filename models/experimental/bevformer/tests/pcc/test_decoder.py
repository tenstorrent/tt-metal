# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.bevformer.tests.backbone_common import assert_pcc
from models.experimental.bevformer.tests.decoder_common import (
    BEV_SHAPES,
    build_reference_decoder,
    build_reg_branches,
    layer_metrics,
    random_decoder_inputs,
)
from models.experimental.bevformer.tt.model_preprocessing_decoder import (
    create_decoder_parameters,
    create_reg_branch_parameters,
)
from models.experimental.bevformer.tt.tt_decoder import GRID_DTYPE, TtDetectionTransformerDecoder

CASES = [
    # (name, bev_shape, batch_size, traced, batch_first)
    ("tiny", BEV_SHAPES["tiny"], 1, False, False),
    ("tiny-traced", BEV_SHAPES["tiny"], 1, True, False),
    ("base", BEV_SHAPES["base"], 1, False, False),
    ("base-traced", BEV_SHAPES["base"], 1, True, False),
    ("tiny-bs2", BEV_SHAPES["tiny"], 2, False, False),
    # Non-square, so a swapped (H, W) anywhere in the grid scale or value layout shows.
    ("50x100", (50, 100), 1, False, False),
    # bs=2, so a batch/query mix-up in the batch-first inputs or outputs shows.
    ("tiny-bs2-batch-first", BEV_SHAPES["tiny"], 2, False, True),
]

SEQUENCE_FIRST_INPUTS = ("query", "value", "query_pos")


def _to_device(tensor, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _input_dtype(name):
    """The decoder requires ``GRID_DTYPE`` (float32) reference points; the rest is bfloat16."""
    return GRID_DTYPE if name == "reference_points" else ttnn.bfloat16


def _host_input(name, tensor, batch_first):
    """The reference's sequence-first inputs, permuted to batch-first for a batch-first decoder."""
    return tensor.permute(1, 0, 2).contiguous() if batch_first and name in SEQUENCE_FIRST_INPUTS else tensor


def _check(torch_outputs, tt_outputs, input_reference_points, bev_shape, batch_first):
    tt_outputs = tuple(ttnn.to_torch(t).float() for t in tt_outputs)
    if batch_first:
        tt_outputs = (tt_outputs[0].permute(0, 2, 1, 3), tt_outputs[1])
    # comp_pcc zeroes NaN and Inf before correlating, so they must be ruled out here.
    for name, tensor in zip(("output", "reference points"), tt_outputs):
        assert torch.isfinite(tensor).all(), f"non-finite values in the decoder {name}"
    for layer, metrics in enumerate(layer_metrics(torch_outputs, tt_outputs, input_reference_points, bev_shape)):
        logger.info(f"layer {layer}: " + ", ".join(f"{key} {value:.5f}" for key, value in metrics.items()))
    # Per layer: the first layers' accuracy would hide a failing last layer in the stack.
    for expected, actual in zip(torch_outputs, tt_outputs, strict=True):
        for expected_layer, actual_layer in zip(expected, actual, strict=True):
            assert_pcc(expected_layer, actual_layer, 0.99)


@torch.no_grad()
@pytest.mark.parametrize("name, bev_shape, batch_size, traced, batch_first", CASES, ids=[case[0] for case in CASES])
# Headroom for the six layers' recorded commands, not a measured size.
@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
def test_decoder(device, reset_seeds, name, bev_shape, batch_size, traced, batch_first):
    torch_model = build_reference_decoder()
    reg_branches = build_reg_branches()
    spatial_shapes = torch.tensor([bev_shape])

    def reference(inputs):
        return torch_model(**inputs, spatial_shapes=spatial_shapes, reg_branches=reg_branches)

    tt_model = TtDetectionTransformerDecoder(
        create_decoder_parameters(torch_model, device), device, bev_shape, batch_first=batch_first
    )
    tt_reg_branches = create_reg_branch_parameters(reg_branches, device)

    inputs = random_decoder_inputs(bev_shape, batch_size, seed=0)
    tt_inputs = {
        key: _to_device(_host_input(key, tensor, batch_first), device, _input_dtype(key))
        for key, tensor in inputs.items()
    }

    def run():
        return tt_model(**tt_inputs, reg_branches=tt_reg_branches)

    # The first run compiles; a second one must only hit the program cache.
    run()
    num_programs = device.num_program_cache_entries()

    if not traced:
        tt_outputs = run()
        assert device.num_program_cache_entries() == num_programs
        _check(reference(inputs), tt_outputs, inputs["reference_points"], bev_shape, batch_first)
        return

    # Capture fails on any host read or write in forward. Replaying on fresh inputs shows
    # the replay, not a leftover eager result, fills the captured output buffers.
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    tt_outputs = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    replay_inputs = random_decoder_inputs(bev_shape, batch_size, seed=1)
    for key, tensor in replay_inputs.items():
        host = ttnn.from_torch(_host_input(key, tensor, batch_first), dtype=_input_dtype(key), layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, tt_inputs[key])
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    try:
        _check(reference(replay_inputs), tt_outputs, replay_inputs["reference_points"], bev_shape, batch_first)
    finally:
        ttnn.release_trace(device, trace_id)
