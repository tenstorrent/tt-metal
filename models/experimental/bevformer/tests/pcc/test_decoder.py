# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.bevformer.tests.decoder_common import (
    BEV_SHAPES,
    THRESHOLDS,
    build_reference_decoder,
    build_reg_branches,
    layer_metrics,
    random_decoder_inputs,
    threshold_failures,
)
from models.experimental.bevformer.tt.model_preprocessing_decoder import (
    create_decoder_parameters,
    create_reg_branch_parameters,
)
from models.experimental.bevformer.tt.tt_decoder import GRID_DTYPE, TtDetectionTransformerDecoder

CASES = [
    # (name, bev_shape, batch_size, traced, thresholds)
    ("tiny", BEV_SHAPES["tiny"], 1, False, THRESHOLDS["tiny"]),
    ("tiny-traced", BEV_SHAPES["tiny"], 1, True, THRESHOLDS["tiny"]),
    ("base", BEV_SHAPES["base"], 1, False, THRESHOLDS["base"]),
    ("base-traced", BEV_SHAPES["base"], 1, True, THRESHOLDS["base"]),
    ("tiny-bs2", BEV_SHAPES["tiny"], 2, False, THRESHOLDS["tiny"]),
    # Non-square, so a swapped (H, W) anywhere in the grid scale or value layout shows. The
    # tiny bounds were measured to hold for it.
    ("50x100", (50, 100), 1, False, THRESHOLDS["tiny"]),
]


def _to_device(tensor, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _input_dtype(name):
    """The decoder requires ``GRID_DTYPE`` (float32) reference points; the rest is bfloat16."""
    return GRID_DTYPE if name == "reference_points" else ttnn.bfloat16


def _check(torch_outputs, tt_outputs, input_reference_points, bev_shape, thresholds):
    tt_outputs = tuple(ttnn.to_torch(t).float() for t in tt_outputs)
    # comp_pcc zeroes NaN and Inf before correlating, so they must be ruled out here.
    for name, tensor in zip(("output", "reference points"), tt_outputs):
        assert torch.isfinite(tensor).all(), f"non-finite values in the decoder {name}"
    # Every layer is measured before anything is asserted, so a failure reports them all.
    failures = []
    for layer, metrics in enumerate(layer_metrics(torch_outputs, tt_outputs, input_reference_points, bev_shape)):
        logger.info(f"layer {layer}: " + ", ".join(f"{key} {value:.5f}" for key, value in metrics.items()))
        failures += [f"layer {layer} {failure}" for failure in threshold_failures(metrics, thresholds, layer)]
    assert not failures, "; ".join(failures)


@torch.no_grad()
@pytest.mark.parametrize("name, bev_shape, batch_size, traced, thresholds", CASES, ids=[case[0] for case in CASES])
# Headroom for the six layers' recorded commands, not a measured size.
@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
def test_decoder(device, reset_seeds, name, bev_shape, batch_size, traced, thresholds):
    torch_model = build_reference_decoder()
    reg_branches = build_reg_branches()
    spatial_shapes = torch.tensor([bev_shape])

    def reference(inputs):
        return torch_model(**inputs, spatial_shapes=spatial_shapes, reg_branches=reg_branches)

    tt_model = TtDetectionTransformerDecoder(create_decoder_parameters(torch_model, device), device, bev_shape)
    tt_reg_branches = create_reg_branch_parameters(reg_branches, device)

    inputs = random_decoder_inputs(bev_shape, batch_size, seed=0)
    tt_inputs = {key: _to_device(tensor, device, _input_dtype(key)) for key, tensor in inputs.items()}

    def run():
        return tt_model(**tt_inputs, reg_branches=tt_reg_branches)

    # The first run compiles; a second one must only hit the program cache.
    run()
    num_programs = device.num_program_cache_entries()

    if not traced:
        tt_outputs = run()
        assert device.num_program_cache_entries() == num_programs
        _check(reference(inputs), tt_outputs, inputs["reference_points"], bev_shape, thresholds)
        return

    # Capture fails on any host read or write in forward. Replaying on fresh inputs shows
    # the replay, not a leftover eager result, fills the captured output buffers.
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    tt_outputs = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    replay_inputs = random_decoder_inputs(bev_shape, batch_size, seed=1)
    for key, tensor in replay_inputs.items():
        host = ttnn.from_torch(tensor, dtype=_input_dtype(key), layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, tt_inputs[key])
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    try:
        _check(reference(replay_inputs), tt_outputs, replay_inputs["reference_points"], bev_shape, thresholds)
    finally:
        ttnn.release_trace(device, trace_id)
