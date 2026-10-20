# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for a single BEVFormer encoder layer, called directly.

A 100x100 or 200x200 BEV grid over nuScenes' six-camera rig, bs=1 and no previous BEV: a PCC gate
that doubles as the warmup, then one signposted run. ``live`` calls the layer between ``start`` and
``stop``, so the stage signposts (``BEVLayer TSA/SCA/FFN``) and any host dispatch are in the
report. ``trace`` captures the layer and replays it once in that window, so the report is
already-compiled programs with no host dispatch; the replay carries only ``start`` and ``stop``.

What the encoder builds once per frame and shares between its layers, the camera geometry's
rebatch plan, the spatial cross-attention's layer inputs and the self-attention's reference
points, is built before the measured run, so the region is the cost the encoder repeats per layer.
"""

import pytest
import torch

import ttnn
from models.experimental.bevformer.model_config import SPATIAL_SHAPES
from models.experimental.bevformer.tests.common import (
    assert_pcc,
    build_reference_encoder,
    camera_rows,
    random_encoder_inputs,
    signposted_forward,
    signposted_trace,
    to_device,
)
from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_encoder_parameters
from models.experimental.bevformer.tt.tt_encoder import TTBEVFormerEncoder


@torch.no_grad()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("bev_shape", [(100, 100), (200, 200)], ids=["bev_100x100", "bev_200x200"])
@pytest.mark.parametrize(
    "traced, device_params",
    [
        (False, {"l1_small_size": 32 * 1024}),
        # Headroom for the layer's recorded commands, not a measured size.
        (True, {"l1_small_size": 32 * 1024, "trace_region_size": 32 * 1024 * 1024}),
    ],
    ids=["live", "trace"],
    indirect=["device_params"],
)
def test_encoder_layer_perf(device, reset_seeds, traced, bev_shape):
    bev_h, bev_w = bev_shape
    # A one-layer encoder is that layer: without a previous BEV the encoder only builds the frame
    # inputs around it.
    torch_model = build_reference_encoder(1)
    inputs = random_encoder_inputs(bev_shape, 1)
    torch_output = torch_model(
        inputs["bev_query"],
        inputs["value"],
        bev_h,
        bev_w,
        inputs["bev_pos"],
        torch.tensor(SPATIAL_SHAPES),
        inputs["img_metas"],
    )

    encoder = TTBEVFormerEncoder(
        create_bevformer_encoder_parameters(torch_model, device),
        device,
        bev_h=bev_h,
        bev_w=bev_w,
        spatial_shapes=SPATIAL_SHAPES,
    )
    layer = encoder.layers[0]
    plan = encoder.prepare_frame(inputs["img_metas"])
    sca_inputs = layer.spatial_cross_attention.layer_inputs(plan)
    query = to_device(inputs["bev_query"].permute(1, 0, 2), device)
    value = to_device(camera_rows(inputs["value"]), device)
    bev_pos = to_device(inputs["bev_pos"].permute(1, 0, 2), device)

    ref_2d = encoder.reference_points(1)

    def run():
        return layer(query, value, bev_pos, None, ref_2d, sca_inputs)

    def check(tt_output):
        assert_pcc(torch_output, ttnn.to_torch(tt_output).float().reshape(torch_output.shape), 0.997)

    # Doubles as the warmup: this call compiles the kernels and fills the program cache.
    check(run())

    if traced:
        with signposted_trace(device, run) as (_, tt_output):
            check(tt_output)
    else:
        check(signposted_forward(device, run))
