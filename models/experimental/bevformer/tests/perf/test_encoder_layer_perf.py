# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for a single BEVFormer encoder layer, called directly.

A 100x100 BEV grid over nuScenes' six-camera rig, bs=1 and no previous BEV: a PCC gate that
doubles as the warmup, then the layer captured as a trace and one signposted replay, so the report
covers already-compiled programs with no host dispatch in between.

What the encoder builds once per frame and shares between its layers, the camera geometry's
rebatch plan, the spatial cross-attention's frame inputs and the self-attention's reference
points, is built before the capture, so the measured region is the cost the encoder repeats per
layer. The layer's stages are signposted (``BEVLayer TSA/SCA/FFN Start/Complete``) when it is
captured; the replay itself carries only the ``start`` and ``stop`` signposts.
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
    signposted_trace,
    to_device,
)
from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_encoder_parameters
from models.experimental.bevformer.tt.tt_encoder import TTBEVFormerEncoder

BEV_SHAPE = (100, 100)


@torch.no_grad()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    # Headroom for the layer's recorded commands, not a measured size.
    [{"l1_small_size": 32 * 1024, "trace_region_size": 32 * 1024 * 1024}],
    indirect=True,
)
def test_encoder_layer_perf(device, reset_seeds):
    bev_h, bev_w = BEV_SHAPE
    # A one-layer encoder is that layer: without a previous BEV the encoder only builds the frame
    # inputs around it.
    torch_model = build_reference_encoder(1)
    inputs = random_encoder_inputs(BEV_SHAPE, 1)
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
    sca_frame = layer.spatial_cross_attention.frame_inputs(plan)
    query = to_device(inputs["bev_query"].permute(1, 0, 2), device)
    value = to_device(camera_rows(inputs["value"]), device)
    bev_pos = to_device(inputs["bev_pos"].permute(1, 0, 2), device)

    ref_2d = encoder.reference_points(1)

    def run():
        return layer(query, value, bev_pos, None, ref_2d, sca_frame)

    def check(tt_output):
        assert_pcc(torch_output, ttnn.to_torch(tt_output).float().reshape(torch_output.shape), 0.997)

    # Doubles as the warmup: this call compiles the kernels and fills the program cache.
    check(run())

    with signposted_trace(device, run) as (_, tt_output):
        check(tt_output)
