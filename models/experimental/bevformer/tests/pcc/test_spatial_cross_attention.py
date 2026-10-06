# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.bevformer.tests.backbone_common import assert_pcc
from models.experimental.bevformer.tests.encoder_common import (
    BEV_SHAPES,
    SPATIAL_SHAPES,
    build_reference_encoder,
    random_bev,
    random_encoder_inputs,
)
from models.experimental.bevformer.tt.model_preprocessing import create_spatial_cross_attention_parameters
from models.experimental.bevformer.tt.tt_encoder import GRID_DTYPE
from models.experimental.bevformer.tt.tt_ms_deformable_attention import fp32_grid_sample_config
from models.experimental.bevformer.tt.tt_spatial_cross_attention import TTSpatialCrossAttention, build_rebatch_plan

CASES = [
    # (name, bev_shape, batch_size)
    ("tiny", BEV_SHAPES["tiny"], 1),
    ("base", BEV_SHAPES["base"], 1),
    # bs=2, so a batch mix-up in the rebatch gather or the scatter back shows.
    ("tiny-bs2", BEV_SHAPES["tiny"], 2),
    # Non-square, so a swapped (h, w) in the pillar grid shows.
    ("50x100", (50, 100), 1),
]


def _to_device(tensor, device):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


@torch.no_grad()
@pytest.mark.parametrize("name, bev_shape, batch_size", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_spatial_cross_attention(device, reset_seeds, name, bev_shape, batch_size):
    """The first layer's cross-attention: each BEV query gathered into the cameras its pillar
    projects into, attending to their FPN features, averaged over those cameras."""
    bev_h, bev_w = bev_shape
    encoder = build_reference_encoder(num_layers=1)
    torch_model = encoder.layers[0].attentions[1]
    inputs = random_encoder_inputs(bev_shape, batch_size)
    query = random_bev(bev_shape, batch_size, torch.Generator().manual_seed(1)).permute(1, 0, 2)
    reference_points_cam, bev_mask = encoder.point_sampling(bev_h, bev_w, batch_size, inputs["img_metas"])
    assert bev_mask.any(), "no BEV query projects into any camera"

    torch_output = torch_model(
        query,
        value=inputs["value"],
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask,
        spatial_shapes=torch.tensor(SPATIAL_SHAPES),
    )

    tt_model = TTSpatialCrossAttention(
        device,
        create_spatial_cross_attention_parameters(torch_model, device),
        spatial_shapes=SPATIAL_SHAPES,
        grid_dtype=GRID_DTYPE,
        grid_sample_compute_config=fp32_grid_sample_config(device),
    )
    rebatch_plan = build_rebatch_plan(reference_points_cam, bev_mask, tt_model.embed_dims, device, GRID_DTYPE)
    tt_output = tt_model(
        query=_to_device(query, device), value=_to_device(inputs["value"], device), rebatch_plan=rebatch_plan
    )
    tt_output = ttnn.to_torch(tt_output).float().reshape(torch_output.shape)
    # comp_pcc zeroes NaN and Inf before correlating, so they must be ruled out here.
    assert torch.isfinite(tt_output).all(), "non-finite values in the cross-attention output"
    assert_pcc(torch_output, tt_output, 0.99)
    # The attended part alone: the residual would carry the PCC on its own.
    assert_pcc(torch_output - query, tt_output - query, 0.99)
