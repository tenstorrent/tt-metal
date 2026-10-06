# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.bevformer.reference.point_sampling_3d_2d import camera_geometry
from models.experimental.bevformer.tests.backbone_common import assert_pcc
from models.experimental.bevformer.tests.encoder_common import (
    BEV_SHAPES,
    NUM_CAMS,
    SPATIAL_SHAPES,
    build_reference_encoder,
    img_metas,
    pc_range,
    random_bev,
    random_encoder_inputs,
)
from models.experimental.bevformer.tt.model_preprocessing import create_spatial_cross_attention_parameters
from models.experimental.bevformer.tt.tt_common import GRID_DTYPE
from models.experimental.bevformer.tt.tt_ms_deformable_attention import fp32_grid_sample_config
from models.experimental.bevformer.tt.tt_spatial_cross_attention import (
    TTSpatialCrossAttention,
    build_rebatch_plan,
    full_capacity,
    update_rebatch_plan,
)

CASES = [
    # (name, bev_shape, batch_size, geometry)
    ("tiny", BEV_SHAPES["tiny"], 1, "nuscenes"),
    ("base", BEV_SHAPES["base"], 1, "nuscenes"),
    # The second sample's rig is turned, so its cameras see other cells: as upstream, both samples
    # gather the first sample's visible queries, each with its own points and camera count.
    ("tiny-bs2-turned-rig", BEV_SHAPES["tiny"], 2, "nuscenes-turned"),
    # Non-square, so a swapped (h, w) in the pillar grid shows.
    ("50x100", (50, 100), 1, "nuscenes"),
    ("tiny-carla", BEV_SHAPES["tiny"], 1, "carla"),
    # No camera sees anything: every row lands in the sink, and the output is the query plus the
    # output projection's bias, as in the reference.
    ("tiny-empty", BEV_SHAPES["tiny"], 1, "empty"),
    # Every camera sees exactly one plan's worth of queries: no padding rows.
    ("tiny-full-plan", BEV_SHAPES["tiny"], 1, "full-plan"),
]
# Two tiles: one tile is also the plan's minimum capacity, which would hide whether the capacity
# came from the mask.
FULL_PLAN_ROWS = 2 * ttnn.TILE_SIZE


def _geometry(kind, bev_shape, batch_size):
    """``reference_points_cam`` and ``bev_mask`` for a case."""
    bev_h, bev_w = bev_shape
    if kind in ("nuscenes", "nuscenes-turned", "carla"):
        preset = "carla_tiny" if kind == "carla" else "nuscenes_base"
        metas = img_metas(batch_size, preset, yaw_step_deg=40.0 if kind == "nuscenes-turned" else 0.0)
        return camera_geometry(metas, bev_h, bev_w, 4, pc_range(preset))
    generator = torch.Generator().manual_seed(2)
    num_queries = bev_h * bev_w
    points = torch.rand(NUM_CAMS, batch_size, num_queries, 4, 2, generator=generator)
    mask = torch.zeros(NUM_CAMS, batch_size, num_queries, 4, dtype=torch.bool)
    if kind == "full-plan":
        for cam in range(NUM_CAMS):
            seen = torch.randperm(num_queries, generator=generator)[:FULL_PLAN_ROWS]
            mask[cam, :, seen, : 1 + cam % 4] = True
    return points, mask


def _to_device(tensor, device):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _tt_model(torch_model, device):
    return TTSpatialCrossAttention(
        create_spatial_cross_attention_parameters(torch_model, device),
        device,
        spatial_shapes=SPATIAL_SHAPES,
        grid_dtype=GRID_DTYPE,
        grid_sample_compute_config=fp32_grid_sample_config(device),
    )


@torch.no_grad()
@pytest.mark.parametrize("name, bev_shape, batch_size, geometry", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_spatial_cross_attention(device, reset_seeds, name, bev_shape, batch_size, geometry):
    """The first layer's cross-attention: each BEV query gathered into the cameras its pillar
    projects into, attending to their FPN features, averaged over those cameras."""
    torch_model = build_reference_encoder(num_layers=1).layers[0].attentions[1]
    inputs = random_encoder_inputs(bev_shape, batch_size)
    query = random_bev(bev_shape, batch_size, torch.Generator().manual_seed(1)).permute(1, 0, 2)
    reference_points_cam, bev_mask = _geometry(geometry, bev_shape, batch_size)

    torch_output = torch_model(query, inputs["value"], reference_points_cam, bev_mask, torch.tensor(SPATIAL_SHAPES))

    tt_model = _tt_model(torch_model, device)
    plan = build_rebatch_plan(reference_points_cam, bev_mask, tt_model.embed_dims, device, GRID_DTYPE)
    if geometry == "full-plan":
        assert plan.capacity == FULL_PLAN_ROWS
    tt_output = tt_model(_to_device(query, device), _to_device(inputs["value"], device), tt_model.frame_inputs(plan))
    tt_output = ttnn.to_torch(tt_output).float().reshape(torch_output.shape)
    # comp_pcc zeroes NaN and Inf before correlating, so they must be ruled out here.
    assert torch.isfinite(tt_output).all(), "non-finite values in the cross-attention output"
    assert_pcc(torch_output, tt_output, 0.999)
    if geometry != "empty":
        # The attended part alone: the residual would carry the PCC on its own.
        assert_pcc(torch_output - query, tt_output - query, 0.999)


@torch.no_grad()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_rebatch_plan_update(device, reset_seeds):
    """A plan refilled in place for another frame's cameras matches the reference on them."""
    bev_shape = BEV_SHAPES["tiny"]
    torch_model = build_reference_encoder(num_layers=1).layers[0].attentions[1]
    inputs = random_encoder_inputs(bev_shape, 1)
    query = random_bev(bev_shape, 1, torch.Generator().manual_seed(1)).permute(1, 0, 2)
    first = camera_geometry(img_metas(1), *bev_shape, 4, pc_range())
    second = camera_geometry(img_metas(2, yaw_step_deg=25.0)[1:], *bev_shape, 4, pc_range())

    tt_model = _tt_model(torch_model, device)
    plan = build_rebatch_plan(
        *first, tt_model.embed_dims, device, GRID_DTYPE, full_capacity(bev_shape[0] * bev_shape[1])
    )
    update_rebatch_plan(plan, *second)

    torch_output = torch_model(query, inputs["value"], *second, torch.tensor(SPATIAL_SHAPES))
    tt_output = tt_model(_to_device(query, device), _to_device(inputs["value"], device), tt_model.frame_inputs(plan))
    tt_output = ttnn.to_torch(tt_output).float().reshape(torch_output.shape)
    assert_pcc(torch_output - query, tt_output - query, 0.999)
