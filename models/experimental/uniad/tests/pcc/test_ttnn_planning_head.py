# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

import ttnn
import pytest

from models.experimental.uniad.reference.planning_head import PlanningHeadSingleMode
from models.experimental.uniad.tt.ttnn_planning_head import TtPlanningHeadSingleMode

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)

from loguru import logger
from models.experimental.uniad.common import load_torch_model


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, nn.Embedding):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    if isinstance(model, nn.Linear):
        parameters["weight"] = preprocess_linear_weight(model.weight, dtype=ttnn.bfloat16)
        if model.bias is not None:
            parameters["bias"] = preprocess_linear_bias(model.bias, dtype=ttnn.bfloat16)

    if isinstance(model, nn.LayerNorm):
        parameters["weight"] = preprocess_layernorm_parameter(model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_layernorm_parameter(model.bias, dtype=ttnn.bfloat16)

    if isinstance(model, nn.Conv2d):
        parameters["conv"] = {}
        parameters["conv"]["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    return parameters


def create_uniad_model_parameters(model, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters["model_args"] = model
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_TtPlanningHeadSingleMode(device, reset_seeds, model_location_generator):
    reference_model = PlanningHeadSingleMode(bev_h=50, bev_w=50, embed_dims=256, planning_steps=6, planning_eval=True)

    reference_model = load_torch_model(
        torch_model=reference_model, layer="planning_head", model_location_generator=model_location_generator
    )
    bev_embed = torch.rand(2500, 1, 256)
    occ_mask = torch.rand(1, 5, 1, 50, 50)
    bev_pos = torch.rand(1, 256, 50, 50)
    sdc_traj_query = torch.rand(3, 1, 6, 256)
    sdc_track_query = torch.rand(1, 256)
    command = [torch.tensor([0])]

    torch_output = reference_model(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command)

    parameters = create_uniad_model_parameters(reference_model, device)

    ttnn_bev_embed = ttnn.from_torch(bev_embed, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_occ_mask = ttnn.from_torch(occ_mask, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
    ttnn_bev_pos = ttnn.from_torch(bev_pos, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_sdc_traj_query = ttnn.from_torch(sdc_traj_query, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_sdc_track_query = ttnn.from_torch(sdc_track_query, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    command[0] = ttnn.from_torch(command[0], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    ttnn_model = TtPlanningHeadSingleMode(
        device,
        parameters=parameters,
        conv_pt=parameters["model_args"],
        bev_h=50,
        bev_w=50,
        embed_dims=256,
        planning_steps=6,
        planning_eval=True,
    )

    ttnn_output = ttnn_model(
        ttnn_bev_embed, ttnn_occ_mask, ttnn_bev_pos, ttnn_sdc_traj_query, ttnn_sdc_track_query, command
    )

    logger.info(assert_with_pcc(torch_output["sdc_traj"], ttnn.to_torch(ttnn_output["sdc_traj"]), pcc=0.99))
