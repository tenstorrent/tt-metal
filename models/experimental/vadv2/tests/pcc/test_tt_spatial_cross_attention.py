# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.vadv2.reference import spatial_cross_attention
from models.experimental.vadv2.tt import tt_spatial_cross_attention
from models.experimental.vadv2.reference.spatial_cross_attention import SpatialCrossAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.experimental.vadv2.common import load_torch_model


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, SpatialCrossAttention):
        parameters["spatial_cross_attention"] = {}
        parameters["spatial_cross_attention"]["sampling_offsets"] = {}
        parameters["spatial_cross_attention"]["sampling_offsets"]["weight"] = preprocess_linear_weight(
            model.deformable_attention.sampling_offsets.weight, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["sampling_offsets"]["bias"] = preprocess_linear_bias(
            model.deformable_attention.sampling_offsets.bias, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["attention_weights"] = {}
        parameters["spatial_cross_attention"]["attention_weights"]["weight"] = preprocess_linear_weight(
            model.deformable_attention.attention_weights.weight, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["attention_weights"]["bias"] = preprocess_linear_bias(
            model.deformable_attention.attention_weights.bias, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["value_proj"] = {}
        parameters["spatial_cross_attention"]["value_proj"]["weight"] = preprocess_linear_weight(
            model.deformable_attention.value_proj.weight, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["value_proj"]["bias"] = preprocess_linear_bias(
            model.deformable_attention.value_proj.bias, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["output_proj"] = {}
        parameters["spatial_cross_attention"]["output_proj"]["weight"] = preprocess_linear_weight(
            model.output_proj.weight, dtype=ttnn.bfloat16
        )
        parameters["spatial_cross_attention"]["output_proj"]["bias"] = preprocess_linear_bias(
            model.output_proj.bias, dtype=ttnn.bfloat16
        )

    return parameters


def create_vadv2_model_parameters_sca(model: SpatialCrossAttention, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(
            input_tensor[0],
            key=input_tensor[1],
            value=input_tensor[2],
            reference_points=input_tensor[3],
            spatial_shapes=input_tensor[4],
            reference_points_cam=input_tensor[5],
            bev_mask=input_tensor[6],
            level_start_index=input_tensor[7],
        ),
        device=None,
    )
    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vadv2_sca(
    device,
    reset_seeds,
    model_location_generator,
):
    point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    batch_first = True
    torch_model = spatial_cross_attention.SpatialCrossAttention(
        embed_dims=256, pc_range=point_cloud_range, batch_first=batch_first
    )
    torch_model = load_torch_model(
        torch_model=torch_model,
        layer="pts_bbox_head.transformer.encoder.layers.0.attentions.1.",
        model_location_generator=model_location_generator,
    )

    query = torch.randn(1, 10000, 256)
    key = torch.randn(6, 240, 1, 256)
    value = torch.randn(6, 240, 1, 256)
    reference_points = torch.randn(1, 4, 10000, 3)
    spatial_shapes = torch.tensor([[12, 20]])
    reference_points_cam = torch.randn(6, 1, 10000, 4, 2)
    bev_mask = torch.randn(6, 1, 10000, 4)
    level_start_index = torch.tensor([0])

    torch_output = torch_model(
        query,
        key,
        value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask,
        level_start_index=level_start_index,
    )

    parameter = create_vadv2_model_parameters_sca(
        torch_model,
        [query, key, value, reference_points, spatial_shapes, reference_points_cam, bev_mask, level_start_index],
        device,
    )
    tt_model = tt_spatial_cross_attention.TtSpatialCrossAttention(
        device=device,
        params=parameter.spatial_cross_attention,
        embed_dims=256,
        pc_range=point_cloud_range,
        batch_first=batch_first,
    )

    query = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    key = ttnn.from_torch(key, device=device, dtype=ttnn.bfloat16)
    value = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)
    spatial_shapes = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16)
    bev_mask = ttnn.from_torch(bev_mask, device=device, dtype=ttnn.bfloat16)
    level_start_index = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)

    tt_output = tt_model(
        query,
        key,
        value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask,
        level_start_index=level_start_index,
    )

    ttnn_output = ttnn.to_torch(tt_output)
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(pcc_message)
