# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.MapTR.reference.bevformer import (
    SpatialCrossAttention,
)
from models.experimental.MapTR.resources.download_chkpoint import ensure_checkpoint_downloaded, MAPTR_WEIGHTS_PATH
from models.experimental.MapTR.tt.ttnn_spatial_cross_attention import TtSpatialCrossAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)

SPATIAL_CROSS_ATTN_LAYER = "pts_bbox_head.transformer.encoder.layers.0.attentions.1."


def load_maptr_spatial_cross_attention_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    ensure_checkpoint_downloaded(weights_path)

    checkpoint = torch.load(weights_path, map_location="cpu")
    full_state_dict = checkpoint.get("state_dict", checkpoint)

    sca_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(SPATIAL_CROSS_ATTN_LAYER):
            relative_key = key[len(SPATIAL_CROSS_ATTN_LAYER) :]
            sca_weights[relative_key] = value

    logger.info(f"Loaded {len(sca_weights)} weight tensors for spatial cross attention")
    return sca_weights


def load_torch_model_maptr(torch_model: SpatialCrossAttention, weights_path: str = MAPTR_WEIGHTS_PATH):
    sca_weights = load_maptr_spatial_cross_attention_weights(weights_path)

    # Map the checkpoint keys to model keys
    model_state_dict = torch_model.state_dict()
    new_state_dict = {}

    for model_key in model_state_dict.keys():
        if model_key in sca_weights:
            new_state_dict[model_key] = sca_weights[model_key]
        else:
            logger.warning(f"Weight not found in checkpoint for: {model_key}")
            new_state_dict[model_key] = model_state_dict[model_key]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    return torch_model


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, SpatialCrossAttention):
        parameters["spatial_cross_attention"] = {
            "sampling_offsets": {
                "weight": preprocess_linear_weight(
                    model.deformable_attention.sampling_offsets.weight, dtype=ttnn.bfloat16
                ),
                "bias": preprocess_linear_bias(model.deformable_attention.sampling_offsets.bias, dtype=ttnn.bfloat16),
            },
            "attention_weights": {
                "weight": preprocess_linear_weight(
                    model.deformable_attention.attention_weights.weight, dtype=ttnn.bfloat16
                ),
                "bias": preprocess_linear_bias(model.deformable_attention.attention_weights.bias, dtype=ttnn.bfloat16),
            },
            "value_proj": {
                "weight": preprocess_linear_weight(model.deformable_attention.value_proj.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.deformable_attention.value_proj.bias, dtype=ttnn.bfloat16),
            },
            "output_proj": {
                "weight": preprocess_linear_weight(model.output_proj.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.output_proj.bias, dtype=ttnn.bfloat16),
            },
        }

    return parameters


def create_maptr_model_parameters_sca(model: SpatialCrossAttention, input_tensor, device=None):
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
def test_maptr_spatial_cross_attention(device, reset_seeds):
    point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    batch_first = True
    embed_dims = 256
    num_levels = 1
    num_points = 8

    torch_model = SpatialCrossAttention(
        embed_dims=embed_dims,
        pc_range=point_cloud_range,
        batch_first=batch_first,
        deformable_attention=dict(
            type="MSDeformableAttention3D", embed_dims=embed_dims, num_levels=num_levels, num_points=num_points
        ),
    )
    torch_model = load_torch_model_maptr(torch_model)

    num_query = 10000
    num_cams = 6
    num_points_per_level = 240
    num_z_anchors = 4

    query = torch.randn(1, num_query, embed_dims)
    key = torch.randn(num_cams, num_points_per_level, 1, embed_dims)
    value = torch.randn(num_cams, num_points_per_level, 1, embed_dims)
    reference_points = torch.randn(1, num_z_anchors, num_query, 3)
    spatial_shapes = torch.tensor([[12, 20]])
    reference_points_cam = torch.randn(num_cams, 1, num_query, num_z_anchors, 2)
    bev_mask = torch.randn(num_cams, 1, num_query, num_z_anchors)
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

    parameter = create_maptr_model_parameters_sca(
        torch_model,
        [query, key, value, reference_points, spatial_shapes, reference_points_cam, bev_mask, level_start_index],
        device,
    )

    tt_model = TtSpatialCrossAttention(
        device=device,
        params=parameter.spatial_cross_attention,
        embed_dims=embed_dims,
        pc_range=point_cloud_range,
        batch_first=batch_first,
        deformable_attention=dict(
            type="MSDeformableAttention3D", embed_dims=embed_dims, num_levels=num_levels, num_points=num_points
        ),
    )

    query_tt = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    key_tt = ttnn.from_torch(key, device=device, dtype=ttnn.bfloat16)
    value_tt = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)
    spatial_shapes_tt = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16)
    bev_mask_tt = ttnn.from_torch(bev_mask, device=device, dtype=ttnn.bfloat16)
    level_start_index_tt = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)

    tt_output = tt_model(
        query_tt,
        key_tt,
        value_tt,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes_tt,
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask_tt,
        level_start_index=level_start_index_tt,
    )

    ttnn_output = ttnn.to_torch(tt_output)
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.99)
    assert pcc_passed, pcc_message
