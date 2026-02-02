# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.BEVFormerV2.reference.spatial_cross_attention import SpatialCrossAttention
from models.experimental.BEVFormerV2.tt.ttnn_spatial_cross_attention import TtSpatialCrossAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
)
from models.experimental.BEVFormerV2.tests.pcc.custom_preprocessors import (
    custom_preprocessor_spatial_cross_attention as custom_preprocessor,
)


def create_bevformerv2_model_parameters_sca(model: SpatialCrossAttention, input_tensor, device=None):
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
def test_spatial_cross_attention_pcc(
    device,
    reset_seeds,
    model_location_generator,
):
    torch.manual_seed(42)

    embed_dims = 256
    num_cams = 6
    num_levels = 4
    num_points = 8
    bs = 1
    num_query = 500
    D = 4

    spatial_shapes = torch.tensor([[20, 20], [10, 10], [5, 5], [3, 3]], dtype=torch.int32)
    num_value = int(torch.prod(spatial_shapes, dim=1).sum().item())

    pytorch_attention = SpatialCrossAttention(
        embed_dims=embed_dims,
        num_cams=num_cams,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        batch_first=False,
        deformable_attention=dict(
            type="MSDeformableAttention3D",
            embed_dims=embed_dims,
            num_levels=num_levels,
            num_points=num_points,
        ),
    )
    pytorch_attention.eval()

    query = torch.randn((bs, num_query, embed_dims), dtype=torch.float32)
    key = torch.randn((num_cams, num_value, bs, embed_dims), dtype=torch.float32)
    value = torch.randn((num_cams, num_value, bs, embed_dims), dtype=torch.float32)
    reference_points = torch.randn(1, 4, num_query, 3)
    reference_points_cam = torch.rand((num_cams, bs, num_query, D, 2), dtype=torch.float32)
    reference_points_cam = reference_points_cam.clamp(0.0, 1.0)
    bev_mask = torch.ones((num_cams, bs, num_query, D), dtype=torch.bool)
    bev_mask[:, :, num_query // 2 :, :] = False
    level_start_index = torch.tensor([0])

    with torch.no_grad():
        torch_output = pytorch_attention(
            query=query,
            key=key,
            value=value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            level_start_index=level_start_index,
        )

    parameter = create_bevformerv2_model_parameters_sca(
        pytorch_attention,
        [query, key, value, reference_points, spatial_shapes, reference_points_cam, bev_mask, level_start_index],
        device,
    )
    tt_model = TtSpatialCrossAttention(
        device=device,
        params=parameter.spatial_cross_attention,
        embed_dims=embed_dims,
        num_cams=num_cams,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        batch_first=False,
        deformable_attention=dict(
            type="MSDeformableAttention3D",
            embed_dims=embed_dims,
            num_levels=num_levels,
            num_points=num_points,
        ),
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
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.96)
    logger.info(pcc_message)
