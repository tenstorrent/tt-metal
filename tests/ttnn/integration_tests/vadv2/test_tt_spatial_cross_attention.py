# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.vadv2.reference import spatial_cross_attention
from models.experimental.vadv2.tt import tt_spatial_cross_attention
from models.experimental.vadv2.tt.model_preprocessing import (
    create_vadv2_model_parameters_sca,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vadv2_sca(
    device,
    reset_seeds,
):
    weights_path = "models/experimental/vadv2/tt/vadv2_weights_1.pth"
    point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    batch_first = True
    torch_model = spatial_cross_attention.SpatialCrossAttention(
        embed_dims=256, pc_range=point_cloud_range, batch_first=batch_first
    )
    torch_dict = torch.load(weights_path)

    state_dict = {
        k: v
        for k, v in torch_dict.items()
        if (k.startswith("pts_bbox_head.transformer.encoder.layers.0.attentions.1."))
    }
    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    query = torch.randn(1, 10000, 256)
    key = torch.randn(6, 240, 1, 256)
    value = torch.randn(6, 240, 1, 256)
    reference_points = torch.randn(1, 4, 10000, 3)
    spatial_shapes = torch.tensor([[12, 20]])
    reference_points_cam = torch.randn(6, 1, 10000, 4, 2)
    bev_mask = torch.randn(6, 1, 10000, 4)
    level_start_index = torch.tensor([0])

    from torchview import draw_graph

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
    model_graph = draw_graph(
        torch_model,
        input_data=(
            query,
            key,
            value,
            reference_points,
            spatial_shapes,
            reference_points_cam,
            bev_mask,
            level_start_index,
        ),
        expand_nested=True,  # Expand nested modules
        save_graph=True,  # Save graph to file
        graph_name="torch_model_graph",
    )

    model_graph.visualize()

    print("torch_output", torch_output.shape)

    # parameter = create_vadv2_model_parameters_sca(
    #     torch_model,
    #     [query, key, value, reference_points, spatial_shapes, reference_points_cam, bev_mask, level_start_index],
    #     device,
    # )
    # print(parameter)
    # tt_model = tt_spatial_cross_attention.TtSpatialCrossAttention(
    #     device=device,
    #     params=parameter.spatial_cross_attention,
    #     embed_dims=256,
    #     pc_range=point_cloud_range,
    #     batch_first=batch_first,
    # )

    # # query = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    # key = ttnn.from_torch(key, device=device, dtype=ttnn.bfloat16)
    # value = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)
    # # reference_points = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16)
    # # reference_points_cam = ttnn.from_torch(reference_points_cam, device=device, dtype=ttnn.bfloat16)
    # spatial_shapes = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16)
    # # bev_mask = ttnn.from_torch(bev_mask, device=device, dtype=ttnn.bfloat16)
    # level_start_index = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)

    # tt_output = tt_model(
    #     query,
    #     key,
    #     value,
    #     reference_points=reference_points,
    #     spatial_shapes=spatial_shapes,
    #     reference_points_cam=reference_points_cam,
    #     bev_mask=bev_mask,
    #     level_start_index=level_start_index,
    # )

    # ttnn_output = ttnn.to_torch(tt_output)
    # pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.99)
    # logger.info(pcc_message)
