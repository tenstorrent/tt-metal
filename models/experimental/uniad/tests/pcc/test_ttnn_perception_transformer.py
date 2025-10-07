# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import pytest
from torch import nn


import ttnn

from models.experimental.uniad.reference.perception_transformer import PerceptionTransformer
from models.experimental.uniad.tt.ttnn_perception_transformer import TtPerceptionTransformer
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.uniad.tt.model_preprocessing_perception_transformer import (
    create_uniad_model_parameters_perception_transformer,
    extract_sequential_branch,
)

from models.experimental.uniad.common import load_torch_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_perception_transformer_get_bev_features_function(device, reset_seeds, model_location_generator):
    reference_model = PerceptionTransformer(
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
    )
    reference_model = load_torch_model(
        torch_model=reference_model,
        layer="pts_bbox_head.transformer",
        model_location_generator=model_location_generator,
    )

    parameters = create_uniad_model_parameters_perception_transformer(reference_model, device)

    mlvl_feats = (
        torch.randn(1, 6, 256, 80, 45),
        torch.randn(1, 6, 256, 40, 23),
        torch.randn(1, 6, 256, 20, 12),
        torch.randn(1, 6, 256, 10, 6),
    )
    bev_queries = torch.randn(2500, 256)
    bev_h = 50
    bev_w = 50
    grid_length = (2.048, 2.048)
    bev_pos = torch.randn(1, 256, 50, 50)
    prev_bev = None

    img_metas = [
        {
            "can_bus": np.array(
                [
                    6.00120214e02,
                    1.64749078e03,
                    0.00000000e00,
                    -9.68669702e-01,
                    -4.04339926e-03,
                    -7.66659427e-03,
                    2.48201296e-01,
                    -6.06941519e-01,
                    -7.63441180e-02,
                    9.87149385e00,
                    -2.10869126e-02,
                    -1.24397185e-02,
                    -2.30670013e-02,
                    8.56405970e00,
                    0.00000000e00,
                    0.00000000e00,
                    5.78155401e00,
                    3.31258644e02,
                ]
            ),
            "lidar2img": [
                np.array(
                    [
                        [1.24298977e03, 8.40649523e02, 3.27625534e01, -3.54351139e02],
                        [-1.82012609e01, 5.36798564e02, -1.22553754e03, -6.44707879e02],
                        [-1.17025046e-02, 9.98471159e-01, 5.40221896e-02, -4.25203639e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [1.36494654e03, -6.19264860e02, -4.03391641e01, -4.61642859e02],
                        [3.79462336e02, 3.20307276e02, -1.23979473e03, -6.92556280e02],
                        [8.43406855e-01, 5.36312055e-01, 3.21598489e-02, -6.10371854e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [3.23698342e01, 1.50315427e03, 7.76231827e01, -3.02437885e02],
                        [-3.89320197e02, 3.20441551e02, -1.23745300e03, -6.79424755e02],
                        [-8.23415292e-01, 5.65940098e-01, 4.12196894e-02, -5.29677094e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [-8.03982245e02, -8.50723862e02, -2.64376631e01, -8.70795988e02],
                        [-1.08232816e01, -4.45285963e02, -8.14897443e02, -7.08684241e02],
                        [-8.33350064e-03, -9.99200442e-01, -3.91028008e-02, -1.01645350e00],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [-1.18656611e03, 9.23261441e02, 5.32641592e01, -6.25341190e02],
                        [-4.62625515e02, -1.02540587e02, -1.25247717e03, -5.61828455e02],
                        [-9.47586752e-01, -3.19482867e-01, 3.16948959e-03, -4.32527296e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [2.85189233e02, -1.46927652e03, -5.95634293e01, -2.72600319e02],
                        [4.44736043e02, -1.22825702e02, -1.25039267e03, -5.88246117e02],
                        [9.24052925e-01, -3.82246554e-01, -3.70989150e-03, -4.64645142e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            ],
            "img_shape": [(640, 360, 3), (640, 360, 3), (640, 360, 3), (640, 360, 3), (640, 360, 3), (640, 360, 3)],
        }
    ]

    ttnn_mlvl_feats = [
        ttnn.from_torch((mlvl_feats[0]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        ttnn.from_torch((mlvl_feats[1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        ttnn.from_torch((mlvl_feats[2]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        ttnn.from_torch((mlvl_feats[3]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
    ]

    ttnn_bev_h = bev_h
    ttnn_bev_w = bev_w
    ttnn_grid_length = grid_length
    ttnn_bev_queries = ttnn.from_torch(bev_queries, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_prev_bev = prev_bev
    ttnn_bev_pos = ttnn.from_torch(bev_pos, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_img_metas = img_metas[:]

    reference_output = reference_model.get_bev_features(
        mlvl_feats=mlvl_feats,
        bev_queries=bev_queries,
        bev_h=bev_h,
        bev_w=bev_w,
        grid_length=grid_length,
        bev_pos=bev_pos,
        prev_bev=prev_bev,
        img_metas=img_metas,
    )

    ttnn_model = TtPerceptionTransformer(
        parameters=parameters,
        device=device,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
    )

    ttnn_output = ttnn_model.get_bev_features(
        mlvl_feats=ttnn_mlvl_feats,
        bev_queries=ttnn_bev_queries,
        bev_h=ttnn_bev_h,
        bev_w=ttnn_bev_w,
        grid_length=ttnn_grid_length,
        bev_pos=ttnn_bev_pos,
        prev_bev=ttnn_prev_bev,
        img_metas=ttnn_img_metas,
    )

    assert_with_pcc(reference_output, ttnn.to_torch(ttnn_output), pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_perception_transformer_get_states_and_refs(device, reset_seeds, model_location_generator):
    reference_model = PerceptionTransformer(
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
    )
    reference_model = load_torch_model(
        torch_model=reference_model,
        layer="pts_bbox_head.transformer",
        model_location_generator=model_location_generator,
    )

    parameters = create_uniad_model_parameters_perception_transformer(reference_model, device)

    bev_embed = torch.randn(2500, 1, 256)
    object_query_embed = torch.randn(901, 512)
    bev_h = 50
    bev_w = 50
    reference_points = torch.randn(901, 3)
    reg_branches = nn.ModuleList(
        [
            nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
            for _ in range(6)
        ]
    )
    cls_branches = None

    ttnn_bev_embed = ttnn.from_torch(bev_embed, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_object_query_embed = ttnn.from_torch(
        object_query_embed, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn_bev_h = bev_h
    ttnn_bev_w = bev_w
    ttnn_reference_points = ttnn.from_torch(
        reference_points, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn_reg_branches = extract_sequential_branch(reg_branches, dtype=ttnn.bfloat16, device=device)
    ttnn_cls_branches = cls_branches

    reference_output = reference_model.get_states_and_refs(
        bev_embed=bev_embed,
        object_query_embed=object_query_embed,
        bev_h=bev_h,
        bev_w=bev_w,
        reference_points=reference_points,
        reg_branches=reg_branches,
        cls_branches=cls_branches,
        img_metas=None,
    )

    ttnn_model = TtPerceptionTransformer(
        parameters=parameters,
        device=device,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
    )

    ttnn_output = ttnn_model.get_states_and_refs(
        bev_embed=ttnn_bev_embed,
        object_query_embed=ttnn_object_query_embed,
        bev_h=ttnn_bev_h,
        bev_w=ttnn_bev_w,
        reference_points=ttnn_reference_points,
        reg_branches=ttnn_reg_branches,
        cls_branches=ttnn_cls_branches,
        img_metas=None,
    )

    assert_with_pcc(reference_output[0], ttnn.to_torch(ttnn_output[0]), pcc=0.99)  # 0.9991961317326385
    assert_with_pcc(reference_output[1], ttnn.to_torch(ttnn_output[1]), pcc=0.99)  # 0.999757470892231
    assert_with_pcc(reference_output[2], ttnn.to_torch(ttnn_output[2]), pcc=0.99)  # 0.9938016126956963
