# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import torch.nn as nn
import copy
import numpy as np
import ttnn
from models.experimental.vadv2.reference import transformer
from models.experimental.vadv2.tt.tt_transformer import TtVADPerceptionTransformer
from models.experimental.vadv2.tt import tt_encoder
from models.experimental.vadv2.tt.model_preprocessing import (
    create_vadv2_model_parameters_encoder,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vadv2_encoder(
    device,
    reset_seeds,
):
    weights_path = "models/experimental/vadv2/tt/vadv2_weights_1.pth"
    torch_model = transformer.VADPerceptionTransformer(
        rotate_prev_bev=True, use_shift=True, use_can_bus=True, decoder=True, map_decoder=True, embed_dims=256
    )

    torch_dict = torch.load(weights_path)

    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("pts_bbox_head.transformer"))}

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    # print(torch_model)
    # ss
    parameter = create_vadv2_model_parameters_encoder(torch_model, device=device)
    # print(parameter)
    # ss
    bev_h = 100
    bev_w = 100
    grid_length = (0.6, 0.3)
    bev_queries = torch.randn(10000, 256)
    object_query_embed = torch.randn(300, 512)
    map_query_embed = torch.randn(2000, 512)
    mlvl_feats = []
    a = torch.randn(1, 6, 256, 12, 20)
    mlvl_feats.append(a)
    bev_pos = torch.randn(1, 256, 100, 100)
    lidar_feat = None
    map_reg_branch = []
    for _ in range(2):
        map_reg_branch.append(nn.Linear(256, 256))
        map_reg_branch.append(nn.ReLU())
    map_reg_branch.append(nn.Linear(256, 2))
    map_reg_branch = nn.Sequential(*map_reg_branch)
    map_reg_branches = nn.ModuleList([copy.deepcopy(map_reg_branch) for i in range(3)])
    reg_branch = []
    for _ in range(2):
        reg_branch.append(nn.Linear(256, 256))
        reg_branch.append(nn.ReLU())
    reg_branch.append(nn.Linear(256, 10))
    reg_branch = nn.Sequential(*reg_branch)
    reg_branches = nn.ModuleList([copy.deepcopy(reg_branch) for i in range(3)])
    img_metas = [
        {
            "can_bus": np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.9686697,
                    -0.9686697,
                    -0.9686697,
                    -0.9686697,
                    -0.60694152,
                    -0.07634412,
                    9.87149385,
                    -0.02108691,
                    -0.01243972,
                    -0.023067,
                    8.5640597,
                    0.0,
                    0.0,
                    5.78155401,
                    0.0,
                ]
            ),
            "lidar2img": [
                np.array(
                    [
                        [2.48597954e02, 1.68129905e02, 6.55251068e00, -7.08702279e01],
                        [-3.64025219e00, 1.07359713e02, -2.45107509e02, -1.28941576e02],
                        [-1.17025046e-02, 9.98471159e-01, 5.40221896e-02, -4.25203639e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [2.72989308e02, -1.23852972e02, -8.06783283e00, -9.23285717e01],
                        [7.58924673e01, 6.40614553e01, -2.47958947e02, -1.38511256e02],
                        [8.43406855e-01, 5.36312055e-01, 3.21598489e-02, -6.10371854e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [6.47396684e00, 3.00630854e02, 1.55246365e01, -6.04875770e01],
                        [-7.78640394e01, 6.40883103e01, -2.47490601e02, -1.35884951e02],
                        [-8.23415292e-01, 5.65940098e-01, 4.12196894e-02, -5.29677094e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [-1.60796449e02, -1.70144772e02, -5.28753263e00, -1.74159198e02],
                        [-2.16465632e00, -8.90571925e01, -1.62979489e02, -1.41736848e02],
                        [-8.33350064e-03, -9.99200442e-01, -3.91028008e-02, -1.01645350e00],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [-2.37313222e02, 1.84652288e02, 1.06528318e01, -1.25068238e02],
                        [-9.25251029e01, -2.05081174e01, -2.50495434e02, -1.12365691e02],
                        [-9.47586752e-01, -3.19482867e-01, 3.16948959e-03, -4.32527296e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [5.70378465e01, -2.93855304e02, -1.19126859e01, -5.45200638e01],
                        [8.89472086e01, -2.45651403e01, -2.50078534e02, -1.17649223e02],
                        [9.24052925e-01, -3.82246554e-01, -3.70989150e-03, -4.64645142e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            ],
            "img_shape": [(192, 320, 3), (192, 320, 3), (192, 320, 3), (192, 320, 3), (192, 320, 3), (192, 320, 3)],
        }
    ]
    use_shift = True
    delta_x = np.array([each["can_bus"][0] for each in img_metas])
    delta_y = np.array([each["can_bus"][1] for each in img_metas])
    ego_angle = np.array([each["can_bus"][-2] / np.pi * 180 for each in img_metas])
    grid_length_y = grid_length[0]
    grid_length_x = grid_length[1]
    translation_length = np.sqrt(delta_x**2 + delta_y**2)
    translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
    bev_angle = ego_angle - translation_angle
    shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
    shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
    shift_y = shift_y * use_shift
    shift_x = shift_x * use_shift
    shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

    model_outputs = torch_model(
        mlvl_feats,
        bev_queries,
        object_query_embed,
        map_query_embed,
        bev_h,
        bev_w,
        grid_length=grid_length,
        bev_pos=bev_pos,
        reg_branches=None,
        map_reg_branches=None,
        img_metas=img_metas,
    )

    ttnn_model = TtVADPerceptionTransformer(
        params=parameter,
        device=device,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        decoder=True,
        map_decoder=True,
        embed_dims=256,
    )
    shift = ttnn.from_torch(shift, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    can_bus = bev_queries.new_tensor([each["can_bus"] for each in img_metas])  # [:, :]
    can_bus = ttnn.from_torch(can_bus, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    bev_queries = ttnn.from_torch(bev_queries, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    object_query_embed = ttnn.from_torch(
        object_query_embed, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    mlvl_feats = []
    # a = torch.randn(1, 6, 256, 12, 20)
    mlvl_feats.append(ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))
    map_query_embed = ttnn.from_torch(map_query_embed, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    bev_pos = ttnn.from_torch(bev_pos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_outputs = ttnn_model(
        mlvl_feats,
        bev_queries,
        object_query_embed,
        map_query_embed,
        bev_h,
        bev_w,
        grid_length=grid_length,
        bev_pos=bev_pos,
        reg_branches=None,
        map_reg_branches=None,
        img_metas=img_metas,
        shift=shift,
        can_bus=can_bus,
    )

    result1 = assert_with_pcc(model_outputs[0], ttnn.to_torch(ttnn_outputs[0]).float(), 0.99)
    result2 = assert_with_pcc(model_outputs[1], ttnn.to_torch(ttnn_outputs[1]).float(), 0.99)
    result3 = assert_with_pcc(model_outputs[2], ttnn.to_torch(ttnn_outputs[2]).float(), 0.99)
    result4 = assert_with_pcc(model_outputs[3], ttnn.to_torch(ttnn_outputs[3]).float(), 0.99)
    result5 = assert_with_pcc(model_outputs[4], ttnn.to_torch(ttnn_outputs[4]).float(), 0.99)
    result6 = assert_with_pcc(model_outputs[5], ttnn.to_torch(ttnn_outputs[5]).float(), 0.99)
    result7 = assert_with_pcc(model_outputs[6], ttnn.to_torch(ttnn_outputs[6]).float(), 0.99)
