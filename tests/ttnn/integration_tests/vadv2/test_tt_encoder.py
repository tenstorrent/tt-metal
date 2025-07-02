# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import torch.nn as nn
import copy
import numpy as np
import ttnn
from models.experimental.vadv2.reference import encoder

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
    torch_model = encoder.BEVFormerEncoder(pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0])

    torch_dict = torch.load(weights_path)

    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("pts_bbox_head.transformer.encoder"))}

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    parameter = create_vadv2_model_parameters_encoder(torch_model, device=device)
    print("cut")
    tt_model = tt_encoder.TtBEVFormerEncoder(
        params=parameter, device=device, pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    )

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
    sequence_length = 10000
    batch_size = 1
    embed_dim = 256

    bev_query = torch.rand(sequence_length, batch_size, embed_dim)
    key = torch.rand(6, 240, 1, 256)
    value = torch.rand(6, 240, 1, 256)
    args = ()
    bev_h = 100
    bev_w = 100
    bev_pos = torch.rand(sequence_length, batch_size, embed_dim)
    spatial_shapes = torch.tensor([[12, 20]], dtype=torch.long)
    level_start_index = torch.tensor([0])
    shift = torch.tensor([[-0.0, 0.0]])
    print("original implementation")
    torch_output = torch_model(
        bev_query=bev_query,
        key=key,
        value=value,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=bev_pos,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=None,
        prev_bev=None,
        shift=shift,
        img_metas=img_metas,
    )
    print(torch_output)
    print("end")

    # bev_query = ttnn.from_torch(bev_query,dtype = ttnn.bfloat16, layout = ttnn.ROW_MAJOR_LAYOUT,device = device)
    key = ttnn.from_torch(key, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    value = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    args = ()
    bev_h = 100
    bev_w = 100
    bev_pos = ttnn.from_torch(bev_pos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    spatial_shapes = ttnn.from_torch(spatial_shapes, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    level_start_index = ttnn.from_torch(
        level_start_index, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    shift = ttnn.from_torch(shift, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    print(
        "----------------------------------------------------------------------------------------------------------------------------------------------"
    )
    ttnn_output = tt_model(
        bev_query=bev_query,
        key=key,
        value=value,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=bev_pos,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=None,
        prev_bev=None,
        shift=shift,
        img_metas=img_metas,
    )

    print(ttnn_output)

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(ttnn_output, torch_output, 1)
