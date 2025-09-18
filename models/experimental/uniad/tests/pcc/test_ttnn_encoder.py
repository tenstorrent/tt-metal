# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
import ttnn
from models.experimental.uniad.reference import encoder

from models.experimental.uniad.tt import ttnn_encoder
from models.experimental.uniad.tt.model_preprocessing_encoder import (
    create_uniad_model_parameters_encoder,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.uniad.common import load_torch_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_uniad_encoder(
    device,
    reset_seeds,
    model_location_generator,
):
    torch_model = encoder.BEVFormerEncoder(pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0])

    torch_model = load_torch_model(
        torch_model=torch_model,
        layer="pts_bbox_head.transformer.encoder",
        model_location_generator=model_location_generator,
    )

    parameter = create_uniad_model_parameters_encoder(torch_model, device=device)

    tt_model = ttnn_encoder.TtBEVFormerEncoder(
        params=parameter, device=device, pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    )

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
    sequence_length = 2500
    batch_size = 1
    embed_dim = 256

    bev_query = torch.rand(sequence_length, batch_size, embed_dim)
    key = torch.rand(6, 4820, 1, 256)
    value = torch.rand(6, 4820, 1, 256)
    args = ()
    bev_h = 50
    bev_w = 50
    bev_pos = torch.rand(sequence_length, batch_size, embed_dim)
    spatial_shapes = torch.tensor([[80, 45], [40, 23], [20, 12], [10, 6]])
    level_start_index = torch.tensor([0, 3600, 4520, 4760])
    shift = torch.tensor([[-16.9247, -2.5979]])

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

    key = ttnn.from_torch(key, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    value = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    bev_pos = ttnn.from_torch(bev_pos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    spatial_shapes = ttnn.from_torch(spatial_shapes, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    level_start_index = ttnn.from_torch(
        level_start_index, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    bev_query = ttnn.from_torch(bev_query, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    shift = ttnn.from_torch(shift, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

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

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(ttnn_output, torch_output, 0.99)
