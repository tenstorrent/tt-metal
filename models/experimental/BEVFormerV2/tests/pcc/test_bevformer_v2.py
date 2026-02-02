# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import tracy
import pytest
import torch
import ttnn
import numpy as np
from models.experimental.BEVFormerV2.reference import bevformer_v2
from models.experimental.BEVFormerV2.tt.ttnn_bevformer_v2 import TtBevFormerV2
from models.experimental.BEVFormerV2.tt.model_preprocessing import (
    create_bevformerv2_model_parameters,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.BEVFormerV2.common import load_torch_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 20 * 1024}], indirect=True)
def test_bevformerv2(
    device,
    reset_seeds,
    model_location_generator,
):
    torch_model = bevformer_v2.BEVFormerV2(
        use_grid_mask=True,
        img_backbone=dict(depth=50, in_channels=3, out_indices=(1, 2, 3), style="caffe"),
        img_neck=dict(in_channels=[512, 1024, 2048], out_channels=256, num_outs=5),
        pts_bbox_head=dict(bev_h=100, bev_w=100, num_query=900, num_classes=10, in_channels=256),
        video_test_mode=True,
    )

    torch_model = load_torch_model(torch_model=torch_model, model_location_generator=model_location_generator)

    torch_model.pts_bbox_head.transformer.encoder.layers = torch.nn.ModuleList(
        list(torch_model.pts_bbox_head.transformer.encoder.layers)[:6]
    )
    torch_model.pts_bbox_head.transformer.encoder.num_layers = 6
    torch_model.pts_bbox_head.transformer.decoder.layers = torch.nn.ModuleList(
        list(torch_model.pts_bbox_head.transformer.decoder.layers)[:6]
    )
    torch_model.pts_bbox_head.transformer.decoder.num_layers = 6

    input_dict = {
        "img_metas": [
            [
                [
                    {
                        "filename": [
                            "./data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg",
                            "./data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg",
                            "./data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg",
                            "./data/nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg",
                            "./data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg",
                            "./data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg",
                        ],
                        "ori_shape": [(360, 640, 3)] * 6,
                        "img_shape": [(256, 704, 3)] * 6,
                        "lidar2img": [
                            np.array(
                                [
                                    [4.97195909e02, 3.36259809e02, 1.31050214e01, -1.41740456e02],
                                    [-7.28050437e00, 2.14719425e02, -4.90215017e02, -2.57883151e02],
                                    [-1.17025046e-02, 9.98471159e-01, 5.40221896e-02, -4.25203639e-01],
                                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                                ]
                            ),
                            np.array(
                                [
                                    [5.45978616e02, -2.47705944e02, -1.61356657e01, -1.84657143e02],
                                    [1.51784935e02, 1.28122911e02, -4.95917894e02, -2.77022512e02],
                                    [8.43406855e-01, 5.36312055e-01, 3.21598489e-02, -6.10371854e-01],
                                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                                ]
                            ),
                            np.array(
                                [
                                    [1.29479337e01, 6.01261709e02, 3.10492731e01, -1.20975154e02],
                                    [-1.55728079e02, 1.28176621e02, -4.94981202e02, -2.71769902e02],
                                    [-8.23415292e-01, 5.65940098e-01, 4.12196894e-02, -5.29677094e-01],
                                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                                ]
                            ),
                            np.array(
                                [
                                    [-3.21592898e02, -3.40289545e02, -1.05750653e01, -3.48318395e02],
                                    [-4.32931264e00, -1.78114385e02, -3.25958977e02, -2.83473696e02],
                                    [-8.33350064e-03, -9.99200442e-01, -3.91028008e-02, -1.01645350e00],
                                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                                ]
                            ),
                            np.array(
                                [
                                    [-4.74626444e02, 3.69304577e02, 2.13056637e01, -2.50136476e02],
                                    [-1.85050206e02, -4.10162348e01, -5.00990867e02, -2.24731382e02],
                                    [-9.47586752e-01, -3.19482867e-01, 3.16948959e-03, -4.32527296e-01],
                                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                                ]
                            ),
                            np.array(
                                [
                                    [1.14075693e02, -5.87710608e02, -2.38253717e01, -1.09040128e02],
                                    [1.77894417e02, -4.91302807e01, -5.00157067e02, -2.35298447e02],
                                    [9.24052925e-01, -3.82246554e-01, -3.70989150e-03, -4.64645142e-01],
                                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                                ]
                            ),
                        ],
                        "pad_shape": [(256, 704, 3)] * 6,
                        "scale_factor": 1.0,
                        "flip": False,
                        "pcd_horizontal_flip": False,
                        "pcd_vertical_flip": False,
                        "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
                        "prev_idx": "",
                        "next_idx": "3950bd41f74548429c0f7700ff3d8269",
                        "pcd_scale_factor": 1.0,
                        "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
                        "can_bus": np.array(
                            [
                                6.50486842e02,
                                1.81754303e03,
                                0.00000000e00,
                                1.84843146e-01,
                                1.84843146e-01,
                                1.84843146e-01,
                                1.84843146e-01,
                                8.47522666e-01,
                                1.34135536e00,
                                9.58588434e00,
                                -9.57939215e-03,
                                6.51179999e-03,
                                3.75314295e-01,
                                3.77446848e00,
                                0.00000000e00,
                                0.00000000e00,
                                3.51370076e00,
                                2.01320224e02,
                            ]
                        ),
                    }
                ]
            ]
        ],
    }
    tensor = torch.randn(1, 6, 3, 256, 704)
    img = []
    img.append(tensor)
    with torch.no_grad():
        model_outputs = torch_model(
            return_loss=False,
            img=img,
            img_metas=input_dict["img_metas"],
        )

    parameter = create_bevformerv2_model_parameters(
        torch_model,
        [
            False,
            img,
            input_dict["img_metas"],
        ],
        device,
    )

    tensor = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    img = []
    img.append(tensor)

    tt_model = TtBevFormerV2(
        device=device,
        params=parameter,
        use_grid_mask=False,
        img_backbone=dict(depth=50, in_channels=3, out_indices=(1, 2, 3), style="caffe"),
        img_neck=dict(in_channels=[512, 1024, 2048], out_channels=256, num_outs=5),
        pts_bbox_head=dict(
            bev_h=100,
            bev_w=100,
            num_query=900,
            num_classes=10,
            in_channels=256,
            encoder_num_layers=torch_model.pts_bbox_head.transformer.encoder.num_layers,
            decoder_num_layers=torch_model.pts_bbox_head.transformer.decoder.num_layers,
        ),
        video_test_mode=True,
    )
    tracy.signpost("start")
    ttnn_outputs = tt_model(
        return_loss=False,
        img=img,
        img_metas=input_dict["img_metas"],
    )
    tracy.signpost("stop")
    keys_to_check = [
        "bev_embed",
        "all_cls_scores",
        "all_bbox_preds",
    ]

    for key in keys_to_check:
        a = torch.load(f"models/experimental/BEVFormerV2/reference/dumps/{key}.pt")
        b = torch.load(f"models/experimental/BEVFormerV2/tt/dumps/{key}.pt")
        _, msg = assert_with_pcc(a, b, 0.99)
