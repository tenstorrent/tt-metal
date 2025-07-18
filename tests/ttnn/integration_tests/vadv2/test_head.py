# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import numpy as np
from models.experimental.vadv2.reference import head
import ttnn
from models.experimental.vadv2.tt import tt_decoder
from models.experimental.vadv2.tt.model_preprocessing import (
    create_vadv2_model_parameters_head,
)
from models.experimental.vadv2.tt.tt_head import TtVADHead
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_vadv2_head(
    device,
    reset_seeds,
):
    weights_path = "models/experimental/vadv2/tt/vadv2_weights_1.pth"
    torch_model = head.VADHead(
        with_box_refine=True,
        as_two_stage=False,
        transformer=True,
        bbox_coder={
            "type": "CustomNMSFreeCoder",
            "post_center_range": [-20, -35, -10.0, 20, 35, 10.0],
            "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            "max_num": 100,
            "voxel_size": [0.15, 0.15, 4],
            "num_classes": 10,
        },
        num_cls_fcs=2,
        code_weights=None,
        bev_h=100,
        bev_w=100,
        fut_ts=6,
        fut_mode=6,
        loss_traj={"type": "L1Loss", "loss_weight": 0.2},
        loss_traj_cls={"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 0.2},
        map_bbox_coder={
            "type": "MapNMSFreeCoder",
            "post_center_range": [-20, -35, -20, -35, 20, 35, 20, 35],
            "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            "max_num": 50,
            "voxel_size": [0.15, 0.15, 4],
            "num_classes": 3,
        },
        map_num_query=900,
        map_num_classes=3,
        map_num_vec=100,
        map_num_pts_per_vec=20,
        map_num_pts_per_gt_vec=20,
        map_query_embed_type="instance_pts",
        map_transform_method="minmax",
        map_gt_shift_pts_pattern="v2",
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        loss_map_cls={"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 0.8},
        loss_map_bbox={"type": "L1Loss", "loss_weight": 0.0},
        loss_map_iou={"type": "GIoULoss", "loss_weight": 0.0},
        loss_map_pts={"type": "PtsL1Loss", "loss_weight": 0.4},
        loss_map_dir={"type": "PtsDirCosLoss", "loss_weight": 0.005},
        tot_epoch=12,
        use_traj_lr_warmup=False,
        motion_decoder=True,
        motion_map_decoder=True,
        use_pe=True,
        motion_det_score=None,
        map_thresh=0.5,
        dis_thresh=0.2,
        pe_normalization=True,
        ego_his_encoder=None,
        ego_fut_mode=3,
        loss_plan_reg={"type": "L1Loss", "loss_weight": 0.25},
        loss_plan_bound={"type": "PlanMapBoundLoss", "loss_weight": 0.1},
        loss_plan_col={"type": "PlanAgentDisLoss", "loss_weight": 0.1},
        loss_plan_dir={"type": "PlanMapThetaLoss", "loss_weight": 0.1},
        ego_agent_decoder=True,
        ego_map_decoder=True,
        query_thresh=0.0,
        query_use_fix_pad=False,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
    )

    torch_dict = torch.load(weights_path)

    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("pts_bbox_head"))}

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    # print(torch_model)
    # ss

    parameter = create_vadv2_model_parameters_head(torch_model, device=device)

    # assert False

    tt_model = TtVADHead(
        params=parameter,
        device=device,
        with_box_refine=True,
        as_two_stage=False,
        transformer=True,
        bbox_coder={
            "type": "CustomNMSFreeCoder",
            "post_center_range": [-20, -35, -10.0, 20, 35, 10.0],
            "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            "max_num": 100,
            "voxel_size": [0.15, 0.15, 4],
            "num_classes": 10,
        },
        num_cls_fcs=2,
        code_weights=None,
        bev_h=100,
        bev_w=100,
        fut_ts=6,
        fut_mode=6,
        map_bbox_coder={
            "type": "MapNMSFreeCoder",
            "post_center_range": [-20, -35, -20, -35, 20, 35, 20, 35],
            "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            "max_num": 50,
            "voxel_size": [0.15, 0.15, 4],
            "num_classes": 3,
        },
        map_num_query=900,
        map_num_classes=3,
        map_num_vec=100,
        map_num_pts_per_vec=20,
        map_num_pts_per_gt_vec=20,
        map_query_embed_type="instance_pts",
        map_transform_method="minmax",
        map_gt_shift_pts_pattern="v2",
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        tot_epoch=12,
        use_traj_lr_warmup=False,
        motion_decoder=True,
        motion_map_decoder=True,
        use_pe=True,
        motion_det_score=None,
        map_thresh=0.5,
        dis_thresh=0.2,
        pe_normalization=True,
        ego_his_encoder=None,
        ego_fut_mode=3,
        ego_agent_decoder=True,
        ego_map_decoder=True,
        query_thresh=0.0,
        query_use_fix_pad=False,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
    )
    print("Initialising is done")

    mlvl_feats = []
    c = torch.randn(1, 6, 256, 12, 20)
    mlvl_feats.append(c)
    img_metas = [
        {
            "ori_shape": [(360, 640, 3), (360, 640, 3), (360, 640, 3), (360, 640, 3), (360, 640, 3), (360, 640, 3)],
            "img_shape": [(384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3)],
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
            "pad_shape": [(384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3), (384, 640, 3)],
            "scale_factor": 1.0,
            "flip": False,
            "pcd_horizontal_flip": False,
            "pcd_vertical_flip": False,
            # 'box_mode_3d': <Box3DMode.LIDAR: 0>,
            # 'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>,
            "img_norm_cfg": {
                "mean": np.array([123.675, 116.28, 103.53], dtype=np.float32),
                "std": np.array([58.395, 57.12, 57.375], dtype=np.float32),
                "to_rgb": True,
            },
            "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
            "prev_idx": "",
            "next_idx": "3950bd41f74548429c0f7700ff3d8269",
            "pcd_scale_factor": 1.0,
            "pts_filename": "./data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin",
            "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
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
        }
    ]

    model_outputs = torch_model(mlvl_feats, img_metas)
    mlvl_feats = []
    c = ttnn.from_torch(c, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    mlvl_feats.append(c)
    ttnn_outputs = tt_model(mlvl_feats, img_metas)

    result1 = assert_with_pcc(model_outputs[0], ttnn.to_torch(ttnn_outputs[0]).float(), 0.99)
    result2 = assert_with_pcc(model_outputs[1], ttnn.to_torch(ttnn_outputs[1]).float(), 0.99)
    result3 = assert_with_pcc(model_outputs[2], ttnn.to_torch(ttnn_outputs[2]).float(), 0.99)
    result4 = assert_with_pcc(model_outputs[3], ttnn.to_torch(ttnn_outputs[3]).float(), 0.99)
    result5 = assert_with_pcc(model_outputs[4], ttnn.to_torch(ttnn_outputs[4]).float(), 0.99)
    # result6 = assert_with_pcc(model_outputs[5], ttnn.to_torch(ttnn_outputs[5]).float(), 0.99)
    # result7 = assert_with_pcc(model_outputs[6], ttnn.to_torch(ttnn_outputs[6]).float(), 0.99)
