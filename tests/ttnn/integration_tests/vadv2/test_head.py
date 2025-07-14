# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from models.experimental.vadv2.reference import head

from models.experimental.vadv2.tt import tt_decoder
from models.experimental.vadv2.tt.model_preprocessing import (
    create_vadv2_model_parameters_head,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
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
    print(parameter)
