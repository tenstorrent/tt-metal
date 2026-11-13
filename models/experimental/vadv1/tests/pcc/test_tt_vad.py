# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import copy
import math
import os
import time

import pytest
import torch
import ttnn
import numpy as np
from loguru import logger
from models.experimental.vadv1.reference import vad
from models.experimental.vadv1.tt import tt_vad
from models.experimental.vadv1.reference.base_box3d import LiDARInstance3DBoxes
from models.experimental.vadv1.tt.model_preprocessing import (
    create_vadv1_model_parameters_vad,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.vadv1.common import init_dummy_weights, synthetic_agents, synthetic_can_bus, synthetic_rig
from models.experimental.vadv1.num_layers import log_layer_counts, num_layers_from_env, reduce_layers


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 20 * 1024}], indirect=True)
def test_vadv1(
    device,
    reset_seeds,
):
    torch_model = vad.VAD(
        use_grid_mask=True,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=True,
        pts_backbone=None,
        img_neck=True,
        pts_neck=None,
        pts_bbox_head=True,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=True,
        fut_ts=6,
        fut_mode=6,
    )

    torch_model = init_dummy_weights(torch_model)
    num_layers = num_layers_from_env()
    if num_layers is not None:
        reduce_layers(torch_model, num_layers)
    log_layer_counts(torch_model, num_layers)

    agent_boxes, agent_labels, agent_attrs = synthetic_agents(11, fut_ts=6, step_s=0.5)
    input_dict = {
        "img_metas": [
            [
                [
                    {
                        "filename": [
                            "cam_front.jpg",
                            "cam_front_right.jpg",
                            "cam_front_left.jpg",
                            "cam_back.jpg",
                            "cam_back_left.jpg",
                            "cam_back_right.jpg",
                        ],
                        "ori_shape": [(360, 640, 3)] * 6,
                        "img_shape": [(384, 640, 3)] * 6,
                        "lidar2img": synthetic_rig(),
                        "pad_shape": [(384, 640, 3)] * 6,
                        "scale_factor": 1.0,
                        "flip": False,
                        "pcd_horizontal_flip": False,
                        "pcd_vertical_flip": False,
                        # 'box_mode_3d': Box3DMode.LIDAR,
                        "box_type_3d": LiDARInstance3DBoxes,
                        "img_norm_cfg": {
                            "mean": np.array([123.675, 116.28, 103.53], dtype=np.float32),
                            "std": np.array([58.395, 57.12, 57.375], dtype=np.float32),
                            "to_rgb": True,
                        },
                        "sample_idx": "sample-0",
                        "prev_idx": "",
                        "next_idx": "sample-1",
                        "pcd_scale_factor": 1.0,
                        "pts_filename": "lidar.bin",
                        "scene_token": "scene-0",
                        "can_bus": synthetic_can_bus(12.0, -7.0, math.radians(30.0), torch.Generator().manual_seed(0)),
                    }
                ]
            ]
        ],
        "gt_bboxes_3d": [[[LiDARInstance3DBoxes(agent_boxes, box_dim=9)]]],
        "gt_labels_3d": [[[agent_labels]]],
        "fut_valid_flag": [torch.tensor([True])],
        "ego_his_trajs": [[torch.tensor([[[[0.0, 2.5], [0.0, 2.5]]]])]],
        "ego_fut_trajs": [[torch.zeros((1, 1, 6, 2))]],
        "ego_fut_masks": [[torch.ones((1, 1, 6))]],
        "ego_fut_cmd": [[torch.tensor([[[[1.0, 0.0, 0.0]]]])]],
        "ego_lcf_feat": [[torch.zeros((1, 1, 1, 9), dtype=torch.float32)]],
        "gt_attr_labels": [[[agent_attrs]]],
        "map_gt_labels_3d": [[torch.zeros((7,), dtype=torch.long)]],
        "map_gt_bboxes_3d": [[None]],
    }
    tensor = torch.randn(1, 6, 3, 384, 640)
    img = []
    img.append(tensor)
    # Absolute CAN bus of the first frame, before the forward makes it relative.
    first_frame_metas = copy.deepcopy(input_dict["img_metas"])
    with torch.no_grad():
        model_outputs = torch_model(
            return_loss=False,
            img=img,
            img_metas=input_dict["img_metas"],
            gt_bboxes_3d=input_dict["gt_bboxes_3d"],
            gt_labels_3d=input_dict["gt_labels_3d"],
            fut_valid_flag=input_dict["fut_valid_flag"],
            ego_his_trajs=input_dict["ego_his_trajs"],
            ego_fut_trajs=input_dict["ego_fut_trajs"],
            ego_fut_cmd=input_dict["ego_fut_cmd"],
            ego_lcf_feat=input_dict["ego_lcf_feat"],
            gt_attr_labels=input_dict["gt_attr_labels"],
        )
    # ss

    parameter = create_vadv1_model_parameters_vad(
        torch_model,
        [
            False,
            img,
            input_dict["img_metas"],
            input_dict["gt_attr_labels"],
            input_dict["gt_bboxes_3d"],
            input_dict["gt_labels_3d"],
            input_dict["ego_his_trajs"],
            input_dict["ego_fut_trajs"],
            input_dict["ego_fut_cmd"],
            input_dict["ego_lcf_feat"],
        ],
        device,
    )

    torch_img = (
        tensor  # keep host-side copy so each warm iter can re-upload (device tensor gets deallocated by forward)
    )
    tensor = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    img = []
    img.append(tensor)
    rig_meta = input_dict["img_metas"][0][0][0]
    tt_model = tt_vad.TtVAD(
        device,
        parameter,
        img_shape=rig_meta["img_shape"][0][:2],
        read_raw_outputs=True,
        use_grid_mask=False,  # set to tru for training only
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=True,
        pts_backbone=None,
        img_neck=True,
        pts_neck=None,
        pts_bbox_head=True,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=True,
        fut_ts=6,
        fut_mode=6,
    )

    n_warm = int(os.environ.get("TT_VADV1_WARM_ITERS", "0"))
    timing = os.environ.get("TT_VADV1_TIMING") == "1"
    memory_report = os.environ.get("TT_VADV1_MEMORY_REPORT") == "1"

    # prepare_frame mutates img_metas[*]["can_bus"] in place and the forward
    # updates tt_model.prev_frame_info. Snapshot both so each
    # warm iteration sees the same first-frame input as the cold call.
    pristine_img_metas = copy.deepcopy(first_frame_metas)
    pristine_prev_frame_info = copy.deepcopy(tt_model.prev_frame_info)

    def _forward():
        return tt_model(
            return_loss=False,
            img=img,
            img_metas=input_dict["img_metas"],
            gt_bboxes_3d=input_dict["gt_bboxes_3d"],
            gt_labels_3d=input_dict["gt_labels_3d"],
            fut_valid_flag=input_dict["fut_valid_flag"],
            ego_his_trajs=input_dict["ego_his_trajs"],
            ego_fut_trajs=input_dict["ego_fut_trajs"],
            ego_fut_cmd=input_dict["ego_fut_cmd"],
            ego_lcf_feat=input_dict["ego_lcf_feat"],
            gt_attr_labels=input_dict["gt_attr_labels"],
        )

    if memory_report:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)

    walls_ms = []
    for i in range(n_warm + 1):
        input_dict["img_metas"] = copy.deepcopy(pristine_img_metas)
        tt_model.prev_frame_info = copy.deepcopy(pristine_prev_frame_info)
        # Re-upload img: forward deallocates the device tensor, so the same
        # ttnn handle can't be reused for the next call.
        img = [ttnn.from_torch(torch_img, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)]
        tt_model.prepare_frame(input_dict["img_metas"])
        t0 = time.perf_counter()
        ttnn_outputs = _forward()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        walls_ms.append(dt_ms)
        if timing:
            logger.info(f"[VADv1 call#{i + 1}] wall={dt_ms:.1f} ms")

    if timing and len(walls_ms) >= 2:
        logger.info(f"[VADv1 warm-wall anchor] call#{len(walls_ms)}={walls_ms[-1]:.1f} ms (cold={walls_ms[0]:.1f} ms)")

    if memory_report:
        report_path = os.environ.get("TT_VADV1_MEMORY_REPORT_PATH", "vadv1_graph_report.json")
        ttnn.graph.end_graph_capture_to_file(report_path)
        logger.info(f"[VADv1 memory report] written to {report_path}")

    keys_to_check = [
        "bev_embed",
        "all_cls_scores",
        "all_bbox_preds",
        "all_traj_preds",
        "all_traj_cls_scores",
        "map_all_cls_scores",
        "map_all_bbox_preds",
        "map_all_pts_preds",
        "ego_fut_preds",
    ]

    # Both frames, the second through each model's own previous BEV, reach at least 0.992
    # on every output.
    pcc_floors = {key: 0.99 for key in keys_to_check}
    for key in keys_to_check:
        a = torch.load(f"models/experimental/vadv1/reference/dumps/{key}.pt")
        b = tt_model._last_outs[key]
        _, msg = assert_with_pcc(a, b, pcc_floors[key])
        logger.info(f"{key}: {msg}")

    # A second frame of the same scene, after the ego moved and turned: both
    # models now take the previous BEV, rotated by the heading change, and a
    # relative CAN bus, which the first frame never exercises.
    second_metas = copy.deepcopy(first_frame_metas)
    second_meta = second_metas[0][0][0]
    second_meta["can_bus"][:2] += np.array([1.5, 0.4])
    second_meta["can_bus"][-2] += np.deg2rad(4.0)
    second_meta["can_bus"][-1] += 4.0
    # Recorded driving data compensates each camera for the ego's motion between its exposure and
    # the lidar sweep, so lidar2img changes from frame to frame; turn the rig by half a degree.
    yaw = np.deg2rad(0.5)
    turn = np.eye(4)
    turn[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    second_meta["lidar2img"] = [lidar2img @ turn for lidar2img in second_meta["lidar2img"]]
    tt_second_metas = copy.deepcopy(second_metas)
    second_img = torch.randn(1, 6, 3, 384, 640)
    frame_inputs = dict(
        gt_bboxes_3d=input_dict["gt_bboxes_3d"],
        gt_labels_3d=input_dict["gt_labels_3d"],
        fut_valid_flag=input_dict["fut_valid_flag"],
        ego_his_trajs=input_dict["ego_his_trajs"],
        ego_fut_trajs=input_dict["ego_fut_trajs"],
        ego_fut_cmd=input_dict["ego_fut_cmd"],
        ego_lcf_feat=input_dict["ego_lcf_feat"],
        gt_attr_labels=input_dict["gt_attr_labels"],
    )
    with torch.no_grad():
        torch_model(return_loss=False, img=[second_img], img_metas=second_metas, **frame_inputs)
    assert tt_model.prev_frame_info["prev_bev"] is not None
    tt_model.prepare_frame(tt_second_metas)
    tt_model(
        return_loss=False,
        img=[ttnn.from_torch(second_img, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)],
        img_metas=tt_second_metas,
        **frame_inputs,
    )
    for key in keys_to_check:
        a = torch.load(f"models/experimental/vadv1/reference/dumps/{key}.pt")
        b = tt_model._last_outs[key]
        _, msg = assert_with_pcc(a, b, pcc_floors[key])
        logger.info(f"second frame {key}: {msg}")
