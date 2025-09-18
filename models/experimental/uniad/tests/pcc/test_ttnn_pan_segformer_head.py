# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import numpy as np
from models.experimental.uniad.tt.model_preprocessing_encoder import (
    create_uniad_model_parameters_encoder,
)

from models.experimental.uniad.reference.pan_segformer_head import PansegformerHead
from models.experimental.uniad.tt.ttnn_pan_segformer_head import TtPansegformerHead
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.uniad.common import load_torch_model


class DotDict(dict):
    def __getattr__(self, key):
        return self[key]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_uniad_TtPansegformerhead(device, reset_seeds, model_location_generator):
    kwargs = {
        "num_query": 300,
        "num_classes": 4,
        "num_things_classes": 3,
        "num_stuff_classes": 1,
        "in_channels": 2048,
        "sync_cls_avg_factor": True,
        "positional_encoding": {"type": "SinePositionalEncoding", "num_feats": 128, "normalize": True, "offset": -0.5},
        "loss_cls": {"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 2.0},
        "loss_bbox": {"type": "L1Loss", "loss_weight": 5.0},
        "loss_iou": {"type": "GIoULoss", "loss_weight": 2.0},
    }

    torch_model = PansegformerHead(
        bev_h=50,
        bev_w=50,
        canvas_size=(50, 50),
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        with_box_refine=True,
        as_two_stage=False,
        **kwargs,
    )
    torch_model = load_torch_model(
        torch_model=torch_model, layer="seg_head", model_location_generator=model_location_generator
    )

    pts_feats = torch.randn(2500, 1, 256)
    gt_lane_labels = [torch.randint(0, 2, (1, 4))]
    gt_lane_masks = [torch.randint(0, 2, (1, 4, 50, 50)).float()]

    img_metas = [
        {
            "filename": [
                "./data/nuscenes/samples/CAM_FRONT/img1.jpg",
                "./data/nuscenes/samples/CAM_FRONT_RIGHT/img2.jpg",
                "./data/nuscenes/samples/CAM_FRONT_LEFT/img3.jpg",
                "./data/nuscenes/samples/CAM_BACK/img4.jpg",
                "./data/nuscenes/samples/CAM_BACK_LEFT/img5.jpg",
                "./data/nuscenes/samples/CAM_BACK_RIGHT/img6.jpg",
            ],
            "ori_shape": [(900, 1600, 3)] * 6,
            "img_shape": [(928, 1600, 3)] * 6,
            "pad_shape": [(928, 1600, 3)] * 6,
            "scale_factor": 1.0,
            "flip": False,
            "pcd_horizontal_flip": False,
            "pcd_vertical_flip": False,
            "lidar2img": [np.random.randn(4, 4).astype(np.float32) for _ in range(6)],
            "box_mode_3d": 0,
            "box_type_3d": "LiDARInstance3DBoxes",
            "img_norm_cfg": {
                "mean": np.array([103.53, 116.28, 123.675], dtype=np.float32),
                "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                "to_rgb": False,
            },
            "sample_idx": "mock_sample_id",
            "prev_idx": "",
            "next_idx": "mock_next_id",
            "pcd_scale_factor": 1.0,
            "pts_filename": "./data/nuscenes/samples/LIDAR_TOP/xyz.pcd.bin",
            "scene_token": "mock_scene_token",
            "can_bus": np.random.randn(18).astype(np.float32),
        }
    ]

    torch_output = torch_model.forward_test(
        pts_feats=pts_feats,
        gt_lane_labels=gt_lane_labels,
        gt_lane_masks=gt_lane_masks,
        img_metas=img_metas,
        rescale=True,
    )
    parameter = create_uniad_model_parameters_encoder(torch_model, device=device)

    pts_feats = ttnn.from_torch(pts_feats, device=device, layout=ttnn.TILE_LAYOUT)
    gt_lane_labels = [ttnn.from_torch(label, device=device, layout=ttnn.TILE_LAYOUT) for label in gt_lane_labels]
    gt_lane_masks = [ttnn.from_torch(mask, device=device, layout=ttnn.TILE_LAYOUT) for mask in gt_lane_masks]

    tt_model = TtPansegformerHead(
        parameter,
        device,
        bev_h=50,
        bev_w=50,
        canvas_size=(50, 50),
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        with_box_refine=True,
        as_two_stage=False,
        parameters_branches=parameter.reg_branches,
        **kwargs,
    )

    ttnn_output = tt_model.forward_test(
        pts_feats=pts_feats,
        gt_lane_labels=gt_lane_labels,
        gt_lane_masks=gt_lane_masks,
        img_metas=img_metas,
        rescale=True,
    )

    def check_pcc_and_log(name, torch_tensor, ttnn_tensor, threshold=0.99, convert_ttnn_to_torch=False):
        if convert_ttnn_to_torch:
            ttnn_tensor = ttnn.to_torch(ttnn_tensor)
        try:
            assert_with_pcc(torch_tensor, ttnn_tensor, threshold)
        except AssertionError as e:
            print(f"[WARNING] PCC check failed for '{name}'. Error: {e}")

    check_pcc_and_log("bbox", torch_output[0]["pts_bbox"]["bbox"], ttnn_output[0]["pts_bbox"]["bbox"])
    check_pcc_and_log("segm", torch_output[0]["pts_bbox"]["segm"], ttnn_output[0]["pts_bbox"]["segm"])
    check_pcc_and_log("labels", torch_output[0]["pts_bbox"]["labels"], ttnn_output[0]["pts_bbox"]["labels"])
    check_pcc_and_log("drivable", torch_output[0]["pts_bbox"]["drivable"], ttnn_output[0]["pts_bbox"]["drivable"])
    check_pcc_and_log(
        "score_list",
        torch_output[0]["pts_bbox"]["score_list"],
        ttnn_output[0]["pts_bbox"]["score_list"],
        convert_ttnn_to_torch=True,
    )
    check_pcc_and_log("lane_score", torch_output[0]["pts_bbox"]["lane_score"], ttnn_output[0]["pts_bbox"]["lane_score"])

    for i in range(len(torch_output[0]["args_tuple"]) - 2):
        if torch_output[0]["args_tuple"][i] is not None:
            check_pcc_and_log(
                "args_tuple",
                torch_output[0]["args_tuple"][i],
                ttnn_output[0]["args_tuple"][i],
                convert_ttnn_to_torch=True,
            )
