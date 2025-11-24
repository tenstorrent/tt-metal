# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import numpy as np

from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.uniad.reference.utils import LiDARInstance3DBoxes
from models.experimental.uniad.reference.detr_track_3d_coder import DETRTrack3DCoder
from models.experimental.uniad.tt.ttnn_detr_track_3d_coder import TtDETRTrack3DCoder

img_metas = [
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
            "ori_shape": [(900, 1600, 3)] * 6,
            "img_shape": [(928, 1600, 3)] * 6,
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
            "pad_shape": [(928, 1600, 3)] * 6,
            "scale_factor": 1.0,
            "flip": False,
            "pcd_horizontal_flip": False,
            "pcd_vertical_flip": False,
            "box_type_3d": LiDARInstance3DBoxes,
            "img_norm_cfg": {
                "mean": np.array([103.53, 116.28, 123.675], dtype=np.float32),
                "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                "to_rgb": False,
            },
            "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
            "prev_idx": "",
            "next_idx": "3950bd41f74548429c0f7700ff3d8269",
            "pcd_scale_factor": 1.0,
            "pts_filename": "./data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin",
            "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
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
        }
    ]
]

# Case 1: 901 detections
bbox_dict_901 = {
    "cls_scores": torch.rand(901, 10),
    "bbox_preds": torch.rand(901, 10),
    "track_scores": torch.rand(901),
    "obj_idxes": torch.randint(0, 100, (901,), dtype=torch.int64),
}

# Case 2: 0 detections
bbox_dict_empty = {
    "cls_scores": torch.empty(0, 10),
    "bbox_preds": torch.empty(0, 10),
    "track_scores": torch.empty(0),
    "obj_idxes": torch.empty(0, dtype=torch.int64),
}

# Case 3: 1 detection
bbox_dict_1 = {
    "cls_scores": torch.rand(1, 10),
    "bbox_preds": torch.rand(1, 10),
    "track_scores": torch.rand(1),
    "obj_idxes": torch.randint(0, 100, (1,), dtype=torch.int64),
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
@pytest.mark.parametrize(
    "bbox_dict", [(bbox_dict_901), (bbox_dict_empty), (bbox_dict_1)], ids=["input1", "input2", "input3"]
)
@pytest.mark.parametrize("img_metas", [img_metas], ids=["inp1"])
def test_TtDETRTrack3DCoder(device, bbox_dict, img_metas, reset_seeds):
    with_mask = True
    reference_model = DETRTrack3DCoder(
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num=300,
        num_classes=10,
        score_threshold=0.0,
        with_nms=False,
        iou_thres=0.3,
    )

    torch_output = reference_model.decode(bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]
    # Assuming `bbox_dict` is already defined (torch version)
    ttnn_bbox_dict = {
        "cls_scores": ttnn.from_torch(bbox_dict["cls_scores"], device=device, layout=ttnn.TILE_LAYOUT),
        "bbox_preds": ttnn.from_torch(bbox_dict["bbox_preds"], device=device, layout=ttnn.TILE_LAYOUT),
        "track_scores": ttnn.from_torch(bbox_dict["track_scores"], device=device, layout=ttnn.TILE_LAYOUT),
        "obj_idxes": ttnn.from_torch(bbox_dict["obj_idxes"], device=device, layout=ttnn.TILE_LAYOUT),
    }

    ttnn_model = TtDETRTrack3DCoder(
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num=300,
        num_classes=10,
        score_threshold=0.0,
        with_nms=False,
        iou_thres=0.3,
        device=device,
    )

    ttnn_output = ttnn_model.decode(ttnn_bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]

    for key in torch_output.keys():
        torch_tensor = torch_output[key]
        ttnn_tensor = ttnn.to_torch(ttnn_output[key])

        logger.info(f"Checking PCC for key: {key}")
        logger.info(assert_with_pcc(torch_tensor, ttnn_tensor, pcc=0.99))
