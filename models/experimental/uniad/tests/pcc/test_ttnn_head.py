# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from models.experimental.uniad.reference.head import BEVFormerTrackHead
from models.experimental.uniad.tt.ttnn_head import TtBEVFormerTrackHead
import pytest
import numpy as np
from collections import OrderedDict

import ttnn

from models.experimental.uniad.tt.model_preprocessing_perception_transformer import (
    create_uniad_model_parameters_perception_transformer,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.uniad.common import load_torch_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_head_get_detections(device, reset_seeds, model_location_generator):
    reference_model = BEVFormerTrackHead(
        args=(),
        with_box_refine=True,
        as_two_stage=False,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=50,
        bev_w=50,
        past_steps=4,
        fut_steps=4,
        **{
            "num_query": 900,
            "num_classes": 10,
            "in_channels": 256,
            "sync_cls_avg_factor": True,
            "positional_encoding": {
                "type": "LearnedPositionalEncoding",
                "num_feats": 128,
                "row_num_embed": 200,
                "col_num_embed": 200,
            },
            "loss_cls": {"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 2.0},
            "loss_bbox": {"type": "L1Loss", "loss_weight": 0.25},
            "loss_iou": {"type": "GIoULoss", "loss_weight": 0.0},
            "train_cfg": None,
            "test_cfg": None,
        },
    )

    reference_model = load_torch_model(
        torch_model=reference_model, layer="pts_bbox_head", model_location_generator=model_location_generator
    )

    parameters = create_uniad_model_parameters_perception_transformer(reference_model, device)

    ttnn_model = TtBEVFormerTrackHead(
        parameters=parameters,
        device=device,
        args=(),
        with_box_refine=True,
        as_two_stage=False,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=50,
        bev_w=50,
        past_steps=4,
        fut_steps=4,
        **{
            "num_query": 900,
            "num_classes": 10,
            "in_channels": 256,
            "sync_cls_avg_factor": True,
            "positional_encoding": {
                "type": "LearnedPositionalEncoding",
                "num_feats": 128,
                "row_num_embed": 200,
                "col_num_embed": 200,
            },
            "loss_cls": {"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 2.0},
            "loss_bbox": {"type": "L1Loss", "loss_weight": 0.25},
            "loss_iou": {"type": "GIoULoss", "loss_weight": 0.0},
            "train_cfg": None,
            "test_cfg": None,
        },
    )

    bev_embed = torch.randn(2500, 1, 256)
    object_query_embeds = torch.randn(901, 512)
    ref_points = torch.randn(901, 3)
    img_metas = None

    reference_output_get_detections = reference_model.get_detections(
        bev_embed=bev_embed, object_query_embeds=object_query_embeds, ref_points=ref_points, img_metas=img_metas
    )

    ttnn_bev_embed = ttnn.from_torch(bev_embed, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_object_query_embed = ttnn.from_torch(
        object_query_embeds, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    ttnn_ref_points = ttnn.from_torch(ref_points, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_img_metas = img_metas

    ttnn_output = ttnn_model.get_detections(
        bev_embed=ttnn_bev_embed,
        object_query_embeds=ttnn_object_query_embed,
        ref_points=ttnn_ref_points,
        img_metas=ttnn_img_metas,
    )

    assert_with_pcc(
        reference_output_get_detections["all_cls_scores"], ttnn.to_torch(ttnn_output["all_cls_scores"]), pcc=0.99
    )
    assert_with_pcc(
        reference_output_get_detections["all_bbox_preds"], ttnn.to_torch(ttnn_output["all_bbox_preds"]), pcc=0.99
    )
    assert_with_pcc(
        reference_output_get_detections["all_past_traj_preds"],
        ttnn.to_torch(ttnn_output["all_past_traj_preds"]),
        pcc=0.99,
    )
    assert_with_pcc(
        reference_output_get_detections["last_ref_points"], ttnn.to_torch(ttnn_output["last_ref_points"]), pcc=0.98
    )
    assert_with_pcc(reference_output_get_detections["query_feats"], ttnn.to_torch(ttnn_output["query_feats"]), pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_head_get_bev_features(device, reset_seeds):
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"

    reference_model = BEVFormerTrackHead(
        args=(),
        with_box_refine=True,
        as_two_stage=False,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=50,
        bev_w=50,
        past_steps=4,
        fut_steps=4,
        **{
            "num_query": 900,
            "num_classes": 10,
            "in_channels": 256,
            "sync_cls_avg_factor": True,
            "positional_encoding": {
                "type": "LearnedPositionalEncoding",
                "num_feats": 128,
                "row_num_embed": 200,
                "col_num_embed": 200,
            },
            "loss_cls": {"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 2.0},
            "loss_bbox": {"type": "L1Loss", "loss_weight": 0.25},
            "loss_iou": {"type": "GIoULoss", "loss_weight": 0.0},
            "train_cfg": None,
            "test_cfg": None,
        },
    )

    weights = torch.load(weights_path, map_location=torch.device("cpu"))

    state_dict = weights.get("state_dict", weights)

    # Your model's expected shape
    new_bev_h = 50
    new_bev_w = 50
    new_bev_size = new_bev_h * new_bev_w

    # 1. Slice row_embed and col_embed from [200, 128] → [50, 128]
    for key in [
        "pts_bbox_head.positional_encoding.row_embed.weight",
        "pts_bbox_head.positional_encoding.col_embed.weight",
    ]:
        if key in state_dict:
            print(f"Slicing {key} from {state_dict[key].shape} to {(new_bev_h, state_dict[key].shape[1])}")
            state_dict[key] = state_dict[key][:new_bev_h, :]

    # 2. Slice bev_embedding from [40000, 256] → [2500, 256]
    for key in ["pts_bbox_head.bev_embedding.weight", "seg_head.bev_embedding.weight"]:
        if key in state_dict:
            print(f"Slicing {key} from {state_dict[key].shape} to {(new_bev_size, state_dict[key].shape[1])}")
            state_dict[key] = state_dict[key][:new_bev_size, :]

    if "criterion.code_weights" in state_dict:
        del state_dict["criterion.code_weights"]

    # Load the modified checkpoint
    prefix = "pts_bbox_head"
    filtered = OrderedDict(
        (
            (k[len(prefix) + 1 :], v)  # Remove the prefix from the key
            for k, v in weights["state_dict"].items()
            if k.startswith(prefix)
        )
    )
    reference_model.load_state_dict(filtered)

    reference_model.eval()

    parameters = create_uniad_model_parameters_perception_transformer(reference_model, device)

    ttnn_model = TtBEVFormerTrackHead(
        parameters=parameters,
        device=device,
        args=(),
        with_box_refine=True,
        as_two_stage=False,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=50,
        bev_w=50,
        past_steps=4,
        fut_steps=4,
        **{
            "num_query": 900,
            "num_classes": 10,
            "in_channels": 256,
            "sync_cls_avg_factor": True,
            "positional_encoding": {
                "type": "LearnedPositionalEncoding",
                "num_feats": 128,
                "row_num_embed": 200,
                "col_num_embed": 200,
            },
            "loss_cls": {"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 2.0},
            "loss_bbox": {"type": "L1Loss", "loss_weight": 0.25},
            "loss_iou": {"type": "GIoULoss", "loss_weight": 0.0},
            "train_cfg": None,
            "test_cfg": None,
        },
    )

    mlvl_feats = []
    mlvl_feats.append(torch.randn([1, 6, 256, 116, 200]))
    mlvl_feats.append(torch.randn([1, 6, 256, 58, 100]))
    mlvl_feats.append(torch.randn([1, 6, 256, 29, 50]))
    mlvl_feats.append(torch.randn([1, 6, 256, 15, 25]))

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

    prev_bev = None

    ttnn_mlvl_feats = [
        ttnn.from_torch((mlvl_feats[0]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        ttnn.from_torch((mlvl_feats[1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        ttnn.from_torch((mlvl_feats[2]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        ttnn.from_torch((mlvl_feats[3]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
    ]

    ttnn_prev_bev = prev_bev
    ttnn_img_metas = img_metas[:]

    reference_output_get_bev_features = reference_model.get_bev_features(
        mlvl_feats=mlvl_feats, img_metas=img_metas, prev_bev=prev_bev
    )

    ttnn_output = ttnn_model.get_bev_features(
        mlvl_feats=ttnn_mlvl_feats, img_metas=ttnn_img_metas, prev_bev=ttnn_prev_bev
    )

    assert_with_pcc(reference_output_get_bev_features[0], ttnn.to_torch(ttnn_output[0]), pcc=0.99)
    assert_with_pcc(reference_output_get_bev_features[1], ttnn.to_torch(ttnn_output[1]), pcc=0.99)
