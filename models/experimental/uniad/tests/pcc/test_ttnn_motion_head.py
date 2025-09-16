# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

from models.experimental.uniad.reference.motion_head import MotionHead
from models.experimental.uniad.reference import utils
from models.experimental.uniad.tests.pcc.test_ttnn_motion_transformer_decoder import custom_preprocessor_motion_decoder
from models.experimental.uniad.tt.ttnn_motion_head import TtMotionHead
from models.experimental.uniad.tt import ttnn_utils
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.uniad.common import load_torch_model


def custom_preprocessor_layer(child, name):
    parameters_tmp = {}
    parameters_tmp[0] = {}
    parameters_tmp[0]["weight"] = preprocess_linear_weight(child[0].weight, dtype=ttnn.bfloat16)
    parameters_tmp[0]["bias"] = preprocess_linear_bias(child[0].bias, dtype=ttnn.bfloat16)

    parameters_tmp[1] = {}

    parameters_tmp[2] = {}
    parameters_tmp[2]["weight"] = preprocess_linear_weight(child[2].weight, dtype=ttnn.bfloat16)
    parameters_tmp[2]["bias"] = preprocess_linear_bias(child[2].bias, dtype=ttnn.bfloat16)

    return parameters_tmp


def custom_preprocessor_motion_head(model, name):
    if isinstance(model, MotionHead):
        parameters = {}
        parameters["learnable_motion_query_embedding"] = {}
        parameters["learnable_motion_query_embedding"]["weight"] = ttnn.from_torch(
            model.learnable_motion_query_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        parameters["motionformer"] = custom_preprocessor_motion_decoder(model.motionformer, None)

        parameters["layer_track_query_fuser"] = {}
        parameters["layer_track_query_fuser"][0] = {}
        parameters["layer_track_query_fuser"][0]["weight"] = preprocess_linear_weight(
            model.layer_track_query_fuser[0].weight, dtype=ttnn.bfloat16
        )
        parameters["layer_track_query_fuser"][0]["bias"] = preprocess_linear_bias(
            model.layer_track_query_fuser[0].bias, dtype=ttnn.bfloat16
        )

        parameters["layer_track_query_fuser"][1] = {}
        parameters["layer_track_query_fuser"][1]["weight"] = preprocess_layernorm_parameter(
            model.layer_track_query_fuser[1].weight, dtype=ttnn.bfloat16
        )
        parameters["layer_track_query_fuser"][1]["bias"] = preprocess_layernorm_parameter(
            model.layer_track_query_fuser[1].bias, dtype=ttnn.bfloat16
        )

        parameters["layer_track_query_fuser"][2] = {}

        parameters["agent_level_embedding_layer"] = custom_preprocessor_layer(model.agent_level_embedding_layer, None)
        parameters["scene_level_ego_embedding_layer"] = custom_preprocessor_layer(
            model.scene_level_ego_embedding_layer, None
        )
        parameters["scene_level_offset_embedding_layer"] = custom_preprocessor_layer(
            model.scene_level_offset_embedding_layer, None
        )
        parameters["boxes_query_embedding_layer"] = custom_preprocessor_layer(model.boxes_query_embedding_layer, None)

        parameters["traj_cls_branches"] = {}
        for index, child in enumerate(model.traj_cls_branches):
            parameters_tmp = {}

            parameters_tmp[0] = {}
            parameters_tmp[0]["weight"] = preprocess_linear_weight(child[0].weight, dtype=ttnn.bfloat16)
            parameters_tmp[0]["bias"] = preprocess_linear_bias(child[0].bias, dtype=ttnn.bfloat16)

            parameters_tmp[1] = {}
            parameters_tmp[1]["weight"] = preprocess_layernorm_parameter(child[1].weight, dtype=ttnn.bfloat16)
            parameters_tmp[1]["bias"] = preprocess_layernorm_parameter(child[1].bias, dtype=ttnn.bfloat16)
            parameters_tmp[2] = {}

            parameters_tmp[3] = {}
            parameters_tmp[3]["weight"] = preprocess_linear_weight(child[3].weight, dtype=ttnn.bfloat16)
            parameters_tmp[3]["bias"] = preprocess_linear_bias(child[3].bias, dtype=ttnn.bfloat16)

            parameters_tmp[4] = {}
            parameters_tmp[4]["weight"] = preprocess_layernorm_parameter(child[4].weight, dtype=ttnn.bfloat16)
            parameters_tmp[4]["bias"] = preprocess_layernorm_parameter(child[4].bias, dtype=ttnn.bfloat16)
            parameters_tmp[5] = {}

            parameters_tmp[6] = {}
            parameters_tmp[6]["weight"] = preprocess_linear_weight(child[6].weight, dtype=ttnn.bfloat16)
            parameters_tmp[6]["bias"] = preprocess_linear_bias(child[6].bias, dtype=ttnn.bfloat16)

            parameters["traj_cls_branches"][index] = parameters_tmp

        parameters["traj_reg_branches"] = {}
        for index, child in enumerate(model.traj_reg_branches):
            parameters_temp = {}

            parameters_temp[0] = {}
            parameters_temp[0]["weight"] = preprocess_linear_weight(child[0].weight, dtype=ttnn.bfloat16)
            parameters_temp[0]["bias"] = preprocess_linear_bias(child[0].bias, dtype=ttnn.bfloat16)

            parameters_temp[1] = {}

            parameters_temp[2] = {}
            parameters_temp[2]["weight"] = preprocess_linear_weight(child[2].weight, dtype=ttnn.bfloat16)
            parameters_temp[2]["bias"] = preprocess_linear_bias(child[2].bias, dtype=ttnn.bfloat16)

            parameters_temp[3] = {}

            parameters_temp[4] = {}
            parameters_temp[4]["weight"] = preprocess_linear_weight(child[4].weight, dtype=ttnn.bfloat16)
            parameters_temp[4]["bias"] = preprocess_linear_bias(child[4].bias, dtype=ttnn.bfloat16)

            parameters["traj_reg_branches"][index] = parameters_temp

    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_uniad_MotionHead(device, reset_seeds, model_location_generator):
    reference_model = MotionHead(
        args=(),
        predict_steps=12,
        transformerlayers={
            "type": "MotionTransformerDecoder",
            "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "embed_dims": 256,
            "num_layers": 3,
            "transformerlayers": {
                "type": "MotionTransformerAttentionLayer",
                "batch_first": True,
                "attn_cfgs": [
                    {
                        "type": "MotionDeformableAttention",
                        "num_steps": 12,
                        "embed_dims": 256,
                        "num_levels": 1,
                        "num_heads": 8,
                        "num_points": 4,
                        "sample_index": -1,
                    }
                ],
                "feedforward_channels": 512,
                "ffn_dropout": 0.1,
                "operation_order": ("cross_attn", "norm", "ffn", "norm"),
            },
        },
        bbox_coder=None,
        num_cls_fcs=3,
        bev_h=50,
        bev_w=50,
        embed_dims=256,
        num_anchor=6,
        det_layer_num=6,
        group_id_list=[[0, 1, 2, 3, 4], [6, 7], [8], [5, 9]],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        use_nonlinear_optimizer=True,
        anchor_info_path="models/experimental/uniad/reference/motion_anchor_infos_mode6.pkl",
        loss_traj={
            "type": "TrajLoss",
            "use_variance": True,
            "cls_loss_weight": 0.5,
            "nll_loss_weight": 0.5,
            "loss_weight_minade": 0.0,
            "loss_weight_minfde": 0.25,
        },
        num_classes=10,
        vehicle_id_list=[0, 1, 2, 3, 4, 6, 7],
        num_query=300,
        predict_modes=6,
    )
    reference_model = load_torch_model(
        torch_model=reference_model, layer="motion_head", model_location_generator=model_location_generator
    )

    bev_embed = torch.randn(2500, 1, 256)

    outs_track = {}

    outs_track["bev_embed"] = torch.randn(2500, 1, 256)
    outs_track["bev_pos"] = torch.randn(1, 256, 50, 50)
    outs_track["track_query_embeddings"] = torch.randn(0, 256)
    outs_track["track_bbox_results"] = [
        [
            utils.LiDARInstance3DBoxes(torch.randn(0, 9), box_dim=9),
            torch.Tensor([]),
            torch.Tensor([]).to(torch.int),
            torch.Tensor([]).to(torch.int),
            torch.empty(0, dtype=torch.bool),
        ]
    ]
    outs_track["boxes_3d"] = utils.LiDARInstance3DBoxes(torch.randn(0, 9), box_dim=9)
    outs_track["scores_3d"] = torch.randn(0)
    outs_track["labels_3d"] = torch.randn(0)
    outs_track["track_scores"] = torch.randn(0)
    outs_track["track_ids"] = torch.randn(0)
    outs_track["sdc_boxes_3d"] = utils.LiDARInstance3DBoxes(torch.randn(1, 9), box_dim=9)
    outs_track["sdc_scores_3d"] = torch.randn(1)
    outs_track["sdc_track_scores"] = torch.randn(1)
    outs_track["sdc_track_bbox_results"] = [
        [
            utils.LiDARInstance3DBoxes(torch.randn(1, 9), box_dim=9),
            torch.Tensor([0.0069]),
            torch.Tensor([1]).to(torch.int),
            torch.Tensor([0]).to(torch.int),
            torch.Tensor([True]).to(dtype=torch.bool),
        ]
    ]
    outs_track["sdc_embedding"] = torch.randn(256)
    outs_track["boxes_3d_det"] = utils.LiDARInstance3DBoxes(torch.randn(300, 9), box_dim=9)
    outs_track["scores_3d_det"] = torch.randn(300)
    outs_track["labels_3d_det"] = torch.randn(300)

    outs_seg = {}
    outs_seg["pts_bbox"] = {}  # Set to None, since it is not used in motion_head
    outs_seg["ret_iou"] = {}  # Set to None, since it is not used in motion_head
    outs_seg["args_tuple"] = {}
    k = []
    k.append(torch.randn(1, 2500, 256))
    k.append(torch.full((1, 2500), False, dtype=torch.bool))
    k.append(torch.randn(1, 2500, 256))
    k.append(torch.randn(1, 300, 256))
    k.append(None)
    k.append(torch.randn(1, 300, 256))
    k.append([torch.zeros(50, 50)])
    outs_seg["args_tuple"] = k

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=custom_preprocessor_motion_head,
        device=device,
    )

    ttnn_model = TtMotionHead(
        parameters=parameters,
        device=device,
        args=(),
        predict_steps=12,
        transformerlayers={
            "type": "MotionTransformerDecoder",
            "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "embed_dims": 256,
            "num_layers": 3,
            "transformerlayers": {
                "type": "MotionTransformerAttentionLayer",
                "batch_first": True,
                "attn_cfgs": [
                    {
                        "type": "MotionDeformableAttention",
                        "num_steps": 12,
                        "embed_dims": 256,
                        "num_levels": 1,
                        "num_heads": 8,
                        "num_points": 4,
                        "sample_index": -1,
                    }
                ],
                "feedforward_channels": 512,
                "ffn_dropout": 0.1,
                "operation_order": ("cross_attn", "norm", "ffn", "norm"),
            },
        },
        bbox_coder=None,
        num_cls_fcs=3,
        bev_h=50,
        bev_w=50,
        embed_dims=256,
        num_anchor=6,
        det_layer_num=6,
        group_id_list=[[0, 1, 2, 3, 4], [6, 7], [8], [5, 9]],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        use_nonlinear_optimizer=True,
        anchor_info_path="models/experimental/uniad/reference/motion_anchor_infos_mode6.pkl",
        loss_traj={
            "type": "TrajLoss",
            "use_variance": True,
            "cls_loss_weight": 0.5,
            "nll_loss_weight": 0.5,
            "loss_weight_minade": 0.0,
            "loss_weight_minfde": 0.25,
        },
        num_classes=10,
        vehicle_id_list=[0, 1, 2, 3, 4, 6, 7],
        num_query=300,
        predict_modes=6,
    )

    ttnn_bev_embed = ttnn.from_torch(bev_embed, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    ttnn_outs_track = {}
    ttnn_outs_track["bev_embed"] = ttnn.from_torch(
        outs_track["bev_embed"], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn_outs_track["bev_pos"] = ttnn.from_torch(
        outs_track["bev_pos"], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn_outs_track["track_query_embeddings"] = ttnn.from_torch(
        outs_track["track_query_embeddings"], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    ttnn_outs_track["track_bbox_results"] = [
        [
            ttnn_utils.TtLiDARInstance3DBoxes(
                ttnn.from_torch(
                    outs_track["track_bbox_results"][0][0].tensor,
                    dtype=ttnn.bfloat16,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                ),
                box_dim=9,
            ),
            ttnn.from_torch(
                outs_track["track_bbox_results"][0][1], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            ttnn.from_torch(
                outs_track["track_bbox_results"][0][2], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT
            ),
            ttnn.from_torch(
                outs_track["track_bbox_results"][0][3], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT
            ),
            ttnn.from_torch(
                outs_track["track_bbox_results"][0][4], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT
            ),
        ]
    ]
    ttnn_outs_track["boxes_3d"] = ttnn_utils.TtLiDARInstance3DBoxes(
        ttnn.from_torch(outs_track["boxes_3d"].tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT),
        box_dim=9,
    )
    ttnn_outs_track["scores_3d"] = ttnn.from_torch(
        outs_track["scores_3d"], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn_outs_track["labels_3d"] = ttnn.from_torch(
        outs_track["labels_3d"], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn_outs_track["track_scores"] = ttnn.from_torch(
        outs_track["track_scores"], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn_outs_track["track_ids"] = ttnn.from_torch(
        outs_track["track_ids"], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn_outs_track["sdc_boxes_3d"] = ttnn_utils.TtLiDARInstance3DBoxes(
        ttnn.from_torch(outs_track["sdc_boxes_3d"].tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT),
        box_dim=9,
    )
    ttnn_outs_track["sdc_scores_3d"] = ttnn.from_torch(
        outs_track["sdc_scores_3d"], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn_outs_track["sdc_track_scores"] = ttnn.from_torch(
        outs_track["sdc_track_scores"], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn_outs_track["sdc_track_bbox_results"] = [
        [
            ttnn_utils.TtLiDARInstance3DBoxes(
                ttnn.from_torch(
                    outs_track["sdc_track_bbox_results"][0][0].tensor,
                    dtype=ttnn.bfloat16,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                ),
                box_dim=9,
            ),
            ttnn.from_torch(
                outs_track["sdc_track_bbox_results"][0][1], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            ttnn.from_torch(
                outs_track["sdc_track_bbox_results"][0][2], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT
            ),
            ttnn.from_torch(
                outs_track["sdc_track_bbox_results"][0][3], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT
            ),
            ttnn.from_torch(
                outs_track["sdc_track_bbox_results"][0][4], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT
            ),
        ]
    ]
    ttnn_outs_track["sdc_embedding"] = ttnn.from_torch(
        outs_track["sdc_embedding"], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn_outs_track["boxes_3d_det"] = ttnn_utils.TtLiDARInstance3DBoxes(
        ttnn.from_torch(outs_track["boxes_3d_det"].tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT),
        box_dim=9,
    )
    ttnn_outs_track["scores_3d_det"] = ttnn.from_torch(
        outs_track["scores_3d_det"], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn_outs_track["labels_3d_det"] = ttnn.from_torch(
        outs_track["labels_3d_det"], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    ttnn_outs_seg = {}
    ttnn_outs_seg["pts_bbox"] = {}  # Set to None, since it is not used in motion_head
    ttnn_outs_seg["ret_iou"] = {}  # Set to None, since it is not used in motion_head
    ttnn_outs_seg["args_tuple"] = [
        ttnn.from_torch(outs_seg["args_tuple"][0], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        ttnn.from_torch(outs_seg["args_tuple"][1], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        ttnn.from_torch(outs_seg["args_tuple"][2], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        ttnn.from_torch(outs_seg["args_tuple"][3], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        None,
        ttnn.from_torch(outs_seg["args_tuple"][5], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        [ttnn.from_torch(outs_seg["args_tuple"][6][0], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)],
    ]

    torch_output = reference_model.forward_test(bev_embed, outs_track, outs_seg)
    ttnn_output = ttnn_model.forward_test(ttnn_bev_embed, ttnn_outs_track, ttnn_outs_seg)

    assert_with_pcc(
        torch_output[0][0]["traj_0"], ttnn.to_torch(ttnn_output[0][0]["traj_0"]), pcc=0.999
    )  # 0.9994312010232113
    assert_with_pcc(
        torch_output[0][0]["traj_scores_0"], ttnn.to_torch(ttnn_output[0][0]["traj_scores_0"]), pcc=0.98
    )  #  0.9891932550464795
    assert_with_pcc(
        torch_output[0][0]["traj_1"], ttnn.to_torch(ttnn_output[0][0]["traj_1"]), pcc=0.99
    )  # 0.9960413270506279
    assert_with_pcc(torch_output[0][0]["traj_scores_1"], ttnn.to_torch(ttnn_output[0][0]["traj_scores_1"]), pcc=0.99)
    assert_with_pcc(
        torch_output[0][0]["traj"], ttnn.to_torch(ttnn_output[0][0]["traj"]), pcc=0.99
    )  # 0.9945363157090221
    assert_with_pcc(torch_output[0][0]["traj_scores"], ttnn.to_torch(ttnn_output[0][0]["traj_scores"]), pcc=0.99)

    assert_with_pcc(torch_output[1]["all_traj_scores"], ttnn.to_torch(ttnn_output[1]["all_traj_scores"]), pcc=0.99)
    assert_with_pcc(
        torch_output[1]["all_traj_preds"], ttnn.to_torch(ttnn_output[1]["all_traj_preds"]), pcc=0.99
    )  # 0.9965912712804653
    assert_with_pcc(
        torch_output[1]["valid_traj_masks"], ttnn.to_torch(ttnn_output[1]["valid_traj_masks"]), pcc=0.99
    )  # 1.0
    assert_with_pcc(
        torch_output[1]["sdc_traj_query"], ttnn.to_torch(ttnn_output[1]["sdc_traj_query"]), pcc=0.99
    )  # 0.9961525207269588
    assert_with_pcc(
        torch_output[1]["sdc_track_query"], ttnn.to_torch(ttnn_output[1]["sdc_track_query"]), pcc=0.999
    )  # 0.9999985195832592
    assert_with_pcc(
        torch_output[1]["sdc_track_query_pos"], ttnn.to_torch(ttnn_output[1]["sdc_track_query_pos"]), pcc=0.9999
    )  # 0.9999986120892685
