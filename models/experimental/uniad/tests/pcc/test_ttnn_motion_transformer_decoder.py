# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import sys
from models.experimental.uniad.reference import utils
from models.experimental.uniad.tt import ttnn_utils
from models.experimental.uniad.reference.motion_head import MotionHead
from models.experimental.uniad.reference.motion_transformer_decoder import (
    MotionDeformableAttention,
    MotionTransformerAttentionLayer,
    FFN,
)
import torch.nn as nn


from models.experimental.uniad.tt.ttnn_motion_transformer_decoder import (
    TtMotionDeformableAttention,
    TtMotionTransformerAttentionLayer,
    TtMotionTransformerDecoder,
)

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)
from models.experimental.uniad.tests.pcc.test_ttnn_interaction import (
    custom_preprocessor as custom_preprocessor_interations,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

import ttnn

sys.modules["mmdet3d"] = sys.modules["models"]
sys.modules["mmdet3d.core.bbox.structures.lidar_box3d"] = utils
from models.experimental.uniad.common import load_torch_model


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, MotionDeformableAttention):
        parameters["sampling_offsets"] = {}
        parameters["sampling_offsets"]["weight"] = preprocess_linear_weight(
            model.sampling_offsets.weight, dtype=ttnn.bfloat16
        )
        parameters["sampling_offsets"]["bias"] = preprocess_linear_bias(
            model.sampling_offsets.bias, dtype=ttnn.bfloat16
        )

        parameters["attention_weights"] = {}
        parameters["attention_weights"]["weight"] = preprocess_linear_weight(
            model.attention_weights.weight, dtype=ttnn.bfloat16
        )
        parameters["attention_weights"]["bias"] = preprocess_linear_bias(
            model.attention_weights.bias, dtype=ttnn.bfloat16
        )

        parameters["value_proj"] = {}
        parameters["value_proj"]["weight"] = preprocess_linear_weight(model.value_proj.weight, dtype=ttnn.bfloat16)
        parameters["value_proj"]["bias"] = preprocess_linear_bias(model.value_proj.bias, dtype=ttnn.bfloat16)

        parameters["output_proj"] = {}
        parameters["output_proj"][0] = {}
        parameters["output_proj"][0]["weight"] = preprocess_linear_weight(
            model.output_proj[0].weight, dtype=ttnn.bfloat16
        )
        parameters["output_proj"][0]["bias"] = preprocess_linear_bias(model.output_proj[0].bias, dtype=ttnn.bfloat16)

        parameters["output_proj"][1] = {}
        parameters["output_proj"][1]["weight"] = preprocess_layernorm_parameter(
            model.output_proj[1].weight, dtype=ttnn.bfloat16
        )
        parameters["output_proj"][1]["bias"] = preprocess_layernorm_parameter(
            model.output_proj[1].bias, dtype=ttnn.bfloat16
        )

    if isinstance(model, MotionTransformerAttentionLayer):
        parameters["attentions"] = {}
        for i, child in enumerate(model.attentions):
            parameters_deformable_attention = {}
            if isinstance(child, MotionDeformableAttention):
                parameters_deformable_attention["sampling_offsets"] = {}
                parameters_deformable_attention["sampling_offsets"]["weight"] = preprocess_linear_weight(
                    child.sampling_offsets.weight, dtype=ttnn.bfloat16
                )
                parameters_deformable_attention["sampling_offsets"]["bias"] = preprocess_linear_bias(
                    child.sampling_offsets.bias, dtype=ttnn.bfloat16
                )

                parameters_deformable_attention["attention_weights"] = {}
                parameters_deformable_attention["attention_weights"]["weight"] = preprocess_linear_weight(
                    child.attention_weights.weight, dtype=ttnn.bfloat16
                )
                parameters_deformable_attention["attention_weights"]["bias"] = preprocess_linear_bias(
                    child.attention_weights.bias, dtype=ttnn.bfloat16
                )

                parameters_deformable_attention["value_proj"] = {}
                parameters_deformable_attention["value_proj"]["weight"] = preprocess_linear_weight(
                    child.value_proj.weight, dtype=ttnn.bfloat16
                )
                parameters_deformable_attention["value_proj"]["bias"] = preprocess_linear_bias(
                    child.value_proj.bias, dtype=ttnn.bfloat16
                )

                parameters_deformable_attention["output_proj"] = {}
                parameters_deformable_attention["output_proj"][0] = {}
                parameters_deformable_attention["output_proj"][0]["weight"] = preprocess_linear_weight(
                    child.output_proj[0].weight, dtype=ttnn.bfloat16
                )
                parameters_deformable_attention["output_proj"][0]["bias"] = preprocess_linear_bias(
                    child.output_proj[0].bias, dtype=ttnn.bfloat16
                )

                parameters_deformable_attention["output_proj"][1] = {}
                parameters_deformable_attention["output_proj"][1]["weight"] = preprocess_layernorm_parameter(
                    child.output_proj[1].weight, dtype=ttnn.bfloat16
                )
                parameters_deformable_attention["output_proj"][1]["bias"] = preprocess_layernorm_parameter(
                    child.output_proj[1].bias, dtype=ttnn.bfloat16
                )
            parameters["attentions"][i] = parameters_deformable_attention

        parameters["ffns"] = {}

        for i, ffn in enumerate(model.ffns):
            if isinstance(ffn, FFN):
                parameters["ffns"][f"{i}"] = {
                    "linear1": {
                        "weight": preprocess_linear_weight(ffn.layers[0][0].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[0][0].bias, dtype=ttnn.bfloat16),
                    },
                    "linear2": {
                        "weight": preprocess_linear_weight(ffn.layers[1].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[1].bias, dtype=ttnn.bfloat16),
                    },
                }

        parameters["norms"] = {}

        parameters["norms"][0] = {}
        parameters["norms"][0]["weight"] = preprocess_layernorm_parameter(model.norms[0].weight, dtype=ttnn.bfloat16)

        parameters["norms"][0]["bias"] = preprocess_layernorm_parameter(model.norms[1].bias, dtype=ttnn.bfloat16)

        parameters["norms"][1] = {}
        parameters["norms"][1]["weight"] = preprocess_layernorm_parameter(model.norms[1].weight, dtype=ttnn.bfloat16)

        parameters["norms"][1]["bias"] = preprocess_layernorm_parameter(model.norms[1].bias, dtype=ttnn.bfloat16)

    return parameters


def custom_preprocessor_motion_decoder(model, name):
    parameters = {}
    parameters["intention_interaction_layers"] = custom_preprocessor_interations(
        model.intention_interaction_layers, None
    )

    parameters["track_agent_interaction_layers"] = {}
    for index, child in enumerate(model.track_agent_interaction_layers):
        parameters["track_agent_interaction_layers"][index] = custom_preprocessor_interations(child, None)

    parameters["map_interaction_layers"] = {}
    for index, child in enumerate(model.map_interaction_layers):
        parameters["map_interaction_layers"][index] = custom_preprocessor_interations(child, None)

    parameters["bev_interaction_layers"] = {}
    for index, child in enumerate(model.bev_interaction_layers):
        parameters["bev_interaction_layers"][index] = custom_preprocessor(child, None)

    parameters["static_dynamic_fuser"] = {}
    parameters["static_dynamic_fuser"][0] = {}

    parameters["static_dynamic_fuser"][0]["weight"] = preprocess_linear_weight(
        model.static_dynamic_fuser[0].weight, dtype=ttnn.bfloat16
    )
    parameters["static_dynamic_fuser"][0]["bias"] = preprocess_linear_bias(
        model.static_dynamic_fuser[0].bias, dtype=ttnn.bfloat16
    )

    parameters["static_dynamic_fuser"][2] = {}
    parameters["static_dynamic_fuser"][2]["weight"] = preprocess_linear_weight(
        model.static_dynamic_fuser[2].weight, dtype=ttnn.bfloat16
    )
    parameters["static_dynamic_fuser"][2]["bias"] = preprocess_linear_bias(
        model.static_dynamic_fuser[2].bias, dtype=ttnn.bfloat16
    )

    parameters["dynamic_embed_fuser"] = {}
    parameters["dynamic_embed_fuser"][0] = {}
    parameters["dynamic_embed_fuser"][0]["weight"] = preprocess_linear_weight(
        model.dynamic_embed_fuser[0].weight, dtype=ttnn.bfloat16
    )
    parameters["dynamic_embed_fuser"][0]["bias"] = preprocess_linear_bias(
        model.dynamic_embed_fuser[0].bias, dtype=ttnn.bfloat16
    )

    parameters["dynamic_embed_fuser"][2] = {}
    parameters["dynamic_embed_fuser"][2]["weight"] = preprocess_linear_weight(
        model.dynamic_embed_fuser[2].weight, dtype=ttnn.bfloat16
    )
    parameters["dynamic_embed_fuser"][2]["bias"] = preprocess_linear_bias(
        model.dynamic_embed_fuser[2].bias, dtype=ttnn.bfloat16
    )

    parameters["in_query_fuser"] = {}
    parameters["in_query_fuser"][0] = {}
    parameters["in_query_fuser"][0]["weight"] = preprocess_linear_weight(
        model.in_query_fuser[0].weight, dtype=ttnn.bfloat16
    )
    parameters["in_query_fuser"][0]["bias"] = preprocess_linear_bias(model.in_query_fuser[0].bias, dtype=ttnn.bfloat16)

    parameters["in_query_fuser"][2] = {}
    parameters["in_query_fuser"][2]["weight"] = preprocess_linear_weight(
        model.in_query_fuser[2].weight, dtype=ttnn.bfloat16
    )
    parameters["in_query_fuser"][2]["bias"] = preprocess_linear_bias(model.in_query_fuser[2].bias, dtype=ttnn.bfloat16)

    parameters["out_query_fuser"] = {}
    parameters["out_query_fuser"][0] = {}
    parameters["out_query_fuser"][0]["weight"] = preprocess_linear_weight(
        model.out_query_fuser[0].weight, dtype=ttnn.bfloat16
    )
    parameters["out_query_fuser"][0]["bias"] = preprocess_linear_bias(
        model.out_query_fuser[0].bias, dtype=ttnn.bfloat16
    )

    parameters["out_query_fuser"][2] = {}
    parameters["out_query_fuser"][2]["weight"] = preprocess_linear_weight(
        model.out_query_fuser[2].weight, dtype=ttnn.bfloat16
    )
    parameters["out_query_fuser"][2]["bias"] = preprocess_linear_bias(
        model.out_query_fuser[2].bias, dtype=ttnn.bfloat16
    )

    return parameters


def custom_preprocessor_traj_reg_branches(model, dtype=ttnn.bfloat16, device=None):
    parameters = {}
    if isinstance(model, nn.ModuleList):
        for i in range(3):
            parameters[i] = {}

            parameters[i][0] = {}
            parameters[i][0]["weight"] = ttnn.to_device(
                preprocess_linear_weight(model[i][0].weight, dtype=ttnn.bfloat16), device=device
            )
            parameters[i][0]["bias"] = ttnn.to_device(
                preprocess_linear_bias(model[i][0].bias, dtype=ttnn.bfloat16), device=device
            )

            parameters[i][1] = {}

            parameters[i][2] = {}
            parameters[i][2]["weight"] = ttnn.to_device(
                preprocess_linear_weight(model[i][2].weight, dtype=ttnn.bfloat16), device=device
            )
            parameters[i][2]["bias"] = ttnn.to_device(
                preprocess_linear_bias(model[i][2].bias, dtype=ttnn.bfloat16), device=device
            )

            parameters[i][3] = {}

            parameters[i][4] = {}
            parameters[i][4]["weight"] = ttnn.to_device(
                preprocess_linear_weight(model[i][4].weight, dtype=ttnn.bfloat16), device=device
            )
            parameters[i][4]["bias"] = ttnn.to_device(
                preprocess_linear_bias(model[i][4].bias, dtype=ttnn.bfloat16), device=device
            )
    return parameters


def custom_preprocessor_embeddding_layer(model, dtype=ttnn.bfloat16, device=None):
    parameters = {}

    parameters[0] = {}
    parameters[0]["weight"] = ttnn.to_device(
        preprocess_linear_weight(model[0].weight, dtype=ttnn.bfloat16), device=device
    )
    parameters[0]["bias"] = ttnn.to_device(preprocess_linear_bias(model[0].bias, dtype=ttnn.bfloat16), device=device)

    parameters[1] = {}

    parameters[2] = {}
    parameters[2]["weight"] = ttnn.to_device(
        preprocess_linear_weight(model[2].weight, dtype=ttnn.bfloat16), device=device
    )
    parameters[2]["bias"] = ttnn.to_device(preprocess_linear_bias(model[2].bias, dtype=ttnn.bfloat16), device=device)

    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_uniad_MotionDeformableAttention(device, reset_seeds, model_location_generator):
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"
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
        **{"num_query": 300, "predict_modes": 6},
    )
    reference_model = reference_model.motionformer.bev_interaction_layers[0].attentions[0]
    reference_model = load_torch_model(
        torch_model=reference_model,
        layer="motion_head.motionformer.bev_interaction_layers.0.attentions.0",
        model_location_generator=model_location_generator,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    query = torch.randn(1, 1, 6, 256)
    key = None
    value = torch.randn(2500, 1, 256)
    identity = None
    query_pos = torch.randn(1, 1, 6, 256)
    key_padding_mask = None
    spatial_shapes = torch.Tensor([[50, 50]])
    level_start_index = torch.Tensor([0])
    bbox_results = [
        [
            utils.LiDARInstance3DBoxes(torch.randn(1, 9), box_dim=9),
            torch.Tensor([0.0069]),
            torch.Tensor([1]).to(torch.int),
            torch.Tensor([0]).to(torch.int),
            torch.empty(0, dtype=torch.bool),
        ]
    ]
    reference_trajs = torch.randn(1, 1, 6, 12, 1, 2)
    flag = "decoder"

    torch_output = reference_model(
        query=query,
        key=key,
        value=value,
        identity=identity,
        query_pos=query_pos,
        key_padding_mask=key_padding_mask,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        bbox_results=bbox_results,
        reference_trajs=reference_trajs,
        flag=flag,
    )

    ttnn_model = TtMotionDeformableAttention(
        parameters=parameters,
        device=device,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=4,
        num_steps=12,
        sample_index=-1,
        im2col_step=64,
        bev_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size=[0.2, 0.2, 8],
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    )

    ttnn_query = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_key = None
    ttnn_value = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_identity = None
    ttnn_query_pos = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_key_padding_mask = None
    ttnn_spatial_shapes = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_level_start_index = ttnn.from_torch(
        level_start_index, device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT
    )
    ttnn_bbox_results = [
        [
            ttnn_utils.TtLiDARInstance3DBoxes(
                ttnn.from_torch(bbox_results[0][0].tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT),
                box_dim=9,
            ),
            ttnn.from_torch(bbox_results[0][1], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(bbox_results[0][2], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(bbox_results[0][3], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(bbox_results[0][4], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT),
        ]
    ]
    ttnn_reference_trajs = ttnn.from_torch(reference_trajs, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    flag = "decoder"

    ttnn_output = ttnn_model(
        ttnn_query,
        key=ttnn_key,
        value=ttnn_value,
        identity=ttnn_identity,
        query_pos=ttnn_query_pos,
        key_padding_mask=ttnn_key_padding_mask,
        spatial_shapes=ttnn_spatial_shapes,
        level_start_index=ttnn_level_start_index,
        bbox_results=ttnn_bbox_results,
        reference_trajs=ttnn_reference_trajs,
        flag="decoder",
    )

    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_uniad_MotionTransformerAttentionLayer(device, reset_seeds, model_location_generator):
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"
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
        **{"num_query": 300, "predict_modes": 6},
    )
    reference_model = reference_model.motionformer.bev_interaction_layers[0]

    reference_model = load_torch_model(
        torch_model=reference_model,
        layer="motion_head.motionformer.bev_interaction_layers.0",
        model_location_generator=model_location_generator,
    )

    ###-----------------Inputs creation---------------------
    query = torch.randn(1, 1, 6, 256)
    key = None
    value = torch.randn(2500, 1, 256)
    query_pos = torch.randn(1, 1, 6, 256)
    key_pos = None
    attn_masks = None
    query_key_padding_mask = None
    key_padding_mask = None

    spatial_shapes = torch.Tensor([[50, 50]])
    level_start_index = torch.Tensor([0])
    bbox_results = [
        [
            utils.LiDARInstance3DBoxes(torch.randn(1, 9), box_dim=9),
            torch.Tensor([0.0069]),
            torch.Tensor([1]).to(torch.int),
            torch.Tensor([0]).to(torch.int),
            torch.empty(0, dtype=torch.bool),
        ]
    ]
    reference_trajs = torch.randn(1, 1, 6, 12, 1, 2)
    traj_cls_branches = None  # It will contain layer but no used inside this sub_module so keeping it as none

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    torch_output = reference_model(
        query=query,
        key=key,
        value=value,
        query_pos=query_pos,
        key_pos=key_pos,
        attn_masks=attn_masks,
        query_key_padding_mask=query_key_padding_mask,
        key_padding_mask=key_padding_mask,
        bbox_results=bbox_results,
        reference_trajs=reference_trajs,
        traj_cls_branches=traj_cls_branches,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    ttnn_model = TtMotionTransformerAttentionLayer(
        parameters=parameters,
        device=device,
        attn_cfgs=[
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
        ffn_cfgs={
            "type": "FFN",
            "embed_dims": 256,
            "feedforward_channels": 1024,
            "num_fcs": 2,
            "ffn_drop": 0.0,
            "act_cfg": {"type": "ReLU", "inplace": True},
        },
        operation_order=("cross_attn", "norm", "ffn", "norm"),
        init_cfg=None,
        feedforward_channels=512,
    )

    ttnn_query = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_key = None
    ttnn_value = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_query_pos = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_key_pos = None
    ttnn_attn_masks = None
    ttnn_query_key_padding_mask = None
    ttnn_key_padding_mask = None
    ttnn_spatial_shapes = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_level_start_index = ttnn.from_torch(
        level_start_index, device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT
    )
    ttnn_bbox_results = [
        [
            ttnn_utils.TtLiDARInstance3DBoxes(
                ttnn.from_torch(bbox_results[0][0].tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT),
                box_dim=9,
            ),
            ttnn.from_torch(bbox_results[0][1], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(bbox_results[0][2], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(bbox_results[0][3], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(bbox_results[0][4], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT),
        ]
    ]
    ttnn_reference_trajs = ttnn.from_torch(reference_trajs, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_traj_cls_branches = None  # It will contain layer but no used inside this sub_module so keeping it as none

    ttnn_output = ttnn_model(
        query=ttnn_query,
        key=ttnn_key,
        value=ttnn_value,
        query_pos=ttnn_query_pos,
        key_pos=ttnn_key_pos,
        attn_masks=ttnn_attn_masks,
        query_key_padding_mask=ttnn_query_key_padding_mask,
        key_padding_mask=ttnn_key_padding_mask,
        bbox_results=ttnn_bbox_results,
        reference_trajs=ttnn_reference_trajs,
        traj_cls_branches=ttnn_traj_cls_branches,
        spatial_shapes=ttnn_spatial_shapes,
        level_start_index=ttnn_level_start_index,
    )

    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_uniad_MotionTransformerDecoder(device, reset_seeds, model_location_generator):
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"
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
        **{"num_query": 300, "predict_modes": 6},
    )
    reference_model = reference_model.motionformer
    reference_model = load_torch_model(
        torch_model=reference_model, layer="motion_head.motionformer", model_location_generator=model_location_generator
    )

    # Preparing inputs
    track_query = torch.randn(1, 1, 256)
    lane_query = torch.randn(1, 300, 256)
    track_query_pos = torch.randn(1, 1, 256)
    lane_query_pos = torch.randn(1, 300, 256)
    track_bbox_results = [
        [
            utils.LiDARInstance3DBoxes(torch.randn(1, 9), box_dim=9),
            torch.Tensor([0.0069]),
            torch.Tensor([1]).to(torch.int),
            torch.Tensor([0]).to(torch.int),
            torch.empty(0, dtype=torch.bool),
        ]
    ]
    bev_embed = torch.randn(2500, 1, 256)
    reference_trajs = torch.randn(1, 1, 6, 12, 2)
    traj_reg_branches = nn.ModuleList(
        [
            nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 60))
            for _ in range(3)
        ]
    )
    traj_reg_branches.eval()
    agent_level_embedding = torch.randn(1, 1, 6, 256)
    scene_level_ego_embedding = torch.randn(1, 1, 6, 256)
    scene_level_offset_embedding = torch.randn(1, 1, 6, 256)
    learnable_embed = torch.randn(1, 1, 6, 256)
    agent_level_embedding_layer = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256))
    agent_level_embedding_layer.eval()
    scene_level_ego_embedding_layer = nn.Sequential(
        nn.Linear(256, 512, bias=True), nn.ReLU(), nn.Linear(512, 256, bias=True)
    )
    scene_level_ego_embedding_layer.eval()
    scene_level_offset_embedding_layer = nn.Sequential(
        nn.Linear(256, 512, bias=True), nn.ReLU(), nn.Linear(512, 256, bias=True)
    )
    scene_level_offset_embedding_layer.eval()

    traj_cls_branches = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(256, 256, bias=True),
                nn.LayerNorm((256,), eps=1e-5, elementwise_affine=True),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256, bias=True),
                nn.LayerNorm((256,), eps=1e-5, elementwise_affine=True),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1, bias=True),
            )
            for _ in range(3)
        ]
    )
    traj_cls_branches.eval()
    spatial_shapes = torch.Tensor([[50, 50]])
    level_start_index = torch.Tensor([0])

    torch_output = reference_model(
        track_query=track_query,
        lane_query=lane_query,
        track_query_pos=track_query_pos,
        lane_query_pos=lane_query_pos,
        track_bbox_results=track_bbox_results,
        bev_embed=bev_embed,
        reference_trajs=reference_trajs,
        traj_reg_branches=traj_reg_branches,
        agent_level_embedding=agent_level_embedding,
        scene_level_ego_embedding=scene_level_ego_embedding,
        scene_level_offset_embedding=scene_level_offset_embedding,
        learnable_embed=learnable_embed,
        agent_level_embedding_layer=agent_level_embedding_layer,
        scene_level_ego_embedding_layer=scene_level_ego_embedding_layer,
        scene_level_offset_embedding_layer=scene_level_offset_embedding_layer,
        traj_cls_branches=traj_cls_branches,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=custom_preprocessor_motion_decoder,
        device=device,
    )

    ttnn_model = TtMotionTransformerDecoder(
        parameters=parameters,
        device=device,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        embed_dims=256,
        transformerlayers={
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
            "operation_order": ("cross_attn", "norm", "ffn", "norm"),
        },
        num_layers=3,
    )

    # Preparing ttnn input
    ttnn_track_query = ttnn.from_torch(track_query, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_lane_query = ttnn.from_torch(lane_query, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_track_query_pos = ttnn.from_torch(track_query_pos, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_lane_query_pos = ttnn.from_torch(lane_query_pos, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_track_bbox_results = [
        [
            ttnn_utils.TtLiDARInstance3DBoxes(
                ttnn.from_torch(
                    track_bbox_results[0][0].tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
                ),
                box_dim=9,
            ),
            ttnn.from_torch(track_bbox_results[0][1], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(track_bbox_results[0][2], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(track_bbox_results[0][3], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(track_bbox_results[0][4], device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT),
        ]
    ]
    ttnn_bev_embed = ttnn.from_torch(bev_embed, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_reference_trajs = ttnn.from_torch(reference_trajs, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_traj_reg_branches = custom_preprocessor_traj_reg_branches(
        traj_reg_branches, dtype=ttnn.bfloat16, device=device
    )
    ttnn_agent_level_embedding = ttnn.from_torch(
        agent_level_embedding, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn_scene_level_ego_embedding = ttnn.from_torch(
        scene_level_ego_embedding, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn_scene_level_offset_embedding = ttnn.from_torch(
        scene_level_offset_embedding, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    ttnn_learnable_embed = ttnn.from_torch(learnable_embed, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_agent_level_embedding_layer = custom_preprocessor_embeddding_layer(
        agent_level_embedding_layer, dtype=ttnn.bfloat16, device=device
    )
    ttnn_scene_level_ego_embedding_layer = custom_preprocessor_embeddding_layer(
        scene_level_ego_embedding_layer, dtype=ttnn.bfloat16, device=device
    )

    ttnn_scene_level_offset_embedding_layer = custom_preprocessor_embeddding_layer(
        scene_level_offset_embedding_layer, dtype=ttnn.bfloat16, device=device
    )

    ttnn_traj_cls_branches = (None,)  # Not used in this layer so keeping it as none

    ttnn_spatial_shapes = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_level_start_index = ttnn.from_torch(
        level_start_index, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    ttnn_output = ttnn_model(
        track_query=ttnn_track_query,
        lane_query=ttnn_lane_query,
        track_query_pos=ttnn_track_query_pos,
        lane_query_pos=ttnn_lane_query_pos,
        track_bbox_results=ttnn_track_bbox_results,
        bev_embed=ttnn_bev_embed,
        reference_trajs=ttnn_reference_trajs,
        traj_reg_branches=ttnn_traj_reg_branches,
        agent_level_embedding=ttnn_agent_level_embedding,
        scene_level_ego_embedding=ttnn_scene_level_ego_embedding,
        scene_level_offset_embedding=ttnn_scene_level_offset_embedding,
        learnable_embed=ttnn_learnable_embed,
        agent_level_embedding_layer=ttnn_agent_level_embedding_layer,
        scene_level_ego_embedding_layer=ttnn_scene_level_ego_embedding_layer,
        scene_level_offset_embedding_layer=ttnn_scene_level_offset_embedding_layer,
        traj_cls_branches=ttnn_traj_cls_branches,
        spatial_shapes=ttnn_spatial_shapes,
        level_start_index=ttnn_level_start_index,
    )

    assert_with_pcc(torch_output[0], ttnn.to_torch(ttnn_output[0]), pcc=0.99)
    assert_with_pcc(torch_output[1], ttnn.to_torch(ttnn_output[1]), pcc=0.99)
