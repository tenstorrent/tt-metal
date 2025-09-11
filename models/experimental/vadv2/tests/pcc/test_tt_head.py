# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import torch.nn as nn
import numpy as np
from models.experimental.vadv2.reference import head
from models.experimental.vadv2.tt.tt_head import TtVADHead
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.vadv2.reference.resnet import ResNet
from models.experimental.vadv2.reference.encoder import BEVFormerEncoder
from models.experimental.vadv2.reference.transformer import VADPerceptionTransformer
from models.experimental.vadv2.reference.head import VADHead, LaneNet, MLP
from models.experimental.vadv2.reference.decoder import (
    CustomTransformerDecoder,
    DetectionTransformerDecoder,
    MapDetectionTransformerDecoder,
    CustomMSDeformableAttention,
    MultiheadAttention,
)
from models.experimental.vadv2.reference.temporal_self_attention import TemporalSelfAttention
from models.experimental.vadv2.reference.spatial_cross_attention import SpatialCrossAttention
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)
from models.experimental.vadv2.common import load_torch_model


def custom_preprocessor(model, name):
    parameters = {}

    def extract_transformer_parameters(transformer_module):
        parameters = {"layers": {}}

        for i, layer in enumerate(transformer_module.layers):
            layer_dict = {
                "attentions": {},
                "ffn": {},
                "norms": {},
            }

            # ---- Norms ----
            for n, norm in enumerate(getattr(layer, "norms", [])):
                if isinstance(norm, nn.LayerNorm):
                    layer_dict["norms"][f"norm{n}"] = {
                        "weight": preprocess_layernorm_parameter(norm.weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_layernorm_parameter(norm.bias, dtype=ttnn.bfloat16),
                    }

            # ---- FFNs ----
            for k, ffn in enumerate(getattr(layer, "ffns", [])):
                layer_dict["ffn"][f"ffn{k}"] = {
                    "linear1": {
                        "weight": preprocess_linear_weight(ffn.layers[0][0].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[0][0].bias, dtype=ttnn.bfloat16),
                    },
                    "linear2": {
                        "weight": preprocess_linear_weight(ffn.layers[1].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[1].bias, dtype=ttnn.bfloat16),
                    },
                }

            # ---- Attentions ----
            for j, attn in enumerate(getattr(layer, "attentions", [])):
                if isinstance(attn, TemporalSelfAttention):
                    layer_dict["attentions"][f"attn{j}"] = {
                        "sampling_offsets": {
                            "weight": preprocess_linear_weight(attn.sampling_offsets.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.sampling_offsets.bias, dtype=ttnn.bfloat16),
                        },
                        "attention_weights": {
                            "weight": preprocess_linear_weight(attn.attention_weights.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.attention_weights.bias, dtype=ttnn.bfloat16),
                        },
                        "value_proj": {
                            "weight": preprocess_linear_weight(attn.value_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.value_proj.bias, dtype=ttnn.bfloat16),
                        },
                        "output_proj": {
                            "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
                        },
                    }

                elif isinstance(attn, SpatialCrossAttention):
                    deform_attn = attn.deformable_attention
                    layer_dict["attentions"][f"attn{j}"] = {
                        "sampling_offsets": {
                            "weight": preprocess_linear_weight(
                                deform_attn.sampling_offsets.weight, dtype=ttnn.bfloat16
                            ),
                            "bias": preprocess_linear_bias(deform_attn.sampling_offsets.bias, dtype=ttnn.bfloat16),
                        },
                        "attention_weights": {
                            "weight": preprocess_linear_weight(
                                deform_attn.attention_weights.weight, dtype=ttnn.bfloat16
                            ),
                            "bias": preprocess_linear_bias(deform_attn.attention_weights.bias, dtype=ttnn.bfloat16),
                        },
                        "value_proj": {
                            "weight": preprocess_linear_weight(deform_attn.value_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(deform_attn.value_proj.bias, dtype=ttnn.bfloat16),
                        },
                        "output_proj": {
                            "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
                        },
                    }

                elif isinstance(attn, MultiheadAttention):
                    layer_dict["attentions"][f"attn{j}"] = {
                        "in_proj": {
                            "weight": preprocess_linear_weight(attn.attn.in_proj_weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.attn.in_proj_bias, dtype=ttnn.bfloat16),
                        },
                        "out_proj": {
                            "weight": preprocess_linear_weight(attn.attn.out_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.attn.out_proj.bias, dtype=ttnn.bfloat16),
                        },
                    }

                elif isinstance(attn, CustomMSDeformableAttention):
                    layer_dict["attentions"][f"attn{j}"] = {
                        "sampling_offsets": {
                            "weight": preprocess_linear_weight(attn.sampling_offsets.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.sampling_offsets.bias, dtype=ttnn.bfloat16),
                        },
                        "attention_weights": {
                            "weight": preprocess_linear_weight(attn.attention_weights.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.attention_weights.bias, dtype=ttnn.bfloat16),
                        },
                        "value_proj": {
                            "weight": preprocess_linear_weight(attn.value_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.value_proj.bias, dtype=ttnn.bfloat16),
                        },
                        "output_proj": {
                            "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
                        },
                    }

            parameters["layers"][f"layer{i}"] = layer_dict
        return parameters

    def extract_sequential_branch(module_list, dtype):
        branch_params = {}

        for i, mod in enumerate(module_list):
            layer_params = {}
            layer_index = 0

            if isinstance(mod, nn.Sequential):
                layers = mod
            elif hasattr(mod, "mlp") and isinstance(mod.mlp, nn.Sequential):
                layers = mod.mlp
            else:
                layers = [mod]

            for layer in layers:
                if isinstance(layer, nn.Linear):
                    layer_params[str(layer_index)] = {
                        "weight": preprocess_linear_weight(layer.weight, dtype=dtype),
                        "bias": preprocess_linear_bias(layer.bias, dtype=dtype),
                    }
                    layer_index += 1
                elif isinstance(layer, nn.LayerNorm):
                    layer_params[f"{layer_index}_norm"] = {
                        "weight": preprocess_layernorm_parameter(layer.weight, dtype=dtype),
                        "bias": preprocess_layernorm_parameter(layer.bias, dtype=dtype),
                    }
                    layer_index += 1

            branch_params[str(i)] = layer_params

        return branch_params

    def extract_single_linears(model, layer_names, dtype):
        params = {}
        for name in layer_names:
            layer = getattr(model, name)
            params[name] = {
                "weight": preprocess_linear_weight(layer.weight, dtype=dtype),
                "bias": preprocess_linear_bias(layer.bias, dtype=dtype),
            }
        return params

    def extract_embeddings_to_ttnn(model, names, dtype):
        return {
            name: {"weight": ttnn.from_torch(getattr(model, name).weight, dtype=dtype, layout=ttnn.TILE_LAYOUT)}
            for name in names
        }

    def extract_lanenet_parameters(lane_encoder, dtype=ttnn.bfloat16):
        lanenet_params = {}

        for name, layer in lane_encoder.layer_seq.named_modules():
            if isinstance(layer, MLP):
                linear_layer = layer.mlp[0]
                norm_layer = layer.mlp[1]

                lanenet_params[name] = {
                    "linear": {
                        "weight": preprocess_linear_weight(linear_layer.weight, dtype=dtype),
                        "bias": preprocess_linear_bias(linear_layer.bias, dtype=dtype),
                    },
                    "norm": {
                        "weight": preprocess_layernorm_parameter(norm_layer.weight, dtype=dtype),
                        "bias": preprocess_layernorm_parameter(norm_layer.bias, dtype=dtype),
                    },
                }

        return lanenet_params

    if isinstance(model, (VADHead, CustomTransformerDecoder, VADPerceptionTransformer)):
        parameters = {}
        parameters["head"] = {}

        parameters["head"]["positional_encoding"] = {}
        pos_encoding = model.positional_encoding
        parameters["head"]["positional_encoding"]["row_embed"] = {
            "weight": ttnn.from_torch(pos_encoding.row_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        }
        parameters["head"]["positional_encoding"]["col_embed"] = {
            "weight": ttnn.from_torch(pos_encoding.col_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        }

        if isinstance(model.motion_decoder, CustomTransformerDecoder):
            parameters["head"]["motion_decoder"] = extract_transformer_parameters(model.motion_decoder)
        if isinstance(model.motion_map_decoder, CustomTransformerDecoder):
            parameters["head"]["motion_map_decoder"] = extract_transformer_parameters(model.motion_map_decoder)
        if isinstance(model.ego_map_decoder, CustomTransformerDecoder):
            parameters["head"]["ego_map_decoder"] = extract_transformer_parameters(model.ego_map_decoder)
        if isinstance(model.ego_agent_decoder, CustomTransformerDecoder):
            parameters["head"]["ego_agent_decoder"] = extract_transformer_parameters(model.ego_agent_decoder)

        if hasattr(model, "lane_encoder") and isinstance(model.lane_encoder, LaneNet):
            parameters["head"]["lane_encoder"] = extract_lanenet_parameters(model.lane_encoder)

        if isinstance(model.transformer, VADPerceptionTransformer):
            parameters["head"]["transformer"] = {}
            if isinstance(model.transformer.encoder, BEVFormerEncoder):
                parameters["head"]["transformer"]["encoder"] = extract_transformer_parameters(model.transformer.encoder)

            if isinstance(model.transformer.decoder, DetectionTransformerDecoder):
                parameters["head"]["transformer"]["decoder"] = extract_transformer_parameters(model.transformer.decoder)

            # Handle map_decoder if present
            if isinstance(model.transformer.map_decoder, MapDetectionTransformerDecoder):
                parameters["head"]["transformer"]["map_decoder"] = extract_transformer_parameters(
                    model.transformer.map_decoder
                )
            single_linear_layers = ["pos_mlp_sa", "pos_mlp", "ego_agent_pos_mlp", "ego_map_pos_mlp"]
            parameters["head"].update(extract_single_linears(model, single_linear_layers, ttnn.bfloat16))

            parameters["head"]["transformer"]["reference_points"] = {
                "weight": preprocess_linear_weight(model.transformer.reference_points.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.transformer.reference_points.bias, dtype=ttnn.bfloat16),
            }

            parameters["head"]["transformer"]["map_reference_points"] = {
                "weight": preprocess_linear_weight(model.transformer.map_reference_points.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.transformer.map_reference_points.bias, dtype=ttnn.bfloat16),
            }

            # CAN Bus MLP
            parameters["head"]["transformer"]["can_bus_mlp"] = {
                "0": {
                    "weight": preprocess_linear_weight(model.transformer.can_bus_mlp[0].weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(model.transformer.can_bus_mlp[0].bias, dtype=ttnn.bfloat16),
                },
                "1": {
                    "weight": preprocess_linear_weight(model.transformer.can_bus_mlp[2].weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(model.transformer.can_bus_mlp[2].bias, dtype=ttnn.bfloat16),
                },
                "norm": {
                    "weight": preprocess_layernorm_parameter(
                        model.transformer.can_bus_mlp.norm.weight, dtype=ttnn.bfloat16
                    ),
                    "bias": preprocess_layernorm_parameter(
                        model.transformer.can_bus_mlp.norm.bias, dtype=ttnn.bfloat16
                    ),
                },
            }

            parameters["head"]["transformer"]["level_embeds"] = ttnn.from_torch(
                model.transformer.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            parameters["head"]["transformer"]["cams_embeds"] = ttnn.from_torch(
                model.transformer.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
        embedding_layers = [
            "bev_embedding",
            "query_embedding",
            "map_instance_embedding",
            "map_pts_embedding",
            "motion_mode_query",
            "ego_query",
        ]
        parameters["head"].update(extract_embeddings_to_ttnn(model, embedding_layers, dtype=ttnn.bfloat16))
        parameters["head"]["branches"] = {}

        parameters["head"]["branches"]["cls_branches"] = extract_sequential_branch(
            model.cls_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["reg_branches"] = extract_sequential_branch(
            model.reg_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["traj_branches"] = extract_sequential_branch(
            model.traj_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["traj_cls_branches"] = extract_sequential_branch(
            model.traj_cls_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["map_cls_branches"] = extract_sequential_branch(
            model.map_cls_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["map_reg_branches"] = extract_sequential_branch(
            model.map_reg_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["ego_fut_decoder"] = extract_sequential_branch(
            model.ego_fut_decoder, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["agent_fus_mlp"] = extract_sequential_branch(
            model.agent_fus_mlp, dtype=ttnn.bfloat16
        )

    return parameters


def create_vadv2_model_parameters_head(model: ResNet, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_vadv2_head(
    device,
    reset_seeds,
    model_location_generator,
):
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

    torch_model = load_torch_model(
        torch_model=torch_model, layer="pts_bbox_head", model_location_generator=model_location_generator
    )

    parameter = create_vadv2_model_parameters_head(torch_model, device=device)

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

    assert_with_pcc(model_outputs["bev_embed"], ttnn.to_torch(ttnn_outputs["bev_embed"]).float(), 0.99)
    assert_with_pcc(model_outputs["all_cls_scores"], ttnn.to_torch(ttnn_outputs["all_cls_scores"]).float(), 0.99)
    assert_with_pcc(model_outputs["all_bbox_preds"], ttnn.to_torch(ttnn_outputs["all_bbox_preds"]).float(), 0.99)
    assert_with_pcc(model_outputs["all_traj_preds"], ttnn.to_torch(ttnn_outputs["all_traj_preds"]).float(), 0.98)
    assert_with_pcc(
        model_outputs["all_traj_cls_scores"], ttnn.to_torch(ttnn_outputs["all_traj_cls_scores"]).float(), 0.99
    )
    assert_with_pcc(
        model_outputs["map_all_cls_scores"], ttnn.to_torch(ttnn_outputs["map_all_cls_scores"]).float(), 0.99
    )
    assert_with_pcc(
        model_outputs["map_all_bbox_preds"], ttnn.to_torch(ttnn_outputs["map_all_bbox_preds"]).float(), 0.99
    )
    assert_with_pcc(model_outputs["map_all_pts_preds"], ttnn.to_torch(ttnn_outputs["map_all_pts_preds"]).float(), 0.99)
    assert_with_pcc(model_outputs["ego_fut_preds"], ttnn.to_torch(ttnn_outputs["ego_fut_preds"]).float(), 0.99)
