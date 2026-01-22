# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import torch.nn as nn
import numpy as np
from models.experimental.BEVFormerV2.reference.head import BEVFormerHead
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.BEVFormerV2.reference.encoder import BEVFormerEncoder
from models.experimental.BEVFormerV2.reference.perception_transformer import PerceptionTransformerV2
from models.experimental.BEVFormerV2.reference.decoder import (
    DetectionTransformerDecoder,
    CustomMSDeformableAttention,
    MultiheadAttention,
)
from models.experimental.BEVFormerV2.reference.temporal_self_attention import TemporalSelfAttention
from models.experimental.BEVFormerV2.reference.spatial_cross_attention import SpatialCrossAttention
from models.experimental.BEVFormerV2.reference.nms_free_coder import NMSFreeCoder
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)
from models.experimental.BEVFormerV2.common import download_bevformerv2_weights


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

            for n, norm in enumerate(getattr(layer, "norms", [])):
                if isinstance(norm, nn.LayerNorm):
                    layer_dict["norms"][f"norm{n}"] = {
                        "weight": preprocess_layernorm_parameter(norm.weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_layernorm_parameter(norm.bias, dtype=ttnn.bfloat16),
                    }

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
                    num_heads = deform_attn.num_heads
                    num_levels_checkpoint = deform_attn.num_levels
                    num_points = deform_attn.num_points

                    transformer_module = None
                    for parent_name, parent_module in model.named_modules():
                        if hasattr(parent_module, "transformer") and hasattr(
                            parent_module.transformer, "num_feature_levels"
                        ):
                            transformer_module = parent_module.transformer
                            break

                    num_feature_levels_used = (
                        transformer_module.num_feature_levels if transformer_module else num_levels_checkpoint
                    )

                    if num_levels_checkpoint > 1 and num_feature_levels_used < num_levels_checkpoint:
                        offsets_per_head = num_levels_checkpoint * num_points * 2
                        offsets_keep = num_points * 2
                        offsets_idx = []
                        for h in range(num_heads):
                            base = h * offsets_per_head
                            offsets_idx.extend(range(base, base + offsets_keep))
                        offsets_idx = torch.tensor(offsets_idx, dtype=torch.long)

                        attn_per_head = num_levels_checkpoint * num_points
                        attn_keep = num_points
                        attn_idx = []
                        for h in range(num_heads):
                            base = h * attn_per_head
                            attn_idx.extend(range(base, base + attn_keep))
                        attn_idx = torch.tensor(attn_idx, dtype=torch.long)

                        sampling_offsets_weight = deform_attn.sampling_offsets.weight.index_select(0, offsets_idx)
                        sampling_offsets_bias = deform_attn.sampling_offsets.bias.index_select(0, offsets_idx)
                        attention_weights_weight = deform_attn.attention_weights.weight.index_select(0, attn_idx)
                        attention_weights_bias = deform_attn.attention_weights.bias.index_select(0, attn_idx)
                    else:
                        sampling_offsets_weight = deform_attn.sampling_offsets.weight
                        sampling_offsets_bias = deform_attn.sampling_offsets.bias
                        attention_weights_weight = deform_attn.attention_weights.weight
                        attention_weights_bias = deform_attn.attention_weights.bias
                    layer_dict["attentions"][f"attn{j}"] = {
                        "sampling_offsets": {
                            "weight": preprocess_linear_weight(sampling_offsets_weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(sampling_offsets_bias, dtype=ttnn.bfloat16),
                        },
                        "attention_weights": {
                            "weight": preprocess_linear_weight(attention_weights_weight, dtype=ttnn.bfloat16),
                            "bias": preprocess_linear_bias(attention_weights_bias, dtype=ttnn.bfloat16),
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

    def extract_embeddings_to_ttnn(model, names, dtype):
        return {
            name: {"weight": ttnn.from_torch(getattr(model, name).weight, dtype=dtype, layout=ttnn.TILE_LAYOUT)}
            for name in names
        }

    if isinstance(model, BEVFormerHead):
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

        if isinstance(model.transformer, PerceptionTransformerV2):
            parameters["head"]["transformer"] = {}
            if isinstance(model.transformer.encoder, BEVFormerEncoder):
                parameters["head"]["transformer"]["encoder"] = extract_transformer_parameters(model.transformer.encoder)

            if isinstance(model.transformer.decoder, DetectionTransformerDecoder):
                parameters["head"]["transformer"]["decoder"] = extract_transformer_parameters(model.transformer.decoder)

            parameters["head"]["transformer"]["reference_points"] = {
                "weight": preprocess_linear_weight(model.transformer.reference_points.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.transformer.reference_points.bias, dtype=ttnn.bfloat16),
            }

            parameters["head"]["transformer"]["map_reference_points"] = {
                "weight": preprocess_linear_weight(model.transformer.reference_points.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.transformer.reference_points.bias, dtype=ttnn.bfloat16),
            }

            parameters["head"]["transformer"]["level_embeds"] = ttnn.from_torch(
                model.transformer.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            if hasattr(model.transformer, "cams_embeds"):
                parameters["head"]["transformer"]["cams_embeds"] = ttnn.from_torch(
                    model.transformer.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
            else:
                parameters["head"]["transformer"]["cams_embeds"] = ttnn.from_torch(
                    torch.zeros(6, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )

            def extract_reg_branches(reg_branches, dtype):
                branches_params = {}
                for i, branch in enumerate(reg_branches):
                    branch_params = {}
                    layer_idx = 0
                    for layer in branch:
                        if isinstance(layer, nn.Linear):
                            branch_params[str(layer_idx)] = {
                                "weight": preprocess_linear_weight(layer.weight, dtype=dtype),
                                "bias": preprocess_linear_bias(layer.bias, dtype=dtype),
                            }
                            layer_idx += 1
                    branches_params[str(i)] = branch_params
                return branches_params

            parameters["head"]["transformer"]["decoder"]["reg_branches"] = extract_reg_branches(
                model.reg_branches, dtype=ttnn.bfloat16
            )

        embedding_layers = ["bev_embedding", "query_embedding"]
        parameters["head"].update(extract_embeddings_to_ttnn(model, embedding_layers, dtype=ttnn.bfloat16))
        parameters["head"]["branches"] = {}

        parameters["head"]["branches"]["cls_branches"] = extract_sequential_branch(
            model.cls_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["branches"]["reg_branches"] = extract_sequential_branch(
            model.reg_branches, dtype=ttnn.bfloat16
        )
        parameters["head"]["cls_branches_torch"] = model.cls_branches
        parameters["head"]["reg_branches_torch"] = model.reg_branches

    return parameters


def create_bevformerv2_model_parameters_head(model: BEVFormerHead, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_bevformer_head_pcc(
    device,
    reset_seeds,
    model_location_generator,
):
    torch_model = BEVFormerHead(
        bev_h=100,
        bev_w=100,
        num_query=900,
        num_classes=10,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
    )

    weights_path = download_bevformerv2_weights()
    checkpoint = torch.load(weights_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    head_state = {}
    for key, value in state_dict.items():
        if key.startswith("pts_bbox_head"):
            new_key = key.replace("pts_bbox_head.", "")
            if new_key == "bev_embedding.weight":
                expected_size = torch_model.bev_h * torch_model.bev_w
                if value.shape[0] != expected_size:
                    head_state[new_key] = value[:expected_size]
                else:
                    head_state[new_key] = value
            else:
                head_state[new_key] = value

    torch_model.load_state_dict(head_state, strict=False)

    torch_model.eval()
    torch_model.transformer.encoder.layers = torch.nn.ModuleList(list(torch_model.transformer.encoder.layers)[:1])
    torch_model.transformer.encoder.num_layers = 1
    torch_model.transformer.decoder.layers = torch.nn.ModuleList(list(torch_model.transformer.decoder.layers)[:1])
    torch_model.transformer.decoder.num_layers = 1

    parameter = create_bevformerv2_model_parameters_head(torch_model, device=device)

    bbox_coder_obj = NMSFreeCoder(
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num=300,
        voxel_size=[0.512, 0.512, 8],
        num_classes=10,
    )

    tt_model = BEVFormerHead_TTNN(
        params=parameter,
        device=device,
        transformer=None,
        bbox_coder=bbox_coder_obj,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=100,
        bev_w=100,
        with_box_refine=True,
        as_two_stage=False,
        num_query=900,
        num_classes=10,
        embed_dims=256,
        num_reg_fcs=2,
        model_config={
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        },
    )

    mlvl_feats = []
    c0 = torch.randn(1, 6, 256, 16, 32)
    c1 = torch.randn(1, 6, 256, 8, 16)
    c2 = torch.randn(1, 6, 256, 4, 8)
    c3 = torch.randn(1, 6, 256, 2, 4)
    mlvl_feats.append(c0)
    mlvl_feats.append(c1)
    mlvl_feats.append(c2)
    mlvl_feats.append(c3)
    img_metas = [
        {
            "can_bus": np.array([0.0] * 18),
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

    model_outputs = torch_model(mlvl_feats, img_metas)
    mlvl_feats_ttnn = []
    c0_ttnn = ttnn.from_torch(c0, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    c1_ttnn = ttnn.from_torch(c1, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    c2_ttnn = ttnn.from_torch(c2, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    c3_ttnn = ttnn.from_torch(c3, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    mlvl_feats_ttnn.append(c0_ttnn)
    mlvl_feats_ttnn.append(c1_ttnn)
    mlvl_feats_ttnn.append(c2_ttnn)
    mlvl_feats_ttnn.append(c3_ttnn)
    ttnn_outputs = tt_model(mlvl_feats_ttnn, img_metas)

    bev_embed_ttnn = ttnn_outputs["bev_embed"]
    if isinstance(bev_embed_ttnn, ttnn.Tensor):
        bev_embed_ttnn = ttnn.to_torch(bev_embed_ttnn).float()
    else:
        bev_embed_ttnn = bev_embed_ttnn.float() if isinstance(bev_embed_ttnn, torch.Tensor) else bev_embed_ttnn

    assert_with_pcc(model_outputs["bev_embed"], bev_embed_ttnn, 0.98)
    assert_with_pcc(model_outputs["all_cls_scores"], ttnn_outputs["all_cls_scores"].float(), 0.99)
    assert_with_pcc(model_outputs["all_bbox_preds"], ttnn_outputs["all_bbox_preds"].float(), 0.99)
