# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np
import torch.nn as nn
from models.experimental.BEVFormerV2.reference.perception_transformer import PerceptionTransformerV2
from models.experimental.BEVFormerV2.tt.ttnn_perception_transformer import TtPerceptionTransformer
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.BEVFormerV2.reference.encoder import BEVFormerEncoder
from models.experimental.BEVFormerV2.reference.decoder import (
    DetectionTransformerDecoder,
    CustomMSDeformableAttention,
    MultiheadAttention,
)
from models.experimental.BEVFormerV2.reference.temporal_self_attention import TemporalSelfAttention
from models.experimental.BEVFormerV2.reference.spatial_cross_attention import SpatialCrossAttention
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
                    num_levels = deform_attn.num_levels
                    num_points = deform_attn.num_points

                    if num_levels > 1:
                        offsets_per_head = num_levels * num_points * 2
                        offsets_keep = num_points * 2
                        offsets_idx = []
                        for h in range(num_heads):
                            base = h * offsets_per_head
                            offsets_idx.extend(range(base, base + offsets_keep))
                        offsets_idx = torch.tensor(offsets_idx, dtype=torch.long)

                        attn_per_head = num_levels * num_points
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

    if isinstance(model, PerceptionTransformerV2):
        parameters = {}

        if isinstance(model.encoder, BEVFormerEncoder):
            parameters["encoder"] = extract_transformer_parameters(model.encoder)

        if isinstance(model.decoder, DetectionTransformerDecoder):
            parameters["decoder"] = extract_transformer_parameters(model.decoder)

        parameters["reference_points"] = {
            "weight": preprocess_linear_weight(model.reference_points.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.reference_points.bias, dtype=ttnn.bfloat16),
        }

        parameters["map_reference_points"] = {
            "weight": preprocess_linear_weight(model.reference_points.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.reference_points.bias, dtype=ttnn.bfloat16),
        }

        parameters["level_embeds"] = ttnn.from_torch(model.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if hasattr(model, "cams_embeds"):
            parameters["cams_embeds"] = ttnn.from_torch(model.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        else:
            parameters["cams_embeds"] = ttnn.from_torch(
                torch.zeros(6, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

    return parameters


def create_bevformerv2_model_parameters_transformer(model, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_bevformerv2_transformer(
    device,
    reset_seeds,
    model_location_generator,
):
    torch_model = PerceptionTransformerV2(
        rotate_prev_bev=False, use_shift=False, use_can_bus=False, decoder=True, embed_dims=256
    )

    weights_path = download_bevformerv2_weights()
    checkpoint = torch.load(weights_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    transformer_state = {}
    for key, value in state_dict.items():
        if "transformer" in key:
            new_key = key.replace("pts_bbox_head.transformer.", "")
            transformer_state[new_key] = value

    torch_model.load_state_dict(transformer_state, strict=False)
    torch_model.eval()
    torch_model.encoder.layers = torch.nn.ModuleList(list(torch_model.encoder.layers)[:1])
    torch_model.encoder.num_layers = 1
    torch_model.decoder.layers = torch.nn.ModuleList(list(torch_model.decoder.layers)[:1])
    torch_model.decoder.num_layers = 1

    parameter = create_bevformerv2_model_parameters_transformer(torch_model, device=device)

    bev_h = 100
    bev_w = 100
    grid_length = (0.512, 0.512)
    bev_queries = torch.randn(10000, 256)
    object_query_embed = torch.randn(900, 512)
    map_query_embed = torch.randn(500, 512)
    mlvl_feats = []
    a = torch.randn(1, 6, 256, 12, 20)
    mlvl_feats.append(a)
    bev_pos = torch.randn(1, 256, 100, 100)

    img_metas = [
        {
            "can_bus": np.array([0.0] * 18),
            "lidar2img": [
                np.array(
                    [
                        [2.48597954e02, 1.68129905e02, 6.55251068e00, -7.08702279e01],
                        [-3.64025219e00, 1.07359713e02, -2.45107509e02, -1.28941576e02],
                        [-1.17025046e-02, 9.98471159e-01, 5.40221896e-02, -4.25203639e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [2.72989308e02, -1.23852972e02, -8.06783283e00, -9.23285717e01],
                        [7.58924673e01, 6.40614553e01, -2.47958947e02, -1.38511256e02],
                        [8.43406855e-01, 5.36312055e-01, 3.21598489e-02, -6.10371854e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [6.47396684e00, 3.00630854e02, 1.55246365e01, -6.04875770e01],
                        [-7.78640394e01, 6.40883103e01, -2.47490601e02, -1.35884951e02],
                        [-8.23415292e-01, 5.65940098e-01, 4.12196894e-02, -5.29677094e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [-1.60796449e02, -1.70144772e02, -5.28753263e00, -1.74159198e02],
                        [-2.16465632e00, -8.90571925e01, -1.62979489e02, -1.41736848e02],
                        [-8.33350064e-03, -9.99200442e-01, -3.91028008e-02, -1.01645350e00],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [-2.37313222e02, 1.84652288e02, 1.06528318e01, -1.25068238e02],
                        [-9.25251029e01, -2.05081174e01, -2.50495434e02, -1.12365691e02],
                        [-9.47586752e-01, -3.19482867e-01, 3.16948959e-03, -4.32527296e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [5.70378465e01, -2.93855304e02, -1.19126859e01, -5.45200638e01],
                        [8.89472086e01, -2.45651403e01, -2.50078534e02, -1.17649223e02],
                        [9.24052925e-01, -3.82246554e-01, -3.70989150e-03, -4.64645142e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            ],
            "img_shape": [(640, 360, 3), (640, 360, 3), (640, 360, 3), (640, 360, 3), (640, 360, 3), (640, 360, 3)],
        }
    ]

    model_outputs = torch_model(
        mlvl_feats,
        bev_queries,
        object_query_embed,
        bev_h,
        bev_w,
        grid_length=grid_length,
        bev_pos=bev_pos,
        reg_branches=None,
        img_metas=img_metas,
    )

    ttnn_model = TtPerceptionTransformer(
        params=parameter,
        device=device,
        rotate_prev_bev=False,
        use_shift=False,
        use_can_bus=False,
        decoder=True,
        map_decoder=None,
        embed_dims=256,
        encoder_num_layers=1,
        decoder_num_layers=1,
        params_branches=None,
    )

    bev_queries_ttnn = ttnn.from_torch(bev_queries, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    object_query_embed_ttnn = ttnn.from_torch(
        object_query_embed, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    mlvl_feats_ttnn = []
    mlvl_feats_ttnn.append(ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device))
    map_query_embed_ttnn = ttnn.from_torch(
        map_query_embed, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    bev_pos_ttnn = ttnn.from_torch(bev_pos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_outputs = ttnn_model(
        mlvl_feats_ttnn,
        bev_queries_ttnn,
        object_query_embed_ttnn,
        map_query_embed_ttnn,
        bev_h,
        bev_w,
        grid_length=grid_length,
        bev_pos=bev_pos_ttnn,
        reg_branches=None,
        map_reg_branches=None,
        img_metas=img_metas,
    )

    torch_bev_embed, torch_hs, torch_init_ref, torch_inter_refs = model_outputs
    ttnn_bev_embed, ttnn_hs, ttnn_init_ref, ttnn_inter_refs = (
        ttnn_outputs[0],
        ttnn_outputs[1],
        ttnn_outputs[2],
        ttnn_outputs[3],
    )

    result1 = assert_with_pcc(torch_bev_embed, ttnn.to_torch(ttnn_bev_embed).float(), 0.96)
    result2 = assert_with_pcc(torch_hs, ttnn.to_torch(ttnn_hs).float(), 0.98)
    result3 = assert_with_pcc(torch_init_ref, ttnn.to_torch(ttnn_init_ref).float(), 0.99)
    result4 = assert_with_pcc(torch_inter_refs, ttnn.to_torch(ttnn_inter_refs).float(), 0.98)
