# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
import ttnn
import torch.nn as nn
from models.common.utility_functions import comp_pcc

from models.experimental.BEVFormerV2.reference.encoder import BEVFormerEncoder
from models.experimental.BEVFormerV2.reference.temporal_self_attention import TemporalSelfAttention
from models.experimental.BEVFormerV2.reference.spatial_cross_attention import SpatialCrossAttention
from models.experimental.BEVFormerV2.tt.ttnn_encoder import TtEncoder
from models.experimental.BEVFormerV2.common import download_bevformerv2_weights
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)


def custom_preprocessor(model, name):
    parameters = {}

    def extract_encoder_parameters(encoder_module):
        parameters = {"layers": {}}

        for i, layer in enumerate(encoder_module.layers):
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

            parameters["layers"][f"layer{i}"] = layer_dict
        return parameters

    if isinstance(model, BEVFormerEncoder):
        parameters = extract_encoder_parameters(model)

    return parameters


def create_bevformerv2_encoder_parameters(model, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_encoder_pcc(
    device,
    reset_seeds,
    model_location_generator,
):
    torch.manual_seed(42)

    try:
        pytorch_encoder = BEVFormerEncoder(
            num_layers=1,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            num_points_in_pillar=4,
            return_intermediate=False,
            embed_dims=256,
            num_heads=8,
            feedforward_channels=512,
        )

        weights_path = download_bevformerv2_weights()
        checkpoint = torch.load(weights_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        encoder_state = {}
        for key, value in state_dict.items():
            if "encoder" in key:
                new_key = key.replace("pts_bbox_head.transformer.encoder.", "")
                encoder_state[new_key] = value

        pytorch_encoder.load_state_dict(encoder_state, strict=False)
        pytorch_encoder.eval()
    except Exception as e:
        pytest.skip(f"Failed to create models: {e}")

    img_metas = [
        {
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
                        [2.85189232e02, -1.46927652e03, -5.95634293e01, -2.72600319e02],
                        [4.44736043e02, -1.22825701e02, -1.25039267e03, -5.88246115e02],
                        [9.24052925e-01, -3.82246554e-01, -3.70989150e-03, -4.64645142e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            ],
            "img_shape": [(640, 360, 3), (640, 360, 3), (640, 360, 3), (640, 360, 3), (640, 360, 3), (640, 360, 3)],
        }
    ]

    batch_size = 1
    embed_dim = 256
    bev_h = 100
    bev_w = 100
    sequence_length = bev_h * bev_w

    bev_query = torch.rand(sequence_length, batch_size, embed_dim, dtype=torch.float32)
    key = torch.rand(6, 240, 1, 256, dtype=torch.float32)
    value = torch.rand(6, 240, 1, 256, dtype=torch.float32)
    bev_pos = torch.rand(sequence_length, batch_size, embed_dim, dtype=torch.float32)
    spatial_shapes = torch.tensor([[12, 20]], dtype=torch.int32)
    level_start_index = torch.tensor([0], dtype=torch.int64)
    shift = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    with torch.no_grad():
        torch_output = pytorch_encoder(
            bev_query=bev_query,
            key=key,
            value=value,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=None,
            prev_bev=None,
            shift=shift,
            img_metas=img_metas,
        )

    encoder_params = create_bevformerv2_encoder_parameters(pytorch_encoder, device=device)

    ttnn_encoder = TtEncoder(
        params=encoder_params,
        device=device,
        num_layers=1,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        num_points_in_pillar=4,
        return_intermediate=False,
        embed_dims=256,
        num_heads=8,
        feedforward_channels=512,
    )

    key_ttnn = ttnn.from_torch(
        key.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    value_ttnn = ttnn.from_torch(
        value.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bev_pos_ttnn = ttnn.from_torch(
        bev_pos.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    spatial_shapes_ttnn = ttnn.from_torch(
        spatial_shapes.to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    level_start_index_ttnn = ttnn.from_torch(
        level_start_index.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bev_query_ttnn = ttnn.from_torch(
        bev_query.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    shift_ttnn = ttnn.from_torch(
        shift.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn_encoder(
        bev_query=bev_query_ttnn,
        key=key_ttnn,
        value=value_ttnn,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=bev_pos_ttnn,
        spatial_shapes=spatial_shapes_ttnn,
        level_start_index=level_start_index_ttnn,
        valid_ratios=None,
        prev_bev=None,
        shift=shift_ttnn,
        img_metas=img_metas,
    )

    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    if ttnn_output_torch.shape != torch_output.shape:
        raise ValueError(f"Shape mismatch: ttnn={ttnn_output_torch.shape}, ref={torch_output.shape}")

    pcc_result = comp_pcc(torch_output, ttnn_output_torch)
    pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result

    print(f"BEVFormerEncoder PCC: {pcc_value:.6f}")
    assert pcc_value > 0.98, f"BEVFormerEncoder PCC {pcc_value:.6f} is below threshold 0.98"
