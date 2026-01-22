# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np
import torch.nn as nn
from loguru import logger

from models.experimental.MapTR.reference.maptr import (
    MapTRPerceptionTransformer,
    MapTRDecoder,
)
from models.experimental.MapTR.reference.bevformer import (
    BEVFormerEncoder,
    TemporalSelfAttention,
)
from models.experimental.MapTR.reference.bevformer import (
    SpatialCrossAttention,
)
from models.experimental.MapTR.resources.download_chkpoint import ensure_checkpoint_downloaded, MAPTR_WEIGHTS_PATH
from models.experimental.MapTR.tt.ttnn_transformer import TtMapTRPerceptionTransformer
from models.experimental.MapTR.tt.ttnn_encoder import TtBEVFormerEncoder
from models.experimental.MapTR.tt.ttnn_decoder import TtMapTRDecoder
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)

TRANSFORMER_LAYER_PREFIX = "pts_bbox_head.transformer."
ENCODER_LAYER_PREFIX = "pts_bbox_head.transformer.encoder.layers."
DECODER_LAYER_PREFIX = "pts_bbox_head.transformer.decoder.layers."


def load_maptr_transformer_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    ensure_checkpoint_downloaded(weights_path)

    checkpoint = torch.load(weights_path, map_location="cpu")
    full_state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    transformer_weights = {}
    encoder_weights = {}
    decoder_weights = {}

    for key, value in full_state_dict.items():
        if key.startswith(TRANSFORMER_LAYER_PREFIX):
            relative_key = key[len(TRANSFORMER_LAYER_PREFIX) :]
            if relative_key.startswith("encoder.layers."):
                encoder_relative_key = relative_key[len("encoder.layers.") :]
                encoder_weights[encoder_relative_key] = value
            elif relative_key.startswith("decoder.layers."):
                decoder_relative_key = relative_key[len("decoder.layers.") :]
                decoder_weights[decoder_relative_key] = value
            else:
                transformer_weights[relative_key] = value

    logger.info(
        f"Loaded {len(transformer_weights)} transformer weights, {len(encoder_weights)} encoder weights, {len(decoder_weights)} decoder weights"
    )
    return transformer_weights, encoder_weights, decoder_weights


def load_torch_transformer_model(torch_model: MapTRPerceptionTransformer, weights_path: str = MAPTR_WEIGHTS_PATH):
    transformer_weights, encoder_weights, decoder_weights = load_maptr_transformer_weights(weights_path)
    model_state_dict = torch_model.state_dict()
    new_state_dict = {}

    for model_key in model_state_dict.keys():
        if model_key.startswith("encoder.layers."):
            encoder_key = model_key[len("encoder.layers.") :]
            if encoder_key in encoder_weights:
                new_state_dict[model_key] = encoder_weights[encoder_key]
            else:
                logger.warning(f"Encoder weight not found: {model_key}")
                new_state_dict[model_key] = model_state_dict[model_key]
        elif model_key.startswith("decoder.layers."):
            decoder_key = model_key[len("decoder.layers.") :]
            if decoder_key in decoder_weights:
                new_state_dict[model_key] = decoder_weights[decoder_key]
            else:
                logger.warning(f"Decoder weight not found: {model_key}")
                new_state_dict[model_key] = model_state_dict[model_key]
        elif model_key in transformer_weights:
            checkpoint_weight = transformer_weights[model_key]
            model_weight = model_state_dict[model_key]
            if checkpoint_weight.shape != model_weight.shape:
                if model_key == "level_embeds" and len(checkpoint_weight.shape) == 2 and len(model_weight.shape) == 2:
                    new_state_dict[model_key] = checkpoint_weight[: model_weight.shape[0]]
                else:
                    logger.warning(f"Shape mismatch for {model_key}: {checkpoint_weight.shape} vs {model_weight.shape}")
                    new_state_dict[model_key] = model_weight
            else:
                new_state_dict[model_key] = checkpoint_weight
        else:
            logger.warning(f"Transformer weight not found: {model_key}")
            new_state_dict[model_key] = model_state_dict[model_key]

    missing_keys, unexpected_keys = torch_model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing {len(missing_keys)} keys")
    if unexpected_keys:
        logger.warning(f"Unexpected {len(unexpected_keys)} keys")
    torch_model.eval()
    return torch_model


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
                        "weight": preprocess_linear_weight(ffn.layers[0].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[0].bias, dtype=ttnn.bfloat16),
                    },
                    "linear2": {
                        "weight": preprocess_linear_weight(ffn.layers[3].weight, dtype=ttnn.bfloat16),
                        "bias": preprocess_linear_bias(ffn.layers[3].bias, dtype=ttnn.bfloat16),
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

                elif hasattr(attn, "attn"):
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
                else:
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

    if isinstance(model, MapTRPerceptionTransformer):
        parameters = {}
        if hasattr(model, "encoder") and isinstance(model.encoder, BEVFormerEncoder):
            parameters["encoder"] = extract_transformer_parameters(model.encoder)
        if hasattr(model, "decoder") and isinstance(model.decoder, MapTRDecoder):
            parameters["decoder"] = extract_transformer_parameters(model.decoder)
        parameters["reference_points"] = {
            "weight": preprocess_linear_weight(model.reference_points.weight, dtype=ttnn.bfloat16),
            "bias": preprocess_linear_bias(model.reference_points.bias, dtype=ttnn.bfloat16),
        }
        parameters["can_bus_mlp"] = {
            "0": {
                "weight": preprocess_linear_weight(model.can_bus_mlp[0].weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.can_bus_mlp[0].bias, dtype=ttnn.bfloat16),
            },
            "2": {
                "weight": preprocess_linear_weight(model.can_bus_mlp[2].weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.can_bus_mlp[2].bias, dtype=ttnn.bfloat16),
            },
            "norm": {
                "weight": preprocess_layernorm_parameter(model.can_bus_mlp.norm.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_layernorm_parameter(model.can_bus_mlp.norm.bias, dtype=ttnn.bfloat16),
            },
        }
        parameters["level_embeds"] = ttnn.from_torch(model.level_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        parameters["cams_embeds"] = ttnn.from_torch(model.cams_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    return parameters


class ParamsWrapper:
    def __init__(self, params_dict):
        for k, v in params_dict.items():
            setattr(self, k, self._dict_to_obj(v))

    def _dict_to_obj(self, d):
        if isinstance(d, dict):
            obj = type("obj", (object,), {})()
            for k, v in d.items():
                setattr(obj, k, self._dict_to_obj(v))
            return obj
        return d


def create_maptr_model_parameters(model: MapTRPerceptionTransformer, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_transformer(device, reset_seeds):
    embed_dims = 256
    num_feature_levels = 1
    num_cams = 6
    bev_h, bev_w = 50, 32
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    num_vec = 10
    num_pts_per_vec = 10
    num_query = num_vec * num_pts_per_vec

    encoder_cfg = dict(
        type="BEVFormerEncoder",
        num_layers=1,
        pc_range=pc_range,
        num_points_in_pillar=4,
        return_intermediate=False,
        transformerlayers=dict(
            type="BEVFormerLayer",
            attn_cfgs=[
                dict(type="TemporalSelfAttention", embed_dims=embed_dims, num_levels=1),
                dict(
                    type="SpatialCrossAttention",
                    pc_range=pc_range,
                    deformable_attention=dict(
                        type="MSDeformableAttention3D",
                        embed_dims=embed_dims,
                        num_points=8,
                        num_levels=1,
                    ),
                    embed_dims=embed_dims,
                ),
            ],
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        ),
    )

    decoder_cfg = dict(
        type="MapTRDecoder",
        num_layers=6,
        return_intermediate=True,
        transformerlayers=dict(
            type="DetrTransformerDecoderLayer",
            attn_cfgs=[
                dict(type="MultiheadAttention", embed_dims=embed_dims, num_heads=8, dropout=0.1),
                dict(type="CustomMSDeformableAttention", embed_dims=embed_dims, num_levels=1),
            ],
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        ),
    )

    torch_model = MapTRPerceptionTransformer(
        encoder=encoder_cfg,
        decoder=decoder_cfg,
        embed_dims=embed_dims,
        num_feature_levels=num_feature_levels,
        num_cams=num_cams,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        len_can_bus=18,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[bev_h // 2, bev_w // 2],
    )
    torch_model = load_torch_transformer_model(torch_model)

    for module in torch_model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0

    feat_h, feat_w = 28, 50
    mlvl_feats_torch = torch.randn(1, num_cams, embed_dims, feat_h, feat_w)
    bev_queries = torch.randn(bev_h * bev_w, embed_dims)
    object_query_embed = torch.randn(num_query, embed_dims * 2)
    bev_pos = torch.randn(1, embed_dims, bev_h, bev_w)

    img_metas = [
        {
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
                ],
                dtype=np.float32,
            ),
            "lidar2img": [np.eye(4, dtype=np.float32) for _ in range(num_cams)],
            "img_shape": [(900, 1600, 3)] * num_cams,
        }
    ]

    with torch.no_grad():
        torch_outputs = torch_model(
            mlvl_feats=[mlvl_feats_torch],
            lidar_feat=None,
            bev_queries=bev_queries,
            object_query_embed=object_query_embed,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            img_metas=img_metas,
        )

    parameters = create_maptr_model_parameters(torch_model, device=device)
    encoder_params = ParamsWrapper(parameters.get("encoder", {}).get("layers", {}))
    encoder_params.layers = encoder_params
    decoder_params = ParamsWrapper(parameters.get("decoder", {}).get("layers", {}))
    decoder_params.layers = decoder_params

    tt_encoder = TtBEVFormerEncoder(
        params=encoder_params,
        device=device,
        num_layers=1,
        pc_range=pc_range,
        num_points_in_pillar=4,
        return_intermediate=False,
        embed_dims=embed_dims,
        feedforward_channels=512,
        num_levels=1,
        num_points=8,
    )

    tt_decoder = TtMapTRDecoder(
        num_layers=6,
        embed_dims=embed_dims,
        num_heads=8,
        params=decoder_params,
        params_branches=None,
        device=device,
        feedforward_channels=512,
    )

    class AttrDict(dict):
        def __getattr__(self, key):
            try:
                value = self[key]
                if isinstance(value, dict):
                    return AttrDict(value)
                return value
            except KeyError:
                raise AttributeError(key)

    class TransformerParams:
        def __init__(self, params_dict):
            for k, v in params_dict.items():
                if isinstance(v, dict):
                    setattr(self, k, AttrDict(v))
                else:
                    setattr(self, k, v)

    transformer_params = TransformerParams(parameters)
    tt_model = TtMapTRPerceptionTransformer(
        params=transformer_params,
        device=device,
        encoder=tt_encoder,
        decoder=tt_decoder,
        embed_dims=embed_dims,
        num_feature_levels=num_feature_levels,
        num_cams=num_cams,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[bev_h // 2, bev_w // 2],
    )

    tt_mlvl_feats = [ttnn.from_torch(mlvl_feats_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)]
    tt_bev_queries = ttnn.from_torch(bev_queries, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_object_query_embed = ttnn.from_torch(
        object_query_embed, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tt_bev_pos = ttnn.from_torch(bev_pos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_outputs = tt_model(
        mlvl_feats=tt_mlvl_feats,
        lidar_feat=None,
        bev_queries=tt_bev_queries,
        object_query_embed=tt_object_query_embed,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=tt_bev_pos,
        img_metas=img_metas,
    )

    pcc_bev_passed, _ = assert_with_pcc(torch_outputs[0], ttnn.to_torch(tt_outputs[0]).float(), 0.90)
    pcc_states_passed, _ = assert_with_pcc(torch_outputs[1], ttnn.to_torch(tt_outputs[1]).float(), 0.90)
    pcc_init_ref_passed, _ = assert_with_pcc(torch_outputs[2], ttnn.to_torch(tt_outputs[2]).float(), 0.95)
    pcc_inter_ref_passed, _ = assert_with_pcc(torch_outputs[3], ttnn.to_torch(tt_outputs[3]).float(), 0.90)

    assert pcc_bev_passed and pcc_states_passed and pcc_init_ref_passed and pcc_inter_ref_passed, "PCC checks failed"
