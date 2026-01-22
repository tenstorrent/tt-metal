# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import torch.nn as nn
from loguru import logger
from models.experimental.MapTR.reference.maptr import MapTRDecoder
from models.experimental.MapTR.reference.dependency import (
    MultiheadAttention,
)
from models.experimental.MapTR.reference.bevformer import (
    CustomMSDeformableAttention,
)
from models.experimental.MapTR.resources.download_chkpoint import ensure_checkpoint_downloaded, MAPTR_WEIGHTS_PATH
from models.experimental.MapTR.tt.ttnn_decoder import TtMapTRDecoder
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)

# Layer prefix for MapTRDecoder in MapTR
# MapTR uses: pts_bbox_head.transformer.decoder (for map decoder, it might be map_decoder)
# For MapTR, the map decoder path might be different - checking the actual structure
MAP_DECODER_LAYER = "pts_bbox_head.transformer.decoder.layers."


def load_maptr_decoder_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    ensure_checkpoint_downloaded(weights_path)

    checkpoint = torch.load(weights_path, map_location="cpu")
    full_state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    decoder_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(MAP_DECODER_LAYER):
            relative_key = key[len(MAP_DECODER_LAYER) :]
            decoder_weights[relative_key] = value

    logger.info(f"Loaded {len(decoder_weights)} weight tensors for MapTRDecoder")
    return decoder_weights


def load_torch_model_maptr(torch_model: MapTRDecoder, weights_path: str = MAPTR_WEIGHTS_PATH):
    decoder_weights = load_maptr_decoder_weights(weights_path)
    model_state_dict = torch_model.state_dict()
    new_state_dict = {}
    matched_keys = []
    missing_keys = []

    for model_key in model_state_dict.keys():
        if model_key.startswith("layers."):
            relative_key = model_key[7:]
        else:
            relative_key = model_key

        # Handle FFN key mapping: Model (flat) vs Checkpoint (nested Sequential)
        checkpoint_key = relative_key
        if "ffns.0.layers.0.weight" in relative_key:
            checkpoint_key = relative_key.replace("ffns.0.layers.0.weight", "ffns.0.layers.0.0.weight")
        elif "ffns.0.layers.0.bias" in relative_key:
            checkpoint_key = relative_key.replace("ffns.0.layers.0.bias", "ffns.0.layers.0.0.bias")
        elif "ffns.0.layers.3.weight" in relative_key:
            checkpoint_key = relative_key.replace("ffns.0.layers.3.weight", "ffns.0.layers.1.weight")
        elif "ffns.0.layers.3.bias" in relative_key:
            checkpoint_key = relative_key.replace("ffns.0.layers.3.bias", "ffns.0.layers.1.bias")

        found = False
        for key in [checkpoint_key, relative_key]:
            if key in decoder_weights and decoder_weights[key].shape == model_state_dict[model_key].shape:
                new_state_dict[model_key] = decoder_weights[key]
                matched_keys.append(model_key)
                found = True
                break

        if not found:
            logger.warning(f"Weight not found: {model_key}")
            new_state_dict[model_key] = model_state_dict[model_key]
            missing_keys.append(model_key)

    logger.info(f"Matched {len(matched_keys)}/{len(model_state_dict)} weights")
    if missing_keys:
        logger.warning(f"Missing {len(missing_keys)} weights")
    torch_model.load_state_dict(new_state_dict, strict=False)
    torch_model.eval()
    return torch_model


def extract_transformer_parameters(transformer_module):
    parameters = {"layers": {}}

    for i, layer in enumerate(transformer_module.layers):  # BaseTransformerLayer
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
                    "weight": preprocess_linear_weight(ffn.layers[0].weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(ffn.layers[0].bias, dtype=ttnn.bfloat16),
                },
                "linear2": {
                    "weight": preprocess_linear_weight(ffn.layers[3].weight, dtype=ttnn.bfloat16),
                    "bias": preprocess_linear_bias(ffn.layers[3].bias, dtype=ttnn.bfloat16),
                },
            }

        # ---- Attentions ----
        for j, attn in enumerate(getattr(layer, "attentions", [])):
            if isinstance(attn, MultiheadAttention):
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


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, MapTRDecoder):
        parameters = extract_transformer_parameters(model)

    return parameters


def create_maptr_model_parameters_decoder(model: MapTRDecoder, device=None):
    # Ensure model is in eval mode before preprocessing
    model.eval()

    # Create a closure to capture the model instance
    def get_model():
        return model

    parameters = preprocess_model_parameters(
        initialize_model=get_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_decoder(device, reset_seeds):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    num_layers = 6
    embed_dims = 256
    num_heads = 8
    feedforward_channels = 512

    transformerlayers_cfg = dict(
        type="DetrTransformerDecoderLayer",
        attn_cfgs=[
            dict(type="MultiheadAttention", embed_dims=embed_dims, num_heads=num_heads, dropout=0.1),
            dict(type="CustomMSDeformableAttention", embed_dims=embed_dims, num_levels=1),
        ],
        feedforward_channels=feedforward_channels,
        ffn_dropout=0.1,
        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
    )

    torch_model = MapTRDecoder(transformerlayer=transformerlayers_cfg, num_layers=num_layers, return_intermediate=True)
    torch_model = load_torch_model_maptr(torch_model)
    torch_model.eval()

    # Disable dropout for deterministic results
    for layer in torch_model.layers:
        for attn in layer.attentions:
            if hasattr(attn, "proj_drop") and isinstance(attn.proj_drop, nn.Dropout):
                attn.proj_drop = nn.Identity()
            if hasattr(attn, "dropout_layer") and isinstance(attn.dropout_layer, nn.Dropout):
                attn.dropout_layer = nn.Identity()
            if hasattr(attn, "dropout") and isinstance(attn.dropout, nn.Dropout):
                attn.dropout = nn.Identity()

    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    batch_size = 1
    num_query = 900
    spatial_h, spatial_w = 200, 100
    num_value = spatial_h * spatial_w

    query = torch.randn(num_query, batch_size, embed_dims) * 0.1
    value = torch.randn(num_value, batch_size, embed_dims) * 0.1
    query_pos = torch.randn(num_query, batch_size, embed_dims) * 0.1
    reference_points = torch.rand(batch_size, num_query, 2) * 0.8 + 0.1
    spatial_shapes = torch.tensor([[spatial_h, spatial_w]], dtype=torch.long)
    level_start_index = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        torch_output, torch_reference_points = torch_model(
            query=query,
            key=None,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            reg_branches=None,
        )

    parameter = create_maptr_model_parameters_decoder(torch_model, device=device)
    tt_model = TtMapTRDecoder(
        num_layers=num_layers,
        embed_dims=embed_dims,
        num_heads=num_heads,
        params=parameter,
        params_branches=None,
        device=device,
        feedforward_channels=feedforward_channels,
    )

    query_tt = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    value_tt = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    query_pos_tt = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    reference_points_tt = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    spatial_shapes_tt = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    level_start_index_tt = ttnn.from_torch(
        level_start_index, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    tt_output, tt_reference_points = tt_model(
        query=query_tt,
        key=None,
        value=value_tt,
        query_pos=query_pos_tt,
        reference_points=reference_points_tt,
        spatial_shapes=spatial_shapes_tt,
        level_start_index=level_start_index_tt,
        map_reg_branches=None,
    )

    ttnn_output = ttnn.to_torch(tt_output).float()
    ttnn_reference_points = ttnn.to_torch(tt_reference_points).float()
    torch_output = torch_output.float()
    torch_reference_points = torch_reference_points.float()

    assert torch_output.shape == ttnn_output.shape, f"Shape mismatch: {torch_output.shape} vs {ttnn_output.shape}"
    assert (
        torch_reference_points.shape == ttnn_reference_points.shape
    ), f"Shape mismatch: {torch_reference_points.shape} vs {ttnn_reference_points.shape}"

    pcc_passed_output, pcc_message_output = assert_with_pcc(ttnn_output, torch_output, 0.99)
    pcc_passed_ref, pcc_message_ref = assert_with_pcc(ttnn_reference_points, torch_reference_points, 0.99)
    assert pcc_passed_output, pcc_message_output
    assert pcc_passed_ref, pcc_message_ref
