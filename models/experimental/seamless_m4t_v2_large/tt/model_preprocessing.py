# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import ttnn
import torch
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight, make_parameter_dict

from models.experimental.seamless_m4t_v2_large.reference.torch_text_decoder import embed_scale_for_config


def _ln_to_device(param: torch.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    x = param.detach().reshape(1, 1, -1).contiguous()
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _linear_pair(linear: torch.nn.Linear, *, device: ttnn.Device) -> dict:
    w = preprocess_linear_weight(linear.weight.detach(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b = preprocess_linear_bias(linear.bias.detach(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return {
        "weight": ttnn.to_device(w, device),
        "bias": ttnn.to_device(b, device),
    }


def create_text_decoder_parameters(decoder, *, device: ttnn.Device) -> dict:
    """
    Convert [`SeamlessM4Tv2Decoder`] weights to TTNN tensors on ``device``.

    Token embeddings include ``embed_scale`` (see [`SeamlessM4Tv2ScaledWordEmbedding`]).
    """
    cfg = decoder.config
    scale = embed_scale_for_config(cfg)

    scaled_emb = (decoder.embed_tokens.weight.detach() * scale).contiguous()
    embed_tokens_weight = ttnn.from_torch(
        scaled_emb,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    pos_w = decoder.embed_positions.weights.detach()
    if pos_w.dtype != torch.bfloat16:
        pos_w = pos_w.to(dtype=torch.bfloat16)
    embed_positions_weight = ttnn.from_torch(
        pos_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    layers = []
    for layer in decoder.layers:
        layer_dict = {
            "self_attn_layer_norm": {
                "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
            },
            "self_attn": {
                "q_proj": _linear_pair(layer.self_attn.q_proj, device=device),
                "k_proj": _linear_pair(layer.self_attn.k_proj, device=device),
                "v_proj": _linear_pair(layer.self_attn.v_proj, device=device),
                "out_proj": _linear_pair(layer.self_attn.out_proj, device=device),
            },
            "cross_attention_layer_norm": {
                "weight": _ln_to_device(layer.cross_attention_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.cross_attention_layer_norm.bias, device=device),
            },
            "cross_attention": {
                "q_proj": _linear_pair(layer.cross_attention.q_proj, device=device),
                "k_proj": _linear_pair(layer.cross_attention.k_proj, device=device),
                "v_proj": _linear_pair(layer.cross_attention.v_proj, device=device),
                "out_proj": _linear_pair(layer.cross_attention.out_proj, device=device),
            },
            "ffn_layer_norm": {
                "weight": _ln_to_device(layer.ffn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.ffn_layer_norm.bias, device=device),
            },
            "ffn": {
                "fc1": _linear_pair(layer.ffn.fc1, device=device),
                "fc2": _linear_pair(layer.ffn.fc2, device=device),
            },
        }
        layers.append(make_parameter_dict(layer_dict))

    out = {
        "embed_tokens": make_parameter_dict({"weight": embed_tokens_weight}),
        "embed_positions": make_parameter_dict({"weight": embed_positions_weight}),
        "layers": layers,
        "layer_norm": make_parameter_dict(
            {
                "weight": _ln_to_device(decoder.layer_norm.weight, device=device),
                "bias": _ln_to_device(decoder.layer_norm.bias, device=device),
            }
        ),
    }
    return make_parameter_dict(out)


def _m4t_encoder_self_attn_ffn_layers(encoder, *, device: ttnn.Device) -> list:
    """Layer parameter dicts shared by text encoder and text-to-unit encoder stacks."""
    layers = []
    for layer in encoder.layers:
        layer_dict = {
            "self_attn_layer_norm": {
                "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
            },
            "self_attn": {
                "q_proj": _linear_pair(layer.self_attn.q_proj, device=device),
                "k_proj": _linear_pair(layer.self_attn.k_proj, device=device),
                "v_proj": _linear_pair(layer.self_attn.v_proj, device=device),
                "out_proj": _linear_pair(layer.self_attn.out_proj, device=device),
            },
            "ffn_layer_norm": {
                "weight": _ln_to_device(layer.ffn_layer_norm.weight, device=device),
                "bias": _ln_to_device(layer.ffn_layer_norm.bias, device=device),
            },
            "ffn": {
                "fc1": _linear_pair(layer.ffn.fc1, device=device),
                "fc2": _linear_pair(layer.ffn.fc2, device=device),
            },
        }
        layers.append(make_parameter_dict(layer_dict))
    return layers


def create_text_encoder_parameters(encoder, *, device: ttnn.Device) -> dict:
    """
    Convert [`SeamlessM4Tv2Encoder`] weights to TTNN tensors on ``device``.

    Token embeddings include ``embed_scale`` (see [`SeamlessM4Tv2ScaledWordEmbedding`]).
    """
    cfg = encoder.config
    scale = embed_scale_for_config(cfg)

    scaled_emb = (encoder.embed_tokens.weight.detach() * scale).contiguous()
    embed_tokens_weight = ttnn.from_torch(
        scaled_emb,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    pos_w = encoder.embed_positions.weights.detach()
    if pos_w.dtype != torch.bfloat16:
        pos_w = pos_w.to(dtype=torch.bfloat16)
    embed_positions_weight = ttnn.from_torch(
        pos_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    layers = _m4t_encoder_self_attn_ffn_layers(encoder, device=device)

    out = {
        "embed_tokens": make_parameter_dict({"weight": embed_tokens_weight}),
        "embed_positions": make_parameter_dict({"weight": embed_positions_weight}),
        "layers": layers,
        "layer_norm": make_parameter_dict(
            {
                "weight": _ln_to_device(encoder.layer_norm.weight, device=device),
                "bias": _ln_to_device(encoder.layer_norm.bias, device=device),
            }
        ),
    }
    return make_parameter_dict(out)


def create_text_to_unit_parameters(encoder, *, device: ttnn.Device) -> dict:
    """
    Convert the encoder submodule of Transformers
    [`SeamlessM4Tv2TextToUnitForConditionalGeneration`] — ``model.encoder``, i.e.
    [`SeamlessM4Tv2Encoder`] with ``is_t2u_encoder=True`` — to TTNN tensors for
    [`TTSeamlessM4Tv2TextToUnitEncoder`].

    Expects ``encoder.layers`` as [`SeamlessM4Tv2EncoderLayer`] (self-attention + FFN only).
    Weights cover the transformer stack only (``inputs_embeds`` path; no token or position
    embeddings on this submodule).
    """
    layers = _m4t_encoder_self_attn_ffn_layers(encoder, device=device)
    out = {
        "layers": layers,
        "layer_norm": make_parameter_dict(
            {
                "weight": _ln_to_device(encoder.layer_norm.weight, device=device),
                "bias": _ln_to_device(encoder.layer_norm.bias, device=device),
            }
        ),
    }
    return make_parameter_dict(out)


def _t2u_variance_predictor_parameters(var_pred: torch.nn.Module, *, device: ttnn.Device) -> dict:
    """[`SeamlessM4Tv2VariancePredictor`] weights for TTNN text-to-unit duration path."""
    conv1_w = var_pred.conv1.weight.detach()
    conv1_b = var_pred.conv1.bias.detach() if var_pred.conv1.bias is not None else None
    conv2_w = var_pred.conv2.weight.detach()
    conv2_b = var_pred.conv2.bias.detach() if var_pred.conv2.bias is not None else None
    out = {
        "conv1": {
            "weight": ttnn.from_torch(conv1_w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            "bias": (
                ttnn.from_torch(conv1_b.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                if conv1_b is not None
                else None
            ),
        },
        "ln1": {
            "weight": _ln_to_device(var_pred.ln1.weight, device=device),
            "bias": _ln_to_device(var_pred.ln1.bias, device=device),
        },
        "conv2": {
            "weight": ttnn.from_torch(conv2_w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            "bias": (
                ttnn.from_torch(conv2_b.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                if conv2_b is not None
                else None
            ),
        },
        "ln2": {
            "weight": _ln_to_device(var_pred.ln2.weight, device=device),
            "bias": _ln_to_device(var_pred.ln2.bias, device=device),
        },
        "proj": _linear_pair(var_pred.proj, device=device),
    }
    return make_parameter_dict(out)


def _t2u_decoder_layer_parameters(layer: torch.nn.Module, *, device: ttnn.Device) -> dict:
    """[`SeamlessM4Tv2TextToUnitDecoderLayer`] weights."""
    conv1_w = layer.conv1.weight.detach()
    conv1_b = layer.conv1.bias.detach() if layer.conv1.bias is not None else None
    conv2_w = layer.conv2.weight.detach()
    conv2_b = layer.conv2.bias.detach() if layer.conv2.bias is not None else None
    layer_dict = {
        "self_attn": {
            "q_proj": _linear_pair(layer.self_attn.q_proj, device=device),
            "k_proj": _linear_pair(layer.self_attn.k_proj, device=device),
            "v_proj": _linear_pair(layer.self_attn.v_proj, device=device),
            "out_proj": _linear_pair(layer.self_attn.out_proj, device=device),
        },
        "self_attn_layer_norm": {
            "weight": _ln_to_device(layer.self_attn_layer_norm.weight, device=device),
            "bias": _ln_to_device(layer.self_attn_layer_norm.bias, device=device),
        },
        "conv1": {
            "weight": ttnn.from_torch(conv1_w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            "bias": (
                ttnn.from_torch(conv1_b.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                if conv1_b is not None
                else None
            ),
        },
        "conv2": {
            "weight": ttnn.from_torch(conv2_w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            "bias": (
                ttnn.from_torch(conv2_b.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                if conv2_b is not None
                else None
            ),
        },
        "conv_layer_norm": {
            "weight": _ln_to_device(layer.conv_layer_norm.weight, device=device),
            "bias": _ln_to_device(layer.conv_layer_norm.bias, device=device),
        },
    }
    return make_parameter_dict(layer_dict)


def _t2u_decoder_parameters(decoder: torch.nn.Module, *, device: ttnn.Device) -> dict:
    """[`SeamlessM4Tv2TextToUnitDecoder`] weights (character + duration + conv decoder stack)."""
    cfg = decoder.config
    scale = math.sqrt(cfg.hidden_size) if cfg.scale_embedding else 1.0
    scaled_char = (decoder.embed_char.weight.detach() * scale).contiguous()
    embed_char_weight = ttnn.from_torch(
        scaled_char,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    char_pos_w = decoder.embed_char_positions.weights.detach()
    if char_pos_w.dtype != torch.bfloat16:
        char_pos_w = char_pos_w.to(dtype=torch.bfloat16)
    embed_char_positions_weight = ttnn.from_torch(
        char_pos_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_w = decoder.embed_positions.weights.detach()
    if pos_w.dtype != torch.bfloat16:
        pos_w = pos_w.to(dtype=torch.bfloat16)
    embed_positions_weight = ttnn.from_torch(
        pos_w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_emb_alpha_char = ttnn.from_torch(
        decoder.pos_emb_alpha_char.detach().reshape(1, 1, 1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_emb_alpha = ttnn.from_torch(
        decoder.pos_emb_alpha.detach().reshape(1, 1, 1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    layers = [_t2u_decoder_layer_parameters(layer, device=device) for layer in decoder.layers]
    out = {
        "embed_char": make_parameter_dict({"weight": embed_char_weight}),
        "embed_char_positions": make_parameter_dict({"weight": embed_char_positions_weight}),
        "embed_positions": make_parameter_dict({"weight": embed_positions_weight}),
        "pos_emb_alpha_char": pos_emb_alpha_char,
        "pos_emb_alpha": pos_emb_alpha,
        "duration_predictor": _t2u_variance_predictor_parameters(decoder.duration_predictor, device=device),
        "layers": layers,
        "layer_norm": make_parameter_dict(
            {
                "weight": _ln_to_device(decoder.layer_norm.weight, device=device),
                "bias": _ln_to_device(decoder.layer_norm.bias, device=device),
            }
        ),
    }
    return make_parameter_dict(out)


def create_text_to_unit_condgen_parameters(
    t2u: torch.nn.Module,
    *,
    device: ttnn.Device,
) -> dict:
    """
    Full [`SeamlessM4Tv2TextToUnitForConditionalGeneration`] weights for TTNN:
    ``model.encoder``, ``model.decoder``, and ``lm_head``.
    """
    w_lm = preprocess_linear_weight(
        t2u.lm_head.weight.detach(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    out = {
        "encoder": create_text_to_unit_parameters(t2u.model.encoder, device=device),
        "decoder": _t2u_decoder_parameters(t2u.model.decoder, device=device),
        "lm_head": make_parameter_dict({"weight": ttnn.to_device(w_lm, device)}),
    }
    return make_parameter_dict(out)
