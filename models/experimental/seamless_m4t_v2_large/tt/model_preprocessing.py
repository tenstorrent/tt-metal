# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import ttnn
import torch
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight, make_parameter_dict

from models.experimental.seamless_m4t_v2_large.reference.torch_text_decoder import embed_scale_for_config


def _conv1d_weight(conv: torch.nn.Conv1d, *, device: ttnn.Device) -> ttnn.Tensor:
    w = conv.weight.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _conv1d_bias(conv: torch.nn.Conv1d, *, device: ttnn.Device) -> Optional[ttnn.Tensor]:
    if conv.bias is None:
        return None
    b = conv.bias.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        b.reshape(1, 1, 1, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _conv_transpose1d_weight(conv: torch.nn.ConvTranspose1d, *, device: ttnn.Device) -> ttnn.Tensor:
    """PyTorch ``[in_c, out_c, K]`` -> TTNN conv_transpose2d-style ``[in_c, out_c, K, 1]``."""
    w = conv.weight.detach().to(torch.bfloat16).contiguous()
    w2 = w.unsqueeze(-1).contiguous()
    return ttnn.from_torch(
        w2,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _conv_transpose1d_bias_host(conv: torch.nn.ConvTranspose1d) -> Optional[ttnn.Tensor]:
    if conv.bias is None:
        return None
    b = conv.bias.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        b.reshape(1, 1, 1, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _embedding_weight(emb: torch.nn.Embedding, *, device: ttnn.Device) -> ttnn.Tensor:
    w = emb.weight.detach().to(torch.bfloat16).contiguous()
    return ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


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


def _conv1d_like_padding_int(conv: torch.nn.Module) -> int:
    """
    TTNN paths need an integral symmetric padding. HF may use ``padding='same'`` on
    [`torch.nn.Conv1d`] (string); map that to PyTorch's stride-1 effective padding.
    """
    p = conv.padding
    if isinstance(p, str):
        if p == "same":
            k = int(conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size)
            d = int(conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation)
            s = int(conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride)
            if s != 1:
                raise ValueError("padding='same' with stride != 1 is not supported for TT export")
            return (k - 1) * d // 2
        if p in ("valid", "zeros"):
            return 0
        raise ValueError(f"Unsupported Conv1d padding mode {p!r}")
    if isinstance(p, (tuple, list)):
        return int(p[0])
    return int(p)


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
    out = {
        "conv1": {
            "weight": _conv1d_weight(var_pred.conv1, device=device),
            "bias": _conv1d_bias(var_pred.conv1, device=device),
        },
        "ln1": {
            "weight": _ln_to_device(var_pred.ln1.weight, device=device),
            "bias": _ln_to_device(var_pred.ln1.bias, device=device),
        },
        "conv2": {
            "weight": _conv1d_weight(var_pred.conv2, device=device),
            "bias": _conv1d_bias(var_pred.conv2, device=device),
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
            "weight": _conv1d_weight(layer.conv1, device=device),
            "bias": _conv1d_bias(layer.conv1, device=device),
        },
        "conv2": {
            "weight": _conv1d_weight(layer.conv2, device=device),
            "bias": _conv1d_bias(layer.conv2, device=device),
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
    scale = embed_scale_for_config(cfg)
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


def create_code_hifigan_parameters(vocoder, *, device: ttnn.Device) -> dict:
    """
    Convert [`SeamlessM4Tv2CodeHifiGan`] (``model.vocoder``) to TTNN tensors for
    [`TTSeamlessM4Tv2CodeHifiGan`].
    """
    dp = vocoder.dur_predictor
    dur_predictor = {
        "conv1": {
            "weight": _conv1d_weight(dp.conv1, device=device),
            "bias": _conv1d_bias(dp.conv1, device=device),
            "in_channels": dp.conv1.in_channels,
            "out_channels": dp.conv1.out_channels,
            "kernel_size": int(dp.conv1.kernel_size[0]),
            "padding": _conv1d_like_padding_int(dp.conv1),
        },
        "ln1": {
            "weight": _ln_to_device(dp.ln1.weight, device=device),
            "bias": _ln_to_device(dp.ln1.bias, device=device),
            "eps": float(dp.ln1.eps),
        },
        "conv2": {
            "weight": _conv1d_weight(dp.conv2, device=device),
            "bias": _conv1d_bias(dp.conv2, device=device),
            "in_channels": dp.conv2.in_channels,
            "out_channels": dp.conv2.out_channels,
            "kernel_size": int(dp.conv2.kernel_size[0]),
            "padding": _conv1d_like_padding_int(dp.conv2),
        },
        "ln2": {
            "weight": _ln_to_device(dp.ln2.weight, device=device),
            "bias": _ln_to_device(dp.ln2.bias, device=device),
            "eps": float(dp.ln2.eps),
        },
        "proj": _linear_pair(dp.proj, device=device),
    }

    hg = vocoder.hifi_gan
    upsampler_layers = []
    for layer in hg.upsampler:
        assert isinstance(layer, torch.nn.ConvTranspose1d)
        k = int(layer.kernel_size[0])
        s = int(layer.stride[0])
        p = _conv1d_like_padding_int(layer)
        upsampler_layers.append(
            {
                "weight": _conv_transpose1d_weight(layer, device=device),
                "bias": _conv_transpose1d_bias_host(layer),
                "kernel_size": k,
                "stride": s,
                "padding": p,
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
            }
        )

    resblock_layers = []
    for rb in hg.resblocks:
        c1 = []
        c2 = []
        for c in rb.convs1:
            c1.append(
                {
                    "weight": _conv1d_weight(c, device=device),
                    "bias": _conv1d_bias(c, device=device),
                    "kernel_size": int(c.kernel_size[0]),
                    "dilation": int(c.dilation[0]),
                    "padding": _conv1d_like_padding_int(c),
                    "in_channels": c.in_channels,
                    "out_channels": c.out_channels,
                }
            )
        for c in rb.convs2:
            c2.append(
                {
                    "weight": _conv1d_weight(c, device=device),
                    "bias": _conv1d_bias(c, device=device),
                    "kernel_size": int(c.kernel_size[0]),
                    "dilation": int(c.dilation[0]),
                    "padding": _conv1d_like_padding_int(c),
                    "in_channels": c.in_channels,
                    "out_channels": c.out_channels,
                }
            )
        resblock_layers.append(make_parameter_dict({"convs1": c1, "convs2": c2}))

    out = {
        "unit_embedding": make_parameter_dict({"weight": _embedding_weight(vocoder.unit_embedding, device=device)}),
        "speaker_embedding": make_parameter_dict(
            {"weight": _embedding_weight(vocoder.speaker_embedding, device=device)}
        ),
        "language_embedding": make_parameter_dict(
            {"weight": _embedding_weight(vocoder.language_embedding, device=device)}
        ),
        "dur_predictor": make_parameter_dict(dur_predictor),
        "hifi_gan": make_parameter_dict(
            {
                "conv_pre": {
                    "weight": _conv1d_weight(hg.conv_pre, device=device),
                    "bias": _conv1d_bias(hg.conv_pre, device=device),
                    "kernel_size": int(hg.conv_pre.kernel_size[0]),
                    "padding": _conv1d_like_padding_int(hg.conv_pre),
                    "in_channels": hg.conv_pre.in_channels,
                    "out_channels": hg.conv_pre.out_channels,
                },
                "upsampler": upsampler_layers,
                "resblocks": resblock_layers,
                "conv_post": {
                    "weight": _conv1d_weight(hg.conv_post, device=device),
                    "bias": _conv1d_bias(hg.conv_post, device=device),
                    "kernel_size": int(hg.conv_post.kernel_size[0]),
                    "padding": _conv1d_like_padding_int(hg.conv_post),
                    "in_channels": hg.conv_post.in_channels,
                    "out_channels": hg.conv_post.out_channels,
                },
            }
        ),
    }
    return make_parameter_dict(out)
