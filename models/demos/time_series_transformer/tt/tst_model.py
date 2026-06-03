# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import torch
import ttnn
from .tst_encoder_layer import tst_encoder_layer
from .tst_decoder_layer import tst_decoder_layer
from .ttnn_utils import layer_norm_padded


def load_weights(hf_model, device):
    state = hf_model.state_dict()
    weights = {}

    def to_ttnn(key, transpose=False):
        t = state[key].float()
        if transpose:
            t = t.T.contiguous()
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def to_ttnn_1d(key):
        t = state[key].float()
        if t.shape[-1] % 32 != 0:
            pad_size = 32 - (t.shape[-1] % 32)
            t = torch.nn.functional.pad(t, (0, pad_size))
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def to_ttnn_bias(key):
        t = state[key].float()
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    for i in range(2):
        prefix = f"model.encoder.layers.{i}"
        out_prefix = f"encoder.layers.{i}"
        weights[out_prefix] = {
            "self_attn.q_proj.weight": to_ttnn(f"{prefix}.self_attn.q_proj.weight", transpose=True),
            "self_attn.k_proj.weight": to_ttnn(f"{prefix}.self_attn.k_proj.weight", transpose=True),
            "self_attn.v_proj.weight": to_ttnn(f"{prefix}.self_attn.v_proj.weight", transpose=True),
            "self_attn.out_proj.weight": to_ttnn(f"{prefix}.self_attn.out_proj.weight", transpose=True),
            "self_attn.q_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.q_proj.bias"),
            "self_attn.k_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.k_proj.bias"),
            "self_attn.v_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.v_proj.bias"),
            "self_attn.out_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.out_proj.bias"),
            "self_attn_layer_norm.weight": to_ttnn_1d(f"{prefix}.self_attn_layer_norm.weight"),
            "self_attn_layer_norm.bias": to_ttnn_1d(f"{prefix}.self_attn_layer_norm.bias"),
            "fc1.weight": to_ttnn(f"{prefix}.fc1.weight", transpose=True),
            "fc1.bias": to_ttnn_bias(f"{prefix}.fc1.bias"),
            "fc2.weight": to_ttnn(f"{prefix}.fc2.weight", transpose=True),
            "fc2.bias": to_ttnn_bias(f"{prefix}.fc2.bias"),
            "final_layer_norm.weight": to_ttnn_1d(f"{prefix}.final_layer_norm.weight"),
            "final_layer_norm.bias": to_ttnn_1d(f"{prefix}.final_layer_norm.bias"),
        }

    for i in range(2):
        prefix = f"model.decoder.layers.{i}"
        out_prefix = f"decoder.layers.{i}"
        weights[out_prefix] = {
            "self_attn.q_proj.weight": to_ttnn(f"{prefix}.self_attn.q_proj.weight", transpose=True),
            "self_attn.k_proj.weight": to_ttnn(f"{prefix}.self_attn.k_proj.weight", transpose=True),
            "self_attn.v_proj.weight": to_ttnn(f"{prefix}.self_attn.v_proj.weight", transpose=True),
            "self_attn.out_proj.weight": to_ttnn(f"{prefix}.self_attn.out_proj.weight", transpose=True),
            "self_attn.q_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.q_proj.bias"),
            "self_attn.k_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.k_proj.bias"),
            "self_attn.v_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.v_proj.bias"),
            "self_attn.out_proj.bias": to_ttnn_bias(f"{prefix}.self_attn.out_proj.bias"),
            "self_attn_layer_norm.weight": to_ttnn_1d(f"{prefix}.self_attn_layer_norm.weight"),
            "self_attn_layer_norm.bias": to_ttnn_1d(f"{prefix}.self_attn_layer_norm.bias"),
            "encoder_attn.q_proj.weight": to_ttnn(f"{prefix}.encoder_attn.q_proj.weight", transpose=True),
            "encoder_attn.k_proj.weight": to_ttnn(f"{prefix}.encoder_attn.k_proj.weight", transpose=True),
            "encoder_attn.v_proj.weight": to_ttnn(f"{prefix}.encoder_attn.v_proj.weight", transpose=True),
            "encoder_attn.out_proj.weight": to_ttnn(f"{prefix}.encoder_attn.out_proj.weight", transpose=True),
            "encoder_attn.q_proj.bias": to_ttnn_bias(f"{prefix}.encoder_attn.q_proj.bias"),
            "encoder_attn.k_proj.bias": to_ttnn_bias(f"{prefix}.encoder_attn.k_proj.bias"),
            "encoder_attn.v_proj.bias": to_ttnn_bias(f"{prefix}.encoder_attn.v_proj.bias"),
            "encoder_attn.out_proj.bias": to_ttnn_bias(f"{prefix}.encoder_attn.out_proj.bias"),
            "encoder_attn_layer_norm.weight": to_ttnn_1d(f"{prefix}.encoder_attn_layer_norm.weight"),
            "encoder_attn_layer_norm.bias": to_ttnn_1d(f"{prefix}.encoder_attn_layer_norm.bias"),
            "fc1.weight": to_ttnn(f"{prefix}.fc1.weight", transpose=True),
            "fc1.bias": to_ttnn_bias(f"{prefix}.fc1.bias"),
            "fc2.weight": to_ttnn(f"{prefix}.fc2.weight", transpose=True),
            "fc2.bias": to_ttnn_bias(f"{prefix}.fc2.bias"),
            "final_layer_norm.weight": to_ttnn_1d(f"{prefix}.final_layer_norm.weight"),
            "final_layer_norm.bias": to_ttnn_1d(f"{prefix}.final_layer_norm.bias"),
        }

    weights["encoder_layernorm"] = {
        "weight": to_ttnn_1d("model.encoder.layernorm_embedding.weight"),
        "bias":   to_ttnn_1d("model.encoder.layernorm_embedding.bias"),
    }
    weights["decoder_layernorm"] = {
        "weight": to_ttnn_1d("model.decoder.layernorm_embedding.weight"),
        "bias":   to_ttnn_1d("model.decoder.layernorm_embedding.bias"),
    }

    weights["encoder_value_proj"] = state["model.encoder.value_embedding.value_projection.weight"].float()
    weights["decoder_value_proj"] = state["model.decoder.value_embedding.value_projection.weight"].float()
    weights["encoder_pos_emb"]    = state["model.encoder.embed_positions.weight"].float()
    weights["decoder_pos_emb"]    = state["model.decoder.embed_positions.weight"].float()
    weights["cat_embedder"]       = state["model.embedder.embedders.0.weight"].float()

    return weights


def run_encoder(device, encoder_input, weights, apply_layernorm=False):
    """
    encoder_input: torch tensor [B, context_length, d_model], already embedded.
    apply_layernorm: set True only when input has NOT yet been through layernorm_embedding.
                     When using HF hook output (post-layernorm), leave False.
    """
    hidden = ttnn.from_torch(
        encoder_input.float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    if apply_layernorm:
        hidden = layer_norm_padded(
            hidden,
            weight=weights["encoder_layernorm"]["weight"],
            bias=weights["encoder_layernorm"]["bias"],
        )
    for i in range(2):
        hidden = tst_encoder_layer(device, hidden, weights, layer_idx=i)
    return hidden


def run_decoder_step(device, decoder_input, encoder_hidden, weights, apply_layernorm=False):
    """
    decoder_input: torch tensor [B, seq, d_model], already embedded.
    apply_layernorm: set True only when input has NOT yet been through layernorm_embedding.
                     When using HF hook output (post-layernorm), leave False.
    """
    hidden = ttnn.from_torch(
        decoder_input.float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    if apply_layernorm:
        hidden = layer_norm_padded(
            hidden,
            weight=weights["decoder_layernorm"]["weight"],
            bias=weights["decoder_layernorm"]["bias"],
        )
    for i in range(2):
        hidden = tst_decoder_layer(device, hidden, encoder_hidden, weights, layer_idx=i)
    return hidden
