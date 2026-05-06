# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for [`SeamlessM4Tv2Decoder`] (Hugging Face)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import SeamlessM4Tv2Config, SeamlessM4Tv2Model
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2Decoder


def tiny_decoder_config_for_tests(
    *,
    vocab_size: int = 512,
    hidden_size: int = 512,
    decoder_layers: int = 2,
    decoder_attention_heads: int = 8,
    decoder_ffn_dim: int = 2048,
    max_position_embeddings: int = 128,
) -> SeamlessM4Tv2Config:
    """Small config for fast PCC / CI (not the pretrained *large* checkpoint)."""
    return SeamlessM4Tv2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        decoder_layers=decoder_layers,
        decoder_attention_heads=decoder_attention_heads,
        encoder_attention_heads=decoder_attention_heads,
        decoder_ffn_dim=decoder_ffn_dim,
        encoder_ffn_dim=decoder_ffn_dim,
        encoder_layers=decoder_layers,
        max_position_embeddings=max_position_embeddings,
        scale_embedding=True,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        decoder_layerdrop=0.0,
        encoder_layerdrop=0.0,
    )


def create_torch_text_decoder(
    config: Optional[SeamlessM4Tv2Config] = None,
    *,
    seed: int = 0,
) -> Tuple[SeamlessM4Tv2Decoder, SeamlessM4Tv2Config]:
    """Build decoder with optional deterministic init."""
    cfg = config or tiny_decoder_config_for_tests()
    torch.manual_seed(seed)
    decoder = SeamlessM4Tv2Decoder(cfg)
    decoder.eval()
    return decoder, cfg


def forward_torch_reference(
    decoder: SeamlessM4Tv2Decoder,
    input_ids: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor],
    encoder_attention_mask: Optional[torch.LongTensor],
) -> torch.Tensor:
    """Decoder forward; returns `last_hidden_state` tensor [B, S, H]."""
    p0 = next(decoder.parameters())
    enc = encoder_hidden_states.to(device=p0.device, dtype=p0.dtype)
    with torch.no_grad():
        out = decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=enc,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    return out.last_hidden_state


def forward_torch_hidden_before_final_layer_norm(
    decoder: SeamlessM4Tv2Decoder,
    input_ids: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor],
    encoder_attention_mask: Optional[torch.LongTensor],
) -> torch.Tensor:
    """
    Same as a full decoder forward, but returns hidden states after the last decoder
    layer and before ``decoder.layer_norm`` (eval / no dropout).

    Use this for PCC against TTNN when comparing the full transformer stack without
    the final LayerNorm, which can amplify small bf16 drift on very deep checkpoints.
    """
    p0 = next(decoder.parameters())
    enc = encoder_hidden_states.to(device=p0.device, dtype=p0.dtype)
    decoder.eval()
    with torch.no_grad():
        if input_ids is not None:
            input_shape = input_ids.size()
            flat_ids = input_ids.view(-1, input_shape[-1])
        else:
            raise ValueError("input_ids required")

        inputs_embeds = decoder.embed_tokens(flat_ids)
        attn_4d = _prepare_4d_causal_attention_mask(
            attention_mask,
            input_shape,
            inputs_embeds,
            past_key_values_length=0,
        )
        enc_mask_4d = None
        if enc is not None and encoder_attention_mask is not None:
            enc_mask_4d = _prepare_4d_attention_mask(
                encoder_attention_mask,
                inputs_embeds.dtype,
                tgt_len=input_shape[-1],
            )

        input_for_pos = input_ids
        positions = decoder.embed_positions(input_for_pos, past_key_values_length=0)
        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)

        for decoder_layer in decoder.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attn_4d,
                encoder_hidden_states=enc,
                encoder_attention_mask=enc_mask_4d,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]

    return hidden_states


def load_pretrained_text_decoder(
    weights_dir: Union[str, Path],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[SeamlessM4Tv2Decoder, SeamlessM4Tv2Config]:
    """
    Load [`SeamlessM4Tv2Model`] from a local Hugging Face snapshot and return ``text_decoder`` + config.

    The decoder shares token embeddings with ``model.shared``; keeping the returned decoder alive
    retains those weights.
    """
    path = os.fspath(weights_dir)
    model = SeamlessM4Tv2Model.from_pretrained(
        path,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if dtype is not None:
        model.to(dtype)
    sample_w = next(model.parameters())
    if sample_w.is_floating_point() and dtype is not None and sample_w.dtype != dtype:
        raise RuntimeError(
            f"Expected loaded model weights in {dtype}, got {sample_w.dtype} for parameter shape {tuple(sample_w.shape)}."
        )
    decoder = model.text_decoder
    return decoder, model.config


def embed_scale_for_config(config: SeamlessM4Tv2Config) -> float:
    return math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
