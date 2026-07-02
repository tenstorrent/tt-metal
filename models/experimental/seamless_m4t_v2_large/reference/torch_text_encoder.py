# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for [`SeamlessM4Tv2Encoder`] (Hugging Face)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import SeamlessM4Tv2Config, SeamlessM4Tv2Model
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2Encoder


def tiny_encoder_config_for_tests(
    *,
    vocab_size: int = 512,
    hidden_size: int = 512,
    encoder_layers: int = 2,
    encoder_attention_heads: int = 8,
    encoder_ffn_dim: int = 2048,
    max_position_embeddings: int = 128,
) -> SeamlessM4Tv2Config:
    """Small config for fast PCC / CI (not the pretrained *large* checkpoint)."""
    return SeamlessM4Tv2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        encoder_layers=encoder_layers,
        encoder_attention_heads=encoder_attention_heads,
        decoder_attention_heads=encoder_attention_heads,
        encoder_ffn_dim=encoder_ffn_dim,
        decoder_ffn_dim=encoder_ffn_dim,
        decoder_layers=encoder_layers,
        max_position_embeddings=max_position_embeddings,
        scale_embedding=True,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        decoder_layerdrop=0.0,
        encoder_layerdrop=0.0,
    )


def create_torch_text_encoder(
    config: Optional[SeamlessM4Tv2Config] = None,
    *,
    seed: int = 0,
) -> Tuple[SeamlessM4Tv2Encoder, SeamlessM4Tv2Config]:
    """Build encoder with optional deterministic init."""
    cfg = config or tiny_encoder_config_for_tests()
    torch.manual_seed(seed)
    encoder = SeamlessM4Tv2Encoder(cfg)
    encoder.eval()
    return encoder, cfg


def forward_torch_reference(
    encoder: SeamlessM4Tv2Encoder,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.LongTensor],
) -> torch.Tensor:
    """Encoder forward; returns `last_hidden_state` tensor [B, S, H]."""
    p0 = next(encoder.parameters())
    with torch.no_grad():
        out = encoder(
            input_ids=input_ids.to(device=p0.device),
            attention_mask=attention_mask.to(device=p0.device) if attention_mask is not None else None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    return out.last_hidden_state


def load_pretrained_text_encoder(
    weights_dir: Union[str, Path],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[SeamlessM4Tv2Encoder, SeamlessM4Tv2Config]:
    """
    Load [`SeamlessM4Tv2Model`] from a local Hugging Face snapshot and return ``text_encoder`` + config.
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
    encoder = model.text_encoder
    return encoder, model.config
