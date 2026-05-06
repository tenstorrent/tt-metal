# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for [`SeamlessM4Tv2Decoder`] (Hugging Face)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import SeamlessM4Tv2Config, SeamlessM4Tv2Model
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2Decoder


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
