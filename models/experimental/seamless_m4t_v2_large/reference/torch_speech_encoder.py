# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for [`SeamlessM4Tv2SpeechEncoder`] (Hugging Face)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import SeamlessM4Tv2Config, SeamlessM4Tv2Model
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2SpeechEncoder


def forward_torch_speech_encoder(
    speech_encoder: SeamlessM4Tv2SpeechEncoder,
    input_features: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Returns `last_hidden_state` tensor ``[B, S, H]``."""
    p0 = next(speech_encoder.parameters())
    with torch.no_grad():
        out = speech_encoder(
            input_features=input_features.to(device=p0.device, dtype=p0.dtype),
            attention_mask=attention_mask.to(device=p0.device) if attention_mask is not None else None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    return out.last_hidden_state


def load_pretrained_speech_encoder(
    weights_dir: Union[str, Path],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[SeamlessM4Tv2SpeechEncoder, SeamlessM4Tv2Config]:
    """
    Load [`SeamlessM4Tv2Model`] from a local Hugging Face snapshot and return ``speech_encoder`` + config.
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
    return model.speech_encoder, model.config
