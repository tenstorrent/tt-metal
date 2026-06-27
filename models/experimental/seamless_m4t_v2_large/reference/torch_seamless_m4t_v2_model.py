# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for [`SeamlessM4Tv2Model`] (Hugging Face).

[`SeamlessM4Tv2Model.generate`] is the custom two-stage pipeline (text ``GenerationMixin`` then
``t2u_model`` + ``vocoder``). This module exposes it explicitly; see upstream:

https://github.com/huggingface/transformers/blob/main/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import SeamlessM4Tv2Config, SeamlessM4Tv2Model
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2GenerationOutput


def load_pretrained_seamless_m4t_v2_model(
    weights_dir: Union[str, Path],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[SeamlessM4Tv2Model, SeamlessM4Tv2Config]:
    """
    Load the full [`SeamlessM4Tv2Model`] from a local Transformers snapshot.
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
    return model, model.config


def forward_text_modality_logits(
    model: SeamlessM4Tv2Model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    HF [`SeamlessM4Tv2Model.forward`] text path: ``text_encoder`` → ``text_decoder`` → ``lm_head``.

    Returns vocabulary logits ``[batch, dec_seq, vocab_size]``.
    """
    p0 = next(model.parameters())
    ii = input_ids.to(device=p0.device)
    am = attention_mask.to(device=p0.device)
    di = decoder_input_ids.to(device=p0.device)
    dm = decoder_attention_mask.to(device=p0.device)
    with torch.no_grad():
        out = model(
            input_ids=ii,
            attention_mask=am,
            decoder_input_ids=di,
            decoder_attention_mask=dm,
            use_cache=False,
            return_dict=True,
        )
    return out.logits


@torch.no_grad()
def generate(
    model: SeamlessM4Tv2Model,
    *,
    input_ids: Optional[torch.Tensor] = None,
    input_features: Optional[torch.Tensor] = None,
    return_intermediate_token_ids: Optional[bool] = None,
    tgt_lang: Optional[str] = None,
    speaker_id: int = 0,
    generate_speech: bool = True,
    **kwargs,
) -> Union[torch.Tensor, SeamlessM4Tv2GenerationOutput, tuple]:
    """
    Hugging Face [`SeamlessM4Tv2Model.generate`]: text generation (``super().generate``) then,
    when ``generate_speech=True``, T2U + vocoder for waveforms.

    Parameters match the Transformers implementation (including ``text_*`` / ``speech_*`` kwargs
    routing via ``format_speech_generation_kwargs`` inside the model).

    Returns:
        Same types as HF: ``ModelOutput`` / sequences when ``generate_speech=False``;
        ``(waveform, waveform_lengths)`` or [`SeamlessM4Tv2GenerationOutput`] when ``generate_speech=True``.
    """
    return model.generate(
        input_ids=input_ids,
        input_features=input_features,
        return_intermediate_token_ids=return_intermediate_token_ids,
        tgt_lang=tgt_lang,
        speaker_id=speaker_id,
        generate_speech=generate_speech,
        **kwargs,
    )
