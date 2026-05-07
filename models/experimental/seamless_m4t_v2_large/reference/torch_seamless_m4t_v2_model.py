# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for [`SeamlessM4Tv2Model`] (Hugging Face)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union

import torch
from transformers import SeamlessM4Tv2Config, SeamlessM4Tv2Model


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
