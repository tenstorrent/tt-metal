# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for [`SeamlessM4Tv2TextToUnitForConditionalGeneration`] (Hugging Face)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union

import torch
from transformers import SeamlessM4Tv2Config, SeamlessM4Tv2Model
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import (
    SeamlessM4Tv2TextToUnitForConditionalGeneration,
)


def load_pretrained_text_to_unit(
    weights_dir: Union[str, Path],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[SeamlessM4Tv2TextToUnitForConditionalGeneration, SeamlessM4Tv2Config]:
    """
    Load the text-to-unit submodule from a local Transformers snapshot.

    Weights are stored under the full [`SeamlessM4Tv2Model`] checkpoint with prefix ``t2u_model.``;
    loading the parent model and returning ``t2u_model`` ensures all parameters load correctly.

    The returned [`SeamlessM4Tv2Config`] is ``t2u_model.config``, not the parent model config: HF merges
    ``t2u_encoder_*`` (etc.) into ``encoder_*`` on that submodule, so ``encoder_layers`` /
    ``encoder_attention_heads`` match [`SeamlessM4Tv2TextToUnitForConditionalGeneration.model.encoder`].
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
    t2u = model.t2u_model
    return t2u, t2u.config


def forward_t2u_logits(
    t2u: SeamlessM4Tv2TextToUnitForConditionalGeneration,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    char_input_ids: torch.Tensor,
    char_count_per_id: torch.Tensor,
) -> torch.Tensor:
    """
    Run the text-to-unit model and return vocabulary logits ``[batch, unit_seq_len, t2u_vocab_size]``.

    ``inputs_embeds`` feeds the **encoder** (see HF ``SeamlessM4Tv2TextToUnitModel.forward``):
    shape ``[batch, encoder_seq_len, hidden_size]``.
    """
    p0 = next(t2u.parameters())
    ie = inputs_embeds.to(device=p0.device, dtype=p0.dtype)
    am = attention_mask.to(device=p0.device)
    cid = char_input_ids.to(device=p0.device)
    cc = char_count_per_id.to(device=p0.device)
    with torch.no_grad():
        out = t2u(
            inputs_embeds=ie,
            attention_mask=am,
            char_input_ids=cid,
            char_count_per_id=cc,
            return_dict=True,
        )
    # HF names this field ``last_hidden_state`` but it holds LM logits for this head.
    return out.last_hidden_state


def forward_t2u_logits_and_padding(
    t2u: SeamlessM4Tv2TextToUnitForConditionalGeneration,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    char_input_ids: torch.Tensor,
    char_count_per_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Same forward as ``forward_t2u_logits``, but also returns HF ``padding_mask`` (float, 1 = valid).

    With ``return_dict=True``, [`SeamlessM4Tv2TextToUnitOutput`] exposes up to eight optional fields;
    in the common case (no ``labels``, no hidden-state flags), the main tensors are **two**:
    ``last_hidden_state`` (LM logits, misnamed) and ``padding_mask``.
    """
    p0 = next(t2u.parameters())
    ie = inputs_embeds.to(device=p0.device, dtype=p0.dtype)
    am = attention_mask.to(device=p0.device)
    cid = char_input_ids.to(device=p0.device)
    cc = char_count_per_id.to(device=p0.device)
    with torch.no_grad():
        out = t2u(
            inputs_embeds=ie,
            attention_mask=am,
            char_input_ids=cid,
            char_count_per_id=cc,
            return_dict=True,
        )
    return out.last_hidden_state, out.padding_mask


def forward_t2u_encoder_hidden(
    t2u: SeamlessM4Tv2TextToUnitForConditionalGeneration,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Run ``SeamlessM4Tv2TextToUnitForConditionalGeneration.model.encoder`` only
    (``inputs_embeds`` → encoder ``layer_norm``).

    Use this as the PyTorch reference for PCC tests against [`TTSeamlessM4Tv2TextToUnitEncoder`].
    """
    p0 = next(t2u.parameters())
    ie = inputs_embeds.to(device=p0.device, dtype=p0.dtype)
    am = attention_mask.to(device=p0.device)
    with torch.no_grad():
        enc = t2u.model.encoder(inputs_embeds=ie, attention_mask=am, return_dict=True)
    return enc.last_hidden_state


def synthetic_t2u_inputs(
    config: SeamlessM4Tv2Config,
    *,
    batch: int = 1,
    encoder_seq_len: int = 8,
    chars_per_encoder_step: int = 2,
    seed: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a minimal valid batch for ``forward_t2u_logits`` (batch size 1 recommended).

    ``char_count_per_id[b, t]`` controls upsampling from encoder frames to character positions;
    ``char_input_ids`` length must equal ``char_count_per_id[b].sum()``.
    """
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cpu")
    h = config.hidden_size
    inputs_embeds = torch.randn(batch, encoder_seq_len, h, device=device, dtype=dtype)
    attention_mask = torch.ones(batch, encoder_seq_len, dtype=torch.long, device=device)
    char_count_per_id = torch.full(
        (batch, encoder_seq_len),
        chars_per_encoder_step,
        dtype=torch.long,
        device=device,
    )
    char_len = int(char_count_per_id[0].sum().item())
    high = min(config.char_vocab_size - 1, 2**31 - 1)
    low = 1
    char_input_ids = torch.randint(low, high, (batch, char_len), dtype=torch.long, device=device)
    return inputs_embeds, attention_mask, char_input_ids, char_count_per_id
