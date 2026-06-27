# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference for Hugging Face [`SeamlessM4Tv2CodeHifiGan`] (vocoder)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import torch
from transformers import SeamlessM4Tv2Config, SeamlessM4Tv2Model


def forward_torch_code_hifigan_reference(
    vocoder,
    input_ids: torch.Tensor,
    speaker_id: torch.Tensor,
    lang_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs HF vocoder forward; returns ``(waveform, lengths)`` like HF."""
    p0 = next(vocoder.parameters())
    with torch.no_grad():
        waveform, lengths = vocoder(
            input_ids=input_ids.to(device=p0.device),
            speaker_id=speaker_id.to(device=p0.device),
            lang_id=lang_id.to(device=p0.device),
        )
    return waveform, lengths


def load_pretrained_code_hifigan(
    weights_dir: Union[str, Path],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[object, SeamlessM4Tv2Config]:
    """
    Load [`SeamlessM4Tv2Model`] from a local snapshot and return ``vocoder`` + config.

    The vocoder is [`SeamlessM4Tv2CodeHifiGan``] attached at ``model.vocoder``.
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
    return model.vocoder, model.config


def compute_dur_output_lengths(
    input_ids: torch.Tensor,
    dur_out: torch.Tensor,
    *,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Port of ``SeamlessM4Tv2CodeHifiGan._get_dur_output_lengths`` (CPU torch).
    ``dur_out``: integer durations ``[B, T]``.
    """
    unit_lengths = (input_ids != pad_token_id).sum(1)
    unit_lengths = torch.clamp(unit_lengths, 0, dur_out.shape[1] - 1)
    cumulative_dur_out = torch.cumsum(dur_out, dim=1)
    return cumulative_dur_out.gather(dim=1, index=unit_lengths.unsqueeze(1)).squeeze()


def compute_hifigan_output_lengths(unit_lengths: torch.Tensor, config: SeamlessM4Tv2Config) -> torch.Tensor:
    """
    Port of ``SeamlessM4Tv2CodeHifiGan._get_output_hifigan_lengths`` (CPU torch).
    """

    def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
        return (
            torch.div(
                input_length + 2 * pad - dilation * (kernel_size - 1) - 1,
                stride,
                rounding_mode="floor",
            )
            + 1
        )

    def _transpose_conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
        return (input_length - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1

    input_lengths = unit_lengths
    input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

    for upsample_rate, kernel_size in zip(config.upsample_rates, config.upsample_kernel_sizes):
        input_lengths = _transpose_conv_out_length(
            input_lengths, kernel_size, upsample_rate, (kernel_size - upsample_rate) // 2
        )

    for _ in range(len(config.upsample_rates)):
        for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
            for dil in dilation:
                input_lengths = _conv_out_length(
                    input_lengths, kernel_size, 1, (kernel_size - 1) * dil // 2, dilation=dil
                )
            for _ in dilation:
                input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) // 2, dilation=1)

    input_lengths = _conv_out_length(input_lengths, 7, 1, 3)
    return input_lengths
