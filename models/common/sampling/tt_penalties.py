# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
On-device penalties module with persistent buffers, mirroring TTSampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


@dataclass
class PenaltyContext:
    prompt_mask: ttnn.Tensor
    output_mask: ttnn.Tensor
    output_counts: ttnn.Tensor
    presence_penalties: ttnn.Tensor
    frequency_penalties: ttnn.Tensor
    repetition_penalties: ttnn.Tensor


def _token_bin_counts_and_mask(
    tokens: torch.Tensor | None,
    max_batch_size: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build dense histograms/masks for the provided token tensor (host-side).
    """
    counts = torch.zeros((max_batch_size, vocab_size), dtype=torch.int32)
    mask = torch.zeros_like(counts)

    if tokens is None:
        return counts, mask

    tokens = tokens.long()
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(1)

    batch = min(tokens.shape[0], max_batch_size)
    if batch == 0 or tokens.shape[1] == 0:
        return counts, mask

    tokens = tokens[:batch]
    valid = tokens >= 0
    clamped = torch.clamp(tokens, min=0, max=vocab_size - 1)
    updates = valid.to(torch.int32)

    counts_slice = counts[:batch]
    counts_slice.scatter_add_(1, clamped, updates)
    mask[:batch] = (counts_slice > 0).to(torch.int32)
    counts[:batch] = counts_slice
    return counts, mask


def apply_penalties(logits: ttnn.Tensor, context: Optional[PenaltyContext]) -> ttnn.Tensor:
    if context is None:
        return logits

    # frequency
    freq_term = ttnn.multiply(context.output_counts, context.frequency_penalties)
    logits = ttnn.subtract(logits, freq_term, output_tensor=logits)

    # presence
    presence_term = ttnn.multiply(context.output_mask, context.presence_penalties)
    logits = ttnn.subtract(logits, presence_term, output_tensor=logits)

    # repetition
    combined_mask = ttnn.add(context.prompt_mask, context.output_mask)
    penalties = ttnn.where(combined_mask, context.repetition_penalties, 1.0)
    inverse_penalties = ttnn.div(1, penalties)
    scaling = ttnn.where(ttnn.gt(logits, 1), inverse_penalties, penalties)
    logits = ttnn.multiply(logits, scaling, output_tensor=logits)

    return logits


class TTPenalties(LightweightModule):
    """
    Penalty module with persistent device tensors, similar to TTSampling.
    """

    def __init__(self, mesh_device, max_batch_size: int, vocab_size: int):
        super().__init__()
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self._zero_int_host = torch.zeros((self.max_batch_size, self.vocab_size), dtype=torch.int32)

        self.prompt_mask = self._alloc_int_buffer()
        self.output_mask = self._alloc_int_buffer()
        self.output_counts = self._alloc_int_buffer()

        self.presence_penalties = self._alloc_bf16_buffer()
        self.frequency_penalties = self._alloc_bf16_buffer()
        self.repetition_penalties = self._alloc_bf16_buffer()

    def _alloc_int_buffer(self):
        host = torch.zeros((self.max_batch_size, self.vocab_size), dtype=torch.int32)
        return ttnn.from_torch(host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)

    def _alloc_bf16_buffer(self):
        host = torch.zeros((self.max_batch_size, 1), dtype=torch.float32)
        return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)

    def _copy_host_to_device(self, dst: ttnn.Tensor, src: torch.Tensor):
        src_tt = ttnn.from_torch(src, dtype=dst.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=None)
        ttnn.copy_host_to_device_tensor(src_tt, dst)

    def reset_params(self, presence: List[float], frequency: List[float], repetition: List[float]):
        presence_tensor = self._pad_params(presence)
        frequency_tensor = self._pad_params(frequency)
        repetition_tensor = self._pad_params(repetition)

        self._copy_host_to_device(self.presence_penalties, presence_tensor)
        self._copy_host_to_device(self.frequency_penalties, frequency_tensor)
        self._copy_host_to_device(self.repetition_penalties, repetition_tensor)

    def _pad_params(self, values: List[float]) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.numel() < self.max_batch_size:
            pad_value = tensor[-1] if tensor.numel() > 0 else torch.tensor(0.0)
            pad = pad_value.repeat(self.max_batch_size - tensor.numel())
            tensor = torch.cat([tensor, pad])
        elif tensor.numel() > self.max_batch_size:
            tensor = tensor[: self.max_batch_size]
        return tensor.view(self.max_batch_size, 1)

    def reset_prompt_tokens(self, prompt_tokens: torch.Tensor | None):
        _, mask = _token_bin_counts_and_mask(prompt_tokens, self.max_batch_size, self.vocab_size)
        self._copy_host_to_device(self.prompt_mask, mask)

    def reset_output_tokens(self):
        self.output_mask = ttnn.mul(self.output_mask, 0, output_tensor=self.output_mask)
        self.output_counts = ttnn.mul(self.output_counts, 0, output_tensor=self.output_counts)

    def update_output_tokens(self, new_tokens):
        if new_tokens is None:
            return
        counts_delta, _ = _token_bin_counts_and_mask(new_tokens, self.max_batch_size, self.vocab_size)
        counts_delta_tt = ttnn.from_torch(
            counts_delta,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None,
        )
        ttnn.add(self.output_counts, counts_delta_tt, output_tensor=self.output_counts)
        self.output_mask = ttnn.gt(self.output_counts, 0, output_tensor=self.output_mask)

    def apply(self, tt_logits: ttnn.Tensor, batch_size: int) -> ttnn.Tensor:
        if tt_logits is None:
            return tt_logits

        context = PenaltyContext(
            prompt_mask=self.prompt_mask,
            output_mask=self.output_mask,
            output_counts=self.output_counts,
            presence_penalties=self.presence_penalties,
            frequency_penalties=self.frequency_penalties,
            repetition_penalties=self.repetition_penalties,
        )

        reshaped = ttnn.reshape(tt_logits, (-1, self.vocab_size))
        apply_penalties(reshaped, context)
        return ttnn.reshape(reshaped, (1, 1, -1, self.vocab_size))
