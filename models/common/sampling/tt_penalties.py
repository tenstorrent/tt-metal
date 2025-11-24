# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
On-device penalties module with persistent buffers, mirroring TTSampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


@dataclass
class PenaltyContext:
    prompt_mask: ttnn.Tensor
    output_mask: ttnn.Tensor
    output_counts: ttnn.Tensor
    output_counts_gathered: ttnn.Tensor
    presence_penalties: ttnn.Tensor
    frequency_penalties: ttnn.Tensor
    repetition_penalties: ttnn.Tensor
    sub_core_grids: Any | None = None


def apply_penalties(logits: ttnn.Tensor, context: Optional[PenaltyContext]) -> ttnn.Tensor:
    if context is None:
        return logits

    op_kwargs = {"sub_core_grids": context.sub_core_grids} if context.sub_core_grids else {}

    # frequency
    freq_term = ttnn.multiply(context.output_counts, context.frequency_penalties, **op_kwargs)
    logits = ttnn.subtract(logits, freq_term, output_tensor=logits, **op_kwargs)

    # presence
    presence_term = ttnn.multiply(context.output_mask, context.presence_penalties, **op_kwargs)
    logits = ttnn.subtract(logits, presence_term, output_tensor=logits, **op_kwargs)

    # repetition

    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.

    combined_mask = ttnn.add(context.prompt_mask, context.output_mask, **op_kwargs)
    penalties = ttnn.where(combined_mask, context.repetition_penalties, 1.0, **op_kwargs)
    inverse_penalties = ttnn.reciprocal(penalties, **op_kwargs)

    # If logits are positive, divide by penalty, otherwise multiply by penalty.

    scaling = ttnn.where(ttnn.gt(logits, 1), inverse_penalties, penalties, **op_kwargs)

    logits = ttnn.multiply(logits, scaling, output_tensor=logits, **op_kwargs)

    return logits


class TTPenalties(LightweightModule):
    """
    Penalty module with persistent device tensors, similar to TTSampling.
    """

    def __init__(self, mesh_device, max_batch_size: int, vocab_size: int, sub_core_grids=None):
        super().__init__()
        self.mesh_device = mesh_device
        self.cluster_shape = mesh_device.shape
        self.max_batch_size = 32  # max_batch_size
        self.vocab_size = 128 * 1024  # vocab_size
        self.sub_core_grids = sub_core_grids
        self._op_kwargs = {"sub_core_grids": sub_core_grids} if sub_core_grids else {}

        self.prompt_mask = self._alloc_int_buffer()
        self.output_mask = self._alloc_int_buffer()
        self.output_counts_gathered = self._alloc_int_buffer(shard_dims=(None, None))
        self.output_counts = self._alloc_int_buffer()

        self.presence_penalties = self._alloc_bf16_buffer()
        self.frequency_penalties = self._alloc_bf16_buffer()
        self.repetition_penalties = self._alloc_bf16_buffer()

        self.slice_start = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=self.mesh_device)
        num_devices = mesh_device.shape[-1]
        end_tensor = torch.tensor(
            [[31] * num_devices, [(n + 1) * (self.vocab_size // num_devices) - 1 for n in range(num_devices)]],
            dtype=torch.int32,
        )[0, :]
        self.slice_end = ttnn.from_torch(
            end_tensor,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 0), mesh_shape=self.cluster_shape),
        )

    def _alloc_int_buffer(self, shard_dims=(None, 1), host=None):
        if host is None:
            host = torch.zeros((self.max_batch_size, self.vocab_size), dtype=torch.int32)
        return ttnn.from_torch(
            host,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=shard_dims, mesh_shape=self.cluster_shape),
        )

    def _alloc_bf16_buffer(self):
        host = torch.zeros((self.max_batch_size, 1), dtype=torch.float32)
        return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)

    def _copy_host_to_device(self, dst: ttnn.Tensor, src: torch.Tensor):
        src_tt = ttnn.from_torch(src, dtype=dst.dtype, layout=ttnn.TILE_LAYOUT, device=None)
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
        prompt_tokens = self._alloc_int_buffer(
            host=prompt_tokens.reshape(-1, prompt_tokens.shape[-1]), shard_dims=(None, None)
        )
        self.token_bin_counts_and_mask(new_tokens=prompt_tokens, mask=self.prompt_mask)

    def reset_output_tokens(self):
        self.output_mask = ttnn.mul(self.output_mask, 0, output_tensor=self.output_mask)
        self.output_counts = ttnn.mul(self.output_counts, 0, output_tensor=self.output_counts)
        self.output_counts_gathered = ttnn.mul(
            self.output_counts_gathered, 0, output_tensor=self.output_counts_gathered
        )

    def update_output_tokens(self, new_tokens):
        self.token_bin_counts_and_mask(
            new_tokens=new_tokens,
            counts=self.output_counts_gathered,
            counts_sliced=self.output_counts,
            mask=self.output_mask,
        )

    def token_bin_counts_and_mask(self, new_tokens, mask, counts=None, counts_sliced=None):
        # counts_new = ttnn.scatter_add(1, new_tokens)
        # fallback
        new_tokens = ttnn.to_torch(ttnn.get_device_tensors(new_tokens)[0]).to(torch.int64)
        counts_new = torch.zeros((self.max_batch_size, self.vocab_size), dtype=torch.int32)
        updates = torch.ones_like(new_tokens, dtype=torch.int32)
        counts_new = counts_new.scatter_add(dim=1, index=new_tokens, src=updates)
        counts_new = ttnn.from_torch(counts_new, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)

        if counts:
            counts = ttnn.add(counts, counts_new, output_tensor=counts, **self._op_kwargs)
        else:
            counts = counts_new
        counts_sliced = ttnn.slice(
            counts,
            self.slice_start,
            self.slice_end,
            output_tensor=counts_sliced,
            slice_dim=1,
            num_devices=self.cluster_shape[-1],
            **self._op_kwargs,
        )
        mask = ttnn.gt(counts_sliced, 0, output_tensor=mask, **self._op_kwargs)
        return counts, mask

    def apply(self, tt_logits: ttnn.Tensor) -> ttnn.Tensor:
        if tt_logits is None:
            return tt_logits

        context = PenaltyContext(
            prompt_mask=self.prompt_mask,
            output_mask=self.output_mask,
            output_counts=self.output_counts,
            output_counts_gathered=self.output_counts_gathered,
            presence_penalties=self.presence_penalties,
            frequency_penalties=self.frequency_penalties,
            repetition_penalties=self.repetition_penalties,
            sub_core_grids=self.sub_core_grids,
        )
        tt_logits = ttnn.typecast(tt_logits, ttnn.bfloat16)
        tt_logits = ttnn.pad(
            tt_logits, [(0, 0), (0, 0), (0, 0), (0, self.vocab_size // self.cluster_shape[-1] - tt_logits.shape[-1])], 0
        )

        reshaped = ttnn.reshape(tt_logits, (-1, tt_logits.shape[-1]))

        apply_penalties(reshaped, context)
        return ttnn.reshape(reshaped, (1, 1, -1, tt_logits.shape[-1]))
