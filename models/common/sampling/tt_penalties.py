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
    inverse_repetition_penalties: ttnn.Tensor
    sub_core_grids: Any | None = None


def apply_penalties(logits: ttnn.Tensor, context: Optional[PenaltyContext]) -> ttnn.Tensor:
    if context is None:
        return logits

    op_kwargs = {"sub_core_grids": context.sub_core_grids} if context.sub_core_grids else {}
    # presence
    presence_term = ttnn.multiply(
        ttnn.typecast(context.output_mask, ttnn.bfloat16, **op_kwargs), context.presence_penalties, **op_kwargs
    )
    presence_term_bf16 = ttnn.typecast(presence_term, ttnn.bfloat16, **op_kwargs)
    logits = ttnn.subtract(logits, presence_term_bf16, output_tensor=logits, **op_kwargs)
    presence_term_bf16.deallocate()

    # frequency
    output_counts_bf16 = ttnn.typecast(context.output_counts, ttnn.bfloat16, **op_kwargs)

    freq_term = ttnn.multiply(output_counts_bf16, context.frequency_penalties, **op_kwargs)

    freq_term_bf16 = ttnn.typecast(freq_term, ttnn.bfloat16, **op_kwargs)
    logits = ttnn.subtract(logits, freq_term_bf16, output_tensor=logits, **op_kwargs)
    freq_term_bf16.deallocate()

    # repetition

    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.

    combined_mask_int32 = ttnn.add(context.prompt_mask, context.output_mask, **op_kwargs)
    combined_mask = ttnn.typecast(combined_mask_int32, ttnn.bfloat16, **op_kwargs)
    combined_mask_int32.deallocate()
    penalties = ttnn.where(combined_mask, context.repetition_penalties, 1.0, **op_kwargs)
    inverse_penalties = ttnn.where(combined_mask, context.inverse_repetition_penalties, 1.0, **op_kwargs)
    combined_mask.deallocate()

    # If logits are >0, divide by penalty, otherwise multiply by penalty.
    logits_bf16 = ttnn.typecast(logits, ttnn.bfloat16, **op_kwargs)
    logits_gt1 = ttnn.gt(logits_bf16, 0, **op_kwargs)
    scaling = ttnn.where(logits_gt1, inverse_penalties, penalties, **op_kwargs)
    logits_gt1.deallocate()
    penalties.deallocate()
    inverse_penalties.deallocate()
    logits = ttnn.multiply(logits, scaling, output_tensor=logits, **op_kwargs)
    scaling.deallocate()

    return logits


class TTPenalties(LightweightModule):
    """
    Penalty module with persistent device tensors, similar to TTSampling.
    """

    def __init__(self, mesh_device, args):
        super().__init__()
        self.mesh_device = mesh_device
        self.cluster_shape = mesh_device.shape
        self.max_batch_size = 32  # max_batch_size -- penalties and sampling only run for padded batch size

        padded_vocab_size = getattr(args, "padded_vocab_size", None)
        self.vocab_size = padded_vocab_size if padded_vocab_size is not None else args.vocab_size
        num_devices = max(mesh_device.shape[-1], mesh_device.shape[-2])
        self.num_devices = num_devices
        self.needs_padding = False
        if self.vocab_size == args.vocab_size:
            # need to add at least one tile padding for the histogram to handle padded tokens
            tile_width = 32
            padding = tile_width - ((self.vocab_size // num_devices) % tile_width)
            self.vocab_size += padding * num_devices
            self.needs_padding = True
        self.sub_core_grids = getattr(args, "sub_core_grids", None)
        self._op_kwargs = {"sub_core_grids": self.sub_core_grids} if self.sub_core_grids else {}

        # shard vocab size over larger cluster dim
        if mesh_device.shape[-1] == self.num_devices:
            shard_dims = (None, 1)
            shard_dims_slice = (None, 0)
        else:
            shard_dims = (1, None)
            shard_dims_slice = (0, None)
        self.prompt_mask = self._alloc_int_buffer(shard_dims=shard_dims)
        self.output_mask = self._alloc_int_buffer(shard_dims=shard_dims)
        self.output_counts_gathered = self._alloc_int_buffer(shard_dims=(None, None))
        self.output_counts = self._alloc_int_buffer(shard_dims=shard_dims)
        self.decode_src = self._alloc_int_buffer(
            host=torch.ones(self.max_batch_size, 1), shard_dims=(None, None), layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.zeros = self._alloc_int_buffer(shard_dims=(None, None), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.presence_penalties = self._alloc_bf16_buffer()
        self.frequency_penalties = self._alloc_bf16_buffer()
        self.repetition_penalties = self._alloc_bf16_buffer()
        self.inverse_repetition_penalties = self._alloc_bf16_buffer()

        vocab_per_dev = self.vocab_size // self.num_devices
        d = torch.arange(self.num_devices, dtype=torch.int32)

        # [0, 0, 0, vocab_per_dev, 0, 2*vocab_per_dev, ...]
        start_1d = torch.empty(2 * self.num_devices, dtype=torch.int32)
        start_1d[0::2] = 0
        start_1d[1::2] = d * vocab_per_dev

        # [32, vocab_per_dev, 32, 2*vocab_per_dev, 32, 3*vocab_per_dev, ...]
        end_1d = torch.empty(2 * self.num_devices, dtype=torch.int32)
        end_1d[0::2] = self.max_batch_size  # 32, exclusive
        end_1d[1::2] = (d + 1) * vocab_per_dev  # exclusive

        self.slice_start = ttnn.from_torch(
            start_1d,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=shard_dims_slice, mesh_shape=self.cluster_shape),
        )
        self.slice_end = ttnn.from_torch(
            end_1d,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=shard_dims_slice, mesh_shape=self.cluster_shape),
        )

        print(self.slice_start, self.slice_end)

    def _alloc_int_buffer(self, shard_dims, host=None, layout=ttnn.TILE_LAYOUT):
        if host is None:
            host = torch.zeros((self.max_batch_size, self.vocab_size), dtype=torch.int32)
        return ttnn.from_torch(
            host,
            dtype=ttnn.int32,
            layout=layout,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=shard_dims, mesh_shape=self.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        inverse_repetition_tensor = 1 / repetition_tensor

        self._copy_host_to_device(self.presence_penalties, presence_tensor)
        self._copy_host_to_device(self.frequency_penalties, frequency_tensor)
        self._copy_host_to_device(self.repetition_penalties, repetition_tensor)
        self._copy_host_to_device(self.inverse_repetition_penalties, inverse_repetition_tensor)

    def _pad_params(self, values: List[float]) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.numel() < self.max_batch_size:
            pad_value = tensor[-1] if tensor.numel() > 0 else torch.tensor(0.0)
            pad = pad_value.repeat(self.max_batch_size - tensor.numel())
            tensor = torch.cat([tensor, pad])
        elif tensor.numel() > self.max_batch_size:
            tensor = tensor[: self.max_batch_size]
        return tensor.view(self.max_batch_size, 1)

    def reset_prompt_tokens(self, prompt_tokens: torch.Tensor):
        # replaces -1s in prompt_tokens with self.vocab_size - 1
        prompt_tokens = torch.where(prompt_tokens == -1, self.vocab_size - 1, prompt_tokens)
        prompt_tokens = self._alloc_int_buffer(
            host=prompt_tokens.reshape(-1, prompt_tokens.shape[-1]),
            shard_dims=(None, None),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        src = self._alloc_int_buffer(
            host=torch.ones(self.max_batch_size, prompt_tokens.shape[-1]),
            shard_dims=(None, None),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.token_bin_counts_and_mask(new_tokens=prompt_tokens, src=src, mask=self.prompt_mask)

    def reset_output_tokens(self, tokens):
        # replaces -1s in tokens with self.vocab_size - 1
        tokens = torch.where(tokens == -1, self.vocab_size - 1, tokens)
        tokens_tt = ttnn.from_torch(
            tokens.reshape(-1, tokens.shape[-1]),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.output_mask = ttnn.mul(self.output_mask, 0, output_tensor=self.output_mask, **self._op_kwargs)
        self.output_counts = ttnn.mul(self.output_counts, 0, output_tensor=self.output_counts, **self._op_kwargs)
        self.output_counts_gathered = ttnn.mul(
            self.output_counts_gathered, 0, output_tensor=self.output_counts_gathered, **self._op_kwargs
        )
        self.update_output_tokens(tokens_tt)

    def update_output_tokens(self, new_tokens):
        # reshape decode token
        if new_tokens.shape[-1] == 32 and new_tokens.shape[-2] == 1:
            new_tokens = ttnn.reshape(new_tokens, [32, 1], **self._op_kwargs)
            src = self.decode_src
        else:
            src = self._alloc_int_buffer(
                host=torch.ones(self.max_batch_size, new_tokens.shape[-1]),
                shard_dims=(None, None),
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        self.token_bin_counts_and_mask(
            new_tokens=new_tokens,
            counts=self.output_counts_gathered,
            src=src,
            counts_sliced=self.output_counts,
            mask=self.output_mask,
        )

    def token_bin_counts_and_mask(self, new_tokens, src, counts=None, mask=None, counts_sliced=None):
        print(f"scatter_add op_kwargs: {self._op_kwargs}")

        print("foo")
        counts_new = ttnn.scatter_add(self.zeros, 1, new_tokens, src, **self._op_kwargs)

        new_tokens.deallocate()
        # need to use use_low_perf because llama galaxy runs out of L1 otherwise
        counts_new = ttnn.tilize(
            counts_new, **self._op_kwargs, use_low_perf=True if self.sub_core_grids is not None else False
        )
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
            num_devices=self.num_devices,
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
            inverse_repetition_penalties=self.inverse_repetition_penalties,
            sub_core_grids=self.sub_core_grids,
        )
        if self.needs_padding:
            original_shape = tt_logits.shape[-1]
            tt_logits = ttnn.typecast(tt_logits, ttnn.bfloat16)
            tt_logits = ttnn.pad(
                tt_logits, [(0, 0), (0, 0), (0, 0), (0, self.vocab_size // self.num_devices - tt_logits.shape[-1])], 0
            )

        reshaped = ttnn.reshape(tt_logits, (-1, tt_logits.shape[-1]))

        apply_penalties(reshaped, context)

        reshaped = ttnn.reshape(reshaped, (1, 1, -1, tt_logits.shape[-1]))

        if self.needs_padding:
            reshaped = reshaped[:, :, :, :original_shape]

        return reshaped
