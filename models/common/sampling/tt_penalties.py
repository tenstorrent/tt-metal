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
        # Floor at 32 so that ROW_MAJOR [batch, vocab] buffers passed to
        # ttnn.tilize always have physical_volume divisible by TILE_HW
        # (32*32 = 1024).  32 * V is 1024-aligned for any 32-aligned V.
        self.max_batch_size = max(getattr(args, "max_batch_size", 32), 32)

        padded_vocab_size = getattr(args, "padded_vocab_size", None)
        self.vocab_size = padded_vocab_size if padded_vocab_size is not None else args.vocab_size
        num_devices = max(mesh_device.shape[-1], mesh_device.shape[-2])
        self.num_devices = num_devices

        self.sub_core_grids = getattr(args, "sub_core_grids", None)
        self._op_kwargs = {"sub_core_grids": self.sub_core_grids} if self.sub_core_grids else {}

        # sampling_dp > 1 when multiple mesh rows each sample independently
        # (e.g. GPT-OSS on [4,8] Galaxy: 4 rows × 32 users = 128 total)
        self._sampling_dp = getattr(args, "sampling_dp", 1)
        # Total batch across all rows. Host tensors use this size; after
        # (0, ...) sharding each row gets max_batch_size entries.
        self._total_batch = self.max_batch_size * self._sampling_dp

        # shard vocab size over larger cluster dim
        if mesh_device.shape[-1] == self.num_devices:
            shard_dims = (None, 1)
            shard_dims_slice = (None, 0)
        else:
            shard_dims = (1, None)
            shard_dims_slice = (0, None)

        # For row-sharded mode (sampling_dp > 1), also shard the batch dimension
        # across mesh rows so each row gets its own per-user penalty state.
        if self._sampling_dp > 1:
            assert (
                mesh_device.shape[-1] == self.num_devices
            ), "Row-sharded penalties require vocab sharding along mesh columns"
            shard_dims = (0, 1)  # batch across rows, vocab across cols
            shard_dims_gathered = (0, None)  # batch across rows, vocab replicated
            shard_dims_bf16 = (0, None)  # per-row penalty params
            per_row_batch = self.max_batch_size  # NOT divided: each row gets max_batch_size
        else:
            shard_dims_gathered = (None, None)
            shard_dims_bf16 = None
            per_row_batch = self.max_batch_size

        self.per_row_batch_size = per_row_batch
        self._shard_dims_gathered = shard_dims_gathered

        self.prompt_mask = self._alloc_int_buffer(shard_dims=shard_dims)
        self.output_mask = self._alloc_int_buffer(shard_dims=shard_dims)
        self.output_counts_gathered = self._alloc_int_buffer(shard_dims=shard_dims_gathered)
        self.output_counts = self._alloc_int_buffer(shard_dims=shard_dims)
        self.decode_src = self._alloc_int_buffer(
            host=torch.ones(self._total_batch, 1), shard_dims=shard_dims_gathered, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.zeros = self._alloc_int_buffer(shard_dims=shard_dims_gathered, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.presence_penalties = self._alloc_bf16_buffer(shard_dims=shard_dims_bf16)
        self.frequency_penalties = self._alloc_bf16_buffer(shard_dims=shard_dims_bf16)
        self.repetition_penalties = self._alloc_bf16_buffer(shard_dims=shard_dims_bf16)
        self.inverse_repetition_penalties = self._alloc_bf16_buffer(shard_dims=shard_dims_bf16)

        vocab_per_dev = self.vocab_size // self.num_devices
        d = torch.arange(self.num_devices, dtype=torch.int32)

        # [0, 0, 0, vocab_per_dev, 0, 2*vocab_per_dev, ...]
        start_1d = torch.empty(2 * self.num_devices, dtype=torch.int32)
        start_1d[0::2] = 0
        start_1d[1::2] = d * vocab_per_dev

        # [batch, vocab_per_dev, batch, 2*vocab_per_dev, ...]
        end_1d = torch.empty(2 * self.num_devices, dtype=torch.int32)
        end_1d[0::2] = per_row_batch  # per-row batch size, exclusive
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

    def _alloc_int_buffer(self, shard_dims, host=None, layout=ttnn.TILE_LAYOUT):
        if host is None:
            host = torch.zeros((self._total_batch, self.vocab_size), dtype=torch.int32)
        return ttnn.from_torch(
            host,
            dtype=ttnn.int32,
            layout=layout,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=shard_dims, mesh_shape=self.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _alloc_bf16_buffer(self, shard_dims=None):
        host = torch.zeros((self._total_batch, 1), dtype=torch.float32)
        if shard_dims is not None:
            return ttnn.from_torch(
                host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=shard_dims, mesh_shape=self.cluster_shape),
            )
        return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)

    def _copy_host_to_device(self, dst: ttnn.Tensor, src: torch.Tensor):
        if self._sampling_dp > 1:
            # For row-sharded buffers, create a properly sharded host tensor
            # so copy_host_to_device_tensor writes per-row shards correctly.
            mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=self.cluster_shape)
            src_tt = ttnn.from_torch(src, dtype=dst.dtype, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=mapper)
        else:
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
        if tensor.numel() < self._total_batch:
            pad_value = tensor[-1] if tensor.numel() > 0 else torch.tensor(0.0)
            pad = pad_value.repeat(self._total_batch - tensor.numel())
            tensor = torch.cat([tensor, pad])
        elif tensor.numel() > self._total_batch:
            tensor = tensor[: self._total_batch]
        return tensor.view(self._total_batch, 1)

    def _pad_batch_to_max(self, tokens_2d: torch.Tensor, pad_value: int) -> torch.Tensor:
        """Pad/truncate first dim to _total_batch."""
        if tokens_2d.dim() != 2:
            raise ValueError(f"Expected 2D tensor [B, S], got {tokens_2d.shape}")
        B, S = tokens_2d.shape
        if B < self._total_batch:
            pad = torch.full((self._total_batch - B, S), pad_value, dtype=tokens_2d.dtype)
            return torch.cat([tokens_2d, pad], dim=0)
        if B > self._total_batch:
            return tokens_2d[: self._total_batch]
        return tokens_2d

    def reset_prompt_tokens(self, prompt_tokens: torch.Tensor):
        # Mask out padding positions (-1) instead of inventing a fake token id by expanding vocab_size.
        prompt_tokens_2d = prompt_tokens.reshape(-1, prompt_tokens.shape[-1])
        prompt_tokens_2d = self._pad_batch_to_max(prompt_tokens_2d, pad_value=-1)

        src_host = (prompt_tokens_2d != -1).to(torch.int32)
        idx_host = torch.where(prompt_tokens_2d == -1, torch.zeros_like(prompt_tokens_2d), prompt_tokens_2d)

        prompt_tokens_tt = self._alloc_int_buffer(
            host=idx_host,
            shard_dims=self._shard_dims_gathered,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        src_tt = self._alloc_int_buffer(
            host=src_host,
            shard_dims=self._shard_dims_gathered,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.token_bin_counts_and_mask(new_tokens=prompt_tokens_tt, src=src_tt, mask=self.prompt_mask)

    def reset_output_tokens(self, tokens=None):
        self.output_mask = ttnn.mul(self.output_mask, 0, output_tensor=self.output_mask, **self._op_kwargs)
        self.output_counts = ttnn.mul(self.output_counts, 0, output_tensor=self.output_counts, **self._op_kwargs)
        self.output_counts_gathered = ttnn.mul(
            self.output_counts_gathered, 0, output_tensor=self.output_counts_gathered, **self._op_kwargs
        )
        if tokens is not None:
            # Mask out padding positions (-1) instead of inventing a fake token id by expanding vocab_size.
            tokens_2d = tokens.reshape(-1, tokens.shape[-1])
            tokens_2d = self._pad_batch_to_max(tokens_2d, pad_value=-1)
            src_host = (tokens_2d != -1).to(torch.int32)
            idx_host = torch.where(tokens_2d == -1, torch.zeros_like(tokens_2d), tokens_2d)

            mapper = (
                ttnn.ShardTensor2dMesh(self.mesh_device, dims=self._shard_dims_gathered, mesh_shape=self.cluster_shape)
                if self._sampling_dp > 1
                else None
            )
            tokens_tt = ttnn.from_torch(
                idx_host,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            src_tt = ttnn.from_torch(
                src_host,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            self.token_bin_counts_and_mask(
                new_tokens=tokens_tt,
                counts=self.output_counts_gathered,
                src=src_tt,
                counts_sliced=self.output_counts,
                mask=self.output_mask,
            )

    def update_output_tokens(self, new_tokens):
        # Reshape decode token to [batch, 1] for scatter_add.
        # Non-row-sharded: token shape is [1,1,1,batch] → shape[-1]==batch, shape[-2]==1
        # Row-sharded:     token shape is [1,1,batch,1] → shape[-2]==batch, shape[-1]==1
        batch = self.per_row_batch_size
        if (new_tokens.shape[-1] == batch and new_tokens.shape[-2] == 1) or (
            new_tokens.shape[-2] == batch and new_tokens.shape[-1] == 1
        ):
            new_tokens = ttnn.reshape(new_tokens, [batch, 1], **self._op_kwargs)
            src = self.decode_src
        else:
            src = self._alloc_int_buffer(
                host=torch.ones(self._total_batch, new_tokens.shape[-1]),
                shard_dims=self._shard_dims_gathered,
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
        original_shape = tt_logits.shape
        reshaped = ttnn.reshape(tt_logits, (-1, original_shape[-1]))
        apply_penalties(reshaped, context)
        return ttnn.reshape(reshaped, original_shape)
