# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
On-device allowed-token mask for constrained decoding.

Provides a persistent device buffer that masks logits so only allowed tokens
can be sampled. Updated between decode steps via copy_host_to_device_tensor()
without invalidating TTNN traces.
"""

from __future__ import annotations

import sys
from typing import List, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TTAllowedTokensMask(LightweightModule):
    """
    Persistent mask buffer for constrained decoding.

    Shape: [total_batch, vocab_size], dtype int32.
    Values: 1 = allowed, 0 = blocked.
    When inactive (no constraint), apply() is a no-op.
    """

    def __init__(self, mesh_device, args):
        super().__init__()
        self.mesh_device = mesh_device
        self.cluster_shape = mesh_device.shape
        self.max_batch_size = max(getattr(args, "max_batch_size", 32), 32)

        padded_vocab_size = getattr(args, "padded_vocab_size", None)
        self.vocab_size = padded_vocab_size if padded_vocab_size is not None else args.vocab_size
        num_devices = max(mesh_device.shape[-1], mesh_device.shape[-2])
        self.num_devices = num_devices

        self.sub_core_grids = getattr(args, "sub_core_grids", None)
        self._op_kwargs = {"sub_core_grids": self.sub_core_grids} if self.sub_core_grids else {}

        self._sampling_dp = getattr(args, "sampling_dp", 1)
        self._total_batch = self.max_batch_size * self._sampling_dp

        # Determine shard dims — same pattern as TTPenalties
        if mesh_device.shape[-1] == num_devices:
            shard_dims = (None, 1)
        else:
            shard_dims = (1, None)

        if self._sampling_dp > 1:
            shard_dims = (0, 1)

        self._shard_dims = shard_dims

        # Allocate persistent mask buffer — initialized to all-ones (unconstrained)
        host = torch.ones((self._total_batch, self.vocab_size), dtype=torch.int32)
        self.mask_tensor = ttnn.from_torch(
            host,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=shard_dims, mesh_shape=self.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self._active = False

    def reset_mask(self, allowed_token_ids: Optional[List[List[int]]]):
        """Update the mask buffer.

        Args:
            allowed_token_ids: Per-batch-entry list of allowed token IDs,
                or None for unconstrained (all tokens allowed).
        """
        if allowed_token_ids is None:
            if self._active:
                # Transition to unconstrained — fill with ones
                host = torch.ones((self._total_batch, self.vocab_size), dtype=torch.int32)
                self._copy_host_to_device(host)
            self._active = False
            return

        # Build dense mask: 1 at allowed positions, 0 elsewhere
        host = torch.zeros((self._total_batch, self.vocab_size), dtype=torch.int32)
        for i, token_ids in enumerate(allowed_token_ids):
            if i >= self._total_batch:
                break
            if token_ids is not None:
                valid_ids = [t for t in token_ids if 0 <= t < self.vocab_size]
                if valid_ids:
                    host[i, valid_ids] = 1
                else:
                    # No valid tokens — allow all to avoid blocking
                    host[i, :] = 1
            else:
                # None entry means unconstrained for this batch slot
                host[i, :] = 1

        # Pad unused batch slots with all-ones (unconstrained)
        if len(allowed_token_ids) < self._total_batch:
            host[len(allowed_token_ids) :, :] = 1

        self._copy_host_to_device(host)
        self._active = True

    def _copy_host_to_device(self, host: torch.Tensor):
        """Copy host tensor to persistent device buffer.

        Must use the same shard dims as the allocation so each device shard
        receives the correct slice of the vocab dimension.
        """
        mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=self._shard_dims, mesh_shape=self.cluster_shape)
        src_tt = ttnn.from_torch(host, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=mapper)
        ttnn.copy_host_to_device_tensor(src_tt, self.mask_tensor)

    def apply(self, tt_logits: ttnn.Tensor) -> ttnn.Tensor:
        """Apply the allowed-token mask to logits.

        Blocked tokens get -sys.float_info.max; allowed tokens are unchanged.
        Returns logits unmodified when no constraint is active.
        """
        if not self._active or tt_logits is None:
            return tt_logits

        original_shape = tt_logits.shape
        reshaped = ttnn.reshape(tt_logits, (-1, original_shape[-1]))

        # mask_tensor is already sharded across devices (vocab dim), matching logits layout
        mask_bf16 = ttnn.typecast(self.mask_tensor, ttnn.bfloat16, **self._op_kwargs)
        reshaped = ttnn.where(mask_bf16, reshaped, -sys.float_info.max, **self._op_kwargs)
        mask_bf16.deallocate()

        return ttnn.reshape(reshaped, original_shape)
