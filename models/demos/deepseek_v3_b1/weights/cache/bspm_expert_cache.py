# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""BSPM-specific wrapper over the generic TensorCache.

Owns all DeepSeek-specific concerns for BSPM expert weights:
  - DRAM shuffle (tile reordering for WIDTH_SHARDED streaming matmul)
  - Expert DRAM MemoryConfig construction
  - CompressedTensor.from_bspm device upload

The generic :class:`~cache.TensorCache` / :class:`~cache.EphemeralTensorCache`
are kept free of these details; callers use :func:`get_or_create_bspm_expert`
instead of calling :meth:`get_or_create` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

import ttnn
from models.demos.deepseek_v3_b1.weights.cache.types import (
    CompressedTensorBuildInputs,
    CompressedTensorTarget,
    Fingerprint,
)

if TYPE_CHECKING:
    from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import CompressedTensor
    from models.demos.deepseek_v3_b1.weights.cache.cache import TensorCacheProtocol


def expert_dram_memory_config(device, K: int, N_padded: int, num_banks: int) -> ttnn.MemoryConfig:
    """Build the WIDTH_SHARDED DRAM MemoryConfig for a routed expert projection."""
    per_core_N = N_padded // num_banks
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
            )
        }
    )
    shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def get_or_create_bspm_expert(
    cache: "TensorCacheProtocol",
    fingerprint: Fingerprint,
    device,
    *,
    raw_tensors: Callable[[], dict[str, torch.Tensor]] | dict[str, torch.Tensor],
    preprocess: Callable[[dict[str, torch.Tensor]], CompressedTensorBuildInputs],
    move_to_device: bool = True,
) -> "CompressedTensor":
    """Load or build a BSPM-encoded routed expert projection.

    Wraps :meth:`TensorCache.get_or_create` (or :class:`EphemeralTensorCache`) with
    BSPM-specific logic:

    1. **Shuffle** — the caller's ``preprocess`` returns logical-order
       ``CompressedTensorBuildInputs``; this wrapper applies DRAM tile shuffle before
       handing data to the generic cache so stored bytes are already in kernel-ready order.
    2. **MemoryConfig** — ``expert_dram_memory_config`` is called here, not in the cache.
    3. **CT construction** — ``CompressedTensor.from_bspm`` is called via the
       ``reconstruct`` callback, keeping the generic cache free of CT imports.

    Args:
        cache: A :class:`TensorCache` or :class:`EphemeralTensorCache` instance.
        fingerprint: Must have a :class:`CompressedTensorTarget` as its ``target``.
        device: TT device (or mesh device).
        raw_tensors: Lazy HF weight loader; skipped on cache hit.
        preprocess: Converts raw HF tensors to logical-order
            ``CompressedTensorBuildInputs(w, assignment)`` for this expert.
        move_to_device: Whether to upload the CompressedTensor to device.

    Returns:
        A device-resident (or host-only if ``move_to_device=False``) CompressedTensor.
    """
    from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import CompressedTensor
    from models.demos.deepseek_v3_b1.weights.transforms.moe import shuffle_dram_assignment, shuffle_dram_tiles

    target = fingerprint.target
    if not isinstance(target, CompressedTensorTarget):
        raise TypeError(f"get_or_create_bspm_expert requires CompressedTensorTarget, got {type(target)}")
    num_banks = target.num_banks

    def _preprocess_and_shuffle(tensors: dict) -> dict:
        inputs = preprocess(tensors)
        w_torch = torch.from_numpy(inputs.w).unsqueeze(0)
        w_shuffled = shuffle_dram_tiles(w_torch, tile_size=32, num_banks=num_banks).squeeze(0)
        assignment_shuffled = shuffle_dram_assignment(inputs.assignment, num_banks)
        return {target.name: CompressedTensorBuildInputs(w=w_shuffled.numpy(), assignment=assignment_shuffled)}

    def _reconstruct(inputs: CompressedTensorBuildInputs, dev) -> CompressedTensor:
        mem_config = expert_dram_memory_config(dev, target.K, target.N_padded, num_banks)
        device_for_ct = dev if move_to_device else None
        return CompressedTensor.from_bspm(
            torch.from_numpy(inputs.w).float(),
            inputs.assignment,
            device=device_for_ct,
            memory_config=mem_config,
        )

    return cache.get_or_create(
        fingerprint,
        device,
        preprocess=_preprocess_and_shuffle,
        raw_tensors=raw_tensors,
        reconstruct=_reconstruct,
    )
