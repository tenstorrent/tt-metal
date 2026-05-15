# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""BSPM-specific wrapper over the generic TensorCache.

Owns all DeepSeek-specific concerns for BSPM expert weights:
  - DRAM shuffle (tile reordering for WIDTH_SHARDED streaming matmul)
  - TP8 mesh sharding (slice + per-rank shuffle + stack)
  - Expert DRAM MemoryConfig construction
  - CompressedTensor.from_bspm device upload

The generic :class:`~cache.TensorCache` / :class:`~cache.EphemeralTensorCache`
are kept free of these details; callers use :func:`get_or_create_bspm_expert`
(single-device) or :func:`get_or_create_bspm_expert_tp8` (2D mesh sharded)
instead of calling :meth:`get_or_create` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
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


def get_or_create_bspm_expert_tp8(
    cache: "TensorCacheProtocol",
    fingerprint: Fingerprint,
    device,
    *,
    raw_tensors: Callable[[], dict[str, torch.Tensor]] | dict[str, torch.Tensor],
    preprocess: Callable[[dict[str, torch.Tensor]], CompressedTensorBuildInputs],
    mesh_shape: tuple[int, int],
    shard_dim: int,
    K_per_device: int,
    N_padded_per_device: int,
    subblock_k: int | None = None,
    subblock_n: int = 1,
    move_to_device: bool = True,
) -> "CompressedTensor":
    """Load or build a BSPM-encoded routed expert projection sliced for a 2D TP mesh.

    The TP8 analogue of :func:`get_or_create_bspm_expert`. Caching unit is one
    ``(layer, expert, projection, mesh_shape)`` — :class:`Fingerprint` already
    captures ``mesh_shape``, so single-device and TP8 entries never collide.

    Pipeline:

    1. **Slice + shuffle (preprocess)** — caller's ``preprocess`` returns the
       *full logical* ``(K, N)`` weight + ``(tiles_h, tiles_w)`` assignment for
       this expert/projection.  This wrapper calls
       :func:`moe_routed_expert_bspm_tp8_torch_for_cache` to slice/shuffle/stack.
    2. **2D-flatten for storage** — the 4D stacked weight
       ``(mesh_rows, mesh_cols, K_per_device, N_padded_per_device)`` is flattened
       to ``(mesh_rows*mesh_cols*K_per_device, N_padded_per_device)`` so it fits
       the existing compact-tile cache layout.  The matching assignment is
       already in this row-stacked form.
    3. **Reconstruct** — at load time we receive the flat 2D weight, reshape
       back to 4D, build per-device :class:`MemoryConfig` from
       ``(K_per_device, N_padded_per_device)``, and call
       :meth:`CompressedTensor.from_bspm` with a 2D mesh mapper so each
       ``(mesh_row, mesh_col)`` slice lands on its device.

    Args:
        mesh_shape: ``(mesh_rows, mesh_cols)``.  Must match
            ``fingerprint.mesh_shape`` (asserted).
        shard_dim: ``0`` = row-parallel (down_proj), ``1`` = column-parallel
            (gate/up_proj).  Captured by the slicing helper, not in the
            fingerprint — projection ``name`` already differentiates the three
            projections within a layer.
        K_per_device, N_padded_per_device: per-rank dims used to build the
            DRAM :class:`MemoryConfig` at reconstruct time and to round-trip
            the 2D-stored weight back to 4D.
    """
    from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import CompressedTensor
    from models.demos.deepseek_v3_b1.weights.transforms.moe import moe_routed_expert_bspm_tp8_torch_for_cache

    target = fingerprint.target
    if not isinstance(target, CompressedTensorTarget):
        raise TypeError(f"get_or_create_bspm_expert_tp8 requires CompressedTensorTarget, got {type(target)}")

    mesh_rows, mesh_cols = mesh_shape
    if fingerprint.mesh_shape != mesh_shape:
        raise ValueError(f"fingerprint.mesh_shape={fingerprint.mesh_shape} disagrees with mesh_shape={mesh_shape}")
    tp = mesh_rows * mesh_cols
    expected_K_flat = tp * K_per_device
    if target.K != expected_K_flat:
        raise ValueError(
            f"target.K={target.K} must equal mesh_rows*mesh_cols*K_per_device={expected_K_flat} "
            f"for TP8 (mesh_shape={mesh_shape}, K_per_device={K_per_device})"
        )
    if target.N_padded != N_padded_per_device:
        raise ValueError(
            f"target.N_padded={target.N_padded} must equal N_padded_per_device={N_padded_per_device} for TP8"
        )
    num_banks = target.num_banks

    def _preprocess_and_slice(tensors: dict) -> dict:
        inputs = preprocess(tensors)
        # inputs.w: (K, N) logical pre-slice float32 numpy.
        # inputs.assignment: (tiles_h, tiles_w) int8 numpy (logical, pre-slice, pre-shuffle).
        w_torch = torch.from_numpy(inputs.w) if isinstance(inputs.w, np.ndarray) else inputs.w
        stacked, stacked_assignment = moe_routed_expert_bspm_tp8_torch_for_cache(
            w_torch,
            inputs.assignment,
            num_banks,
            mesh_shape,
            shard_dim,
            subblock_k=subblock_k,
            subblock_n=subblock_n,
        )
        # Flatten 4D (mesh_rows, mesh_cols, K_per_device, N_padded_per_device) to 2D for storage.
        # stacked_assignment is already (mesh_rows*mesh_cols*(K_per_device//32), N_padded_per_device//32).
        w_flat = stacked.reshape(tp * K_per_device, N_padded_per_device).contiguous().numpy()
        return {target.name: CompressedTensorBuildInputs(w=w_flat, assignment=stacked_assignment)}

    def _reconstruct(inputs: CompressedTensorBuildInputs, dev) -> CompressedTensor:
        # Cache hands us 2D-flat (K_total, N_padded_per_device); reshape back to 4D mesh layout.
        w_flat = torch.from_numpy(inputs.w).float()
        if w_flat.shape != (tp * K_per_device, N_padded_per_device):
            raise ValueError(
                f"Cached TP8 weight shape {tuple(w_flat.shape)} does not match expected "
                f"({tp * K_per_device}, {N_padded_per_device})"
            )
        stacked = w_flat.reshape(mesh_rows, mesh_cols, K_per_device, N_padded_per_device).contiguous()
        mem_config = expert_dram_memory_config(dev, K_per_device, N_padded_per_device, num_banks)
        device_for_ct = dev if move_to_device else None
        mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)])
        return CompressedTensor.from_bspm(
            stacked,
            inputs.assignment,
            device=device_for_ct,
            memory_config=mem_config,
            mesh_mapper_config=mesh_mapper_config,
        )

    return cache.get_or_create(
        fingerprint,
        device,
        preprocess=_preprocess_and_slice,
        raw_tensors=raw_tensors,
        reconstruct=_reconstruct,
    )
