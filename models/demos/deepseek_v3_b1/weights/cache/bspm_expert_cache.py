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
    ShardMeshMapper,
    TensorTarget,
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
        # The cache passes ``dev=None`` to signal host-stage, but CT needs the real
        # device for mesh-shape / mapper metadata regardless. Use the closure ``device``
        # for that, and ``move_to_device`` from the closure to control the upload.
        mem_config = expert_dram_memory_config(device, target.K, target.N_padded, num_banks)
        return CompressedTensor.from_bspm(
            torch.from_numpy(inputs.w).float(),
            inputs.assignment,
            device=device,
            memory_config=mem_config,
            move_to_device=move_to_device,
        )

    return cache.get_or_create(
        fingerprint,
        device,
        move_to_device=move_to_device,
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

    Caches the post-pack ``ct.data`` ttnn.Tensor (device-padded layout) via the
    standard :class:`TensorTarget` cache path — same hot path as the baseline
    uniform tensor cache: ``ttnn.dump_tensor`` on cold write, ``ttnn.load_tensor``
    on warm hit, no numpy / torch repack on the warm path.

    The caller-supplied ``fingerprint`` (whose target carries the BSPM
    uniquifying fields) is converted to an internal :class:`TensorTarget`
    fingerprint for the cache; the on-disk identity changes but the user-facing
    fingerprint stays stable.

    Cold path: builds the CT host-staged via :meth:`CompressedTensor.from_bspm`
    inside the cache's ``preprocess`` callback, then uses
    :func:`ttnn.to_torch` (with a matching mesh composer) to hand the combined
    torch tensor to the cache.  The cache calls ``ttnn.from_torch`` + dumps the
    result — exactly the baseline flow.

    Warm path: cache loads the ttnn.Tensor via ``ttnn.load_tensor`` and we wrap
    it in a :class:`CompressedTensor` via :meth:`CompressedTensor.from_dumped_data`.

    Ephemeral cache: bypasses the disk lookup entirely; builds cold and returns.
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
    mem_config = expert_dram_memory_config(device, K_per_device, N_padded_per_device, num_banks)
    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)])

    # Compute the BYTE-level memory_config for the packed CT data.  The on-device
    # buffer is WIDTH_SHARDED DRAM with shard_shape=[1, max_shard_bytes] (one
    # height row × N bytes per DRAM bank), where ``max_shard_bytes`` is the
    # packed BFP4 size of one (K_per_device × N_padded_per_device/num_banks)
    # block.  Assumes uniform BFP4 (production: ``bspm_dir=None`` or fallback);
    # mixed-precision BSPM would need to derive ``max_shard_bytes`` per CT from
    # the assignment, not done here.
    from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import bfp_tile_packed_size

    _BFP4_MANT_BITS = 3
    _bfp4_tile_bytes = bfp_tile_packed_size(_BFP4_MANT_BITS)  # 576
    _K_tiles = K_per_device // 32
    _per_core_N_tiles = (N_padded_per_device // num_banks) // 32
    max_shard_bytes_per_device = _K_tiles * _per_core_N_tiles * _bfp4_tile_bytes
    byte_shard_spec = ttnn.ShardSpec(
        mem_config.shard_spec.grid,
        [1, max_shard_bytes_per_device],
        mem_config.shard_spec.orientation,
    )
    byte_mem_config = ttnn.MemoryConfig(
        mem_config.memory_layout,
        mem_config.buffer_type,
        byte_shard_spec,
    )

    # Build a TensorTarget fingerprint that drives the standard cache flow.
    # All BSPM uniquifying fields are folded into the name so the artifact_id
    # depends on them.  ``transform_version`` offset avoids any collision with
    # historical TensorTarget version bumps.
    tensor_target_name = (
        f"{target.name}_tp8_dumped_v{target.bspm_variant}_b{target.bspm_budget:.2f}_h{target.assignment_hash}"
    )
    tensor_target = TensorTarget(
        name=tensor_target_name,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=byte_mem_config,
        mesh_mapper_config=ShardMeshMapper(dim=0),
        transform_version=target.transform_version + 100,
    )
    tensor_fingerprint = Fingerprint(
        schema_version=fingerprint.schema_version,
        source=fingerprint.source,
        hf_model_id=fingerprint.hf_model_id,
        hf_revision=fingerprint.hf_revision,
        mesh_shape=fingerprint.mesh_shape,
        target=tensor_target,
    )

    def _build_host_ct() -> "CompressedTensor":
        """Run the slice/shuffle/pack pipeline host-staged; returns CT with host data."""
        tensors = raw_tensors() if callable(raw_tensors) else raw_tensors
        inputs = preprocess(tensors)
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
        return CompressedTensor.from_bspm(
            stacked,
            stacked_assignment,
            device=device,
            memory_config=mem_config,
            mesh_mapper_config=mesh_mapper_config,
            move_to_device=False,  # always host-staged so we can convert to torch for the cache
        )

    # Hold a ref to the host CT (built once during preprocess) so we can grab
    # its assignment for the warm wrap.  ``cache.get_or_create`` only calls
    # ``preprocess`` on cache miss; on hit we re-derive assignment from the
    # source (uniform-BFP4 production case).
    host_ct_holder: dict = {}

    def _preprocess_to_torch(tensors: dict) -> dict:
        host_ct = _build_host_ct()
        host_ct_holder["ct"] = host_ct
        # Convert the host mesh-distributed ttnn.Tensor back to a combined torch
        # tensor that ``ttnn.from_torch(..., mesh_mapper=ShardTensorToMesh(dim=0))``
        # will re-shard the same way at cache-write time.
        combined_torch = ttnn.to_torch(
            host_ct.data,
            mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0),
        )
        return {tensor_target.name: combined_torch}

    data_tensor = cache.get_or_create(
        tensor_fingerprint,
        device,
        move_to_device=move_to_device,
        preprocess=_preprocess_to_torch,
        raw_tensors=raw_tensors,
    )

    host_ct = host_ct_holder.get("ct")
    if host_ct is not None:
        # Cache miss path: we built the CT, reuse its assignment_flat / shape.
        assignment_flat = host_ct._assignment_flat
        full_shape = host_ct.shape
        per_device_shape = host_ct._per_device_shape
        tile_hw = host_ct.tile_hw
    else:
        # Cache hit path: re-derive minimal metadata from known dims + target.
        # Uniform BFP4 production assumption: assignment is all 1s (bfp4 index).
        # TP8 stacking always produces per-device chunks of (K_per_device,
        # N_padded_per_device) regardless of ``shard_dim`` — caller already
        # baked the sharding into the K_per_device / N_padded_per_device kwargs.
        tile_hw = 32
        per_device_shape = (K_per_device, N_padded_per_device)
        full_shape = (tp * K_per_device, N_padded_per_device)
        tiles_h_full = full_shape[0] // tile_hw
        tiles_w_full = full_shape[1] // tile_hw
        assignment_flat = np.ones((tiles_h_full, tiles_w_full), dtype=np.int8).ravel()

    return CompressedTensor.from_dumped_data(
        data_tensor,
        shape=full_shape,
        assignment_flat=assignment_flat,
        per_device_shape=per_device_shape,
        device=device,
        memory_config=mem_config,
        tile_hw=tile_hw,
        mesh_mapper_config=mesh_mapper_config,
    )
