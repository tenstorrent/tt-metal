# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""SRAM hot-expert wrapper over :class:`TensorCache` for per-core L1 ``CompressedTensor``s.

Owns all DeepSeek-specific concerns for SRAM (per-core L1) hot-expert weights:
  - Cold pack (``CompressedTensor.from_torch`` / ``from_bspm`` with
    ``keep_packed_data=True``) followed by post-pack byte extraction and
    persistence under :class:`SramCompressedTensorTarget`.
  - Warm hydration via :meth:`CompressedTensor.from_packed_artifacts`,
    skipping the (expensive) BFP pack pipeline entirely.

The generic :class:`TensorCache` is kept free of these details; callers
use :func:`get_or_create_sram_compressed_expert` instead of calling
:meth:`TensorCache.get_or_create` directly (the latter explicitly
rejects :class:`SramCompressedTensorTarget`).

Cache strategy: lazy on first miss.  An expert / projection that is
never accepted into the SRAM trim band of *any* layer never lands on
disk; once it is accepted, its post-pack bytes are written to a
content-addressed object that subsequent runs (and other layers using
the same expert + layout) can hydrate in one ``ttnn.from_torch`` worth
of time per shard.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.weights.cache.fingerprint import compute_artifact_id
from models.demos.deepseek_v3_b1.weights.cache.types import (
    Fingerprint,
    MeshMapperConfig,
    ReplicateMeshMapper,
    Shard2dMeshMapper,
    ShardMeshMapper,
    SramCompressedTensorTarget,
)

if TYPE_CHECKING:
    from models.demos.deepseek_v3_b1.compressed_tensor.assigner import CompressedTensorAssigner
    from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import CompressedTensor
    from models.demos.deepseek_v3_b1.weights.cache.cache import TensorCacheProtocol


# ---------------------------------------------------------------------------
# Assigner fingerprint
# ---------------------------------------------------------------------------


def to_canonical_mesh_mapper(mc) -> MeshMapperConfig:
    """Convert a runtime ``ttnn.MeshMapperConfig`` (or ``None``) into a canonical dataclass form.

    The SRAM cache fingerprint stores mesh-placement strategy via the
    declarative dataclasses :class:`ReplicateMeshMapper`,
    :class:`ShardMeshMapper`, or :class:`Shard2dMeshMapper` (so JSON
    canonicalization is stable across processes).  Callers in
    ``weights.prepare`` build live ``ttnn.MeshMapperConfig`` instances; this
    helper bridges the two representations.
    """
    if mc is None:
        return ReplicateMeshMapper()
    placements = list(getattr(mc, "placements", []))
    if not placements:
        return ReplicateMeshMapper()
    if len(placements) == 1:
        p = placements[0]
        if isinstance(p, ttnn.PlacementShard):
            return ShardMeshMapper(dim=p.dim)
        return ReplicateMeshMapper()
    if len(placements) == 2:
        d0 = placements[0].dim if isinstance(placements[0], ttnn.PlacementShard) else None
        d1 = placements[1].dim if isinstance(placements[1], ttnn.PlacementShard) else None
        if d0 is None and d1 is None:
            return ReplicateMeshMapper()
        return Shard2dMeshMapper(dims=(d0, d1))
    raise ValueError(f"Unsupported mesh_mapper_config dimensionality: {len(placements)} placements")


def assigner_fingerprint(assigner: "CompressedTensorAssigner | None") -> str:
    """Return a stable digest of the assigner's externally-observable config.

    Two assigners that produce identical per-tile format choices for any
    weight tensor must hash to the same digest.  Currently
    :class:`CompressedTensorAssigner` is parametrized by ``metric``,
    ``threshold``, ``formats``, and ``bfp0_mae_threshold`` — all
    deterministically picked at construction time.
    """
    if assigner is None:
        return ""
    parts = [
        type(assigner).__name__,
        getattr(assigner, "metric", ""),
        f"{float(getattr(assigner, 'threshold', 0.0)):.6f}",
        ",".join(sorted(getattr(assigner, "formats", []) or [])),
        f"{float(getattr(assigner, 'bfp0_mae_threshold', 0.0)):.6f}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Wrapper entry point
# ---------------------------------------------------------------------------


def get_or_create_sram_compressed_expert(
    cache: "TensorCacheProtocol",
    fingerprint: Fingerprint,
    device,
    *,
    weight_provider: Callable[[], torch.Tensor],
    assigner: "CompressedTensorAssigner | None" = None,
    assignment: np.ndarray | None = None,
    memory_config,
    assignment_memory_config=None,
    per_core_allocation: bool = True,
    mesh_mapper_config=None,
    tile_hw: int = 32,
    min_shard_bytes: int = 0,
) -> "CompressedTensor":
    """Load or build a per-core L1 SRAM ``CompressedTensor`` for one (expert, projection).

    Cache hit
        Reads ``shards.bin`` + ``metadata.json`` from the CAS object,
        rebuilds the per-device geometry deterministically, and uploads
        via :meth:`CompressedTensor.from_packed_artifacts` — host-side
        cost is dominated by ``ttnn.from_torch``s (no BFP pack).

    Cache miss
        Runs the regular cold pack with ``keep_packed_data=True``,
        extracts the post-pack byte buffers via
        :meth:`CompressedTensor.extract_packed_artifacts`, persists them
        under :class:`SramCompressedTensorTarget`, and returns the
        already-uploaded ``CompressedTensor`` (no second build).

    Ephemeral cache (in-memory, no disk)
        Bypasses lookup and persistence entirely — same cost as the
        existing ``CompressedTensor.from_torch`` / ``from_bspm`` path.

    Args:
        cache: A :class:`TensorCache` (disk-backed) or
            :class:`EphemeralTensorCache` instance.
        fingerprint: ``target`` must be a
            :class:`SramCompressedTensorTarget`.
        device: TT device or mesh device.  Required (per-core L1 tensors
            cannot be host-staged).
        weight_provider: Lazy callable returning the (already-preprocessed)
            float weight tensor.  Skipped on cache hit.
        assigner: Mixed-precision assigner.  Mutually exclusive with
            ``assignment``.
        assignment: Pre-computed BSPM tile-format codes (mutually
            exclusive with ``assigner``).
        memory_config / assignment_memory_config / per_core_allocation /
        mesh_mapper_config / tile_hw / min_shard_bytes: Same semantics as
        :meth:`CompressedTensor.from_torch` (forwarded to both cold and
        warm code paths).

    Returns:
        Device-resident :class:`CompressedTensor`.
    """
    from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import CompressedTensor
    from models.demos.deepseek_v3_b1.weights.cache.cache import (
        AbsentCacheEntry,
        CorruptCacheEntry,
        PresentCacheEntry,
        TensorCache,
    )

    target = fingerprint.target
    if not isinstance(target, SramCompressedTensorTarget):
        raise TypeError(f"get_or_create_sram_compressed_expert requires SramCompressedTensorTarget, got {type(target)}")
    assert (assigner is None) != (
        assignment is None
    ), "Provide exactly one of assigner / assignment to get_or_create_sram_compressed_expert"
    assert (
        device is not None
    ), "SRAM hot-expert CompressedTensors require a non-None device (per-core L1 has no host stage)"

    is_disk_cache = isinstance(cache, TensorCache)
    # Ephemeral / in-memory cache: there is no warm hit path and nothing to
    # persist, so skip fingerprint computation and the wrapper bookkeeping
    # entirely — falling straight through to the cold build.  Keeps the
    # pre-cache fast-path byte-identical for unit tests that wire in
    # ``CacheConfig.ephemeral``.
    if not is_disk_cache:
        return _build_cold(
            weight_provider=weight_provider,
            assigner=assigner,
            assignment=assignment,
            device=device,
            memory_config=memory_config,
            assignment_memory_config=assignment_memory_config,
            per_core_allocation=per_core_allocation,
            mesh_mapper_config=mesh_mapper_config,
            tile_hw=tile_hw,
            min_shard_bytes=min_shard_bytes,
            keep_packed_data=False,
        )

    artifact_id = compute_artifact_id(fingerprint)
    logical = target.name or artifact_id[:12]

    # ----- Warm path (disk hit) ----------------------------------------
    if is_disk_cache:
        entry = cache._lookup_sram_compressed(artifact_id)
        if isinstance(entry, PresentCacheEntry):
            t0 = time.perf_counter()
            artifacts = cache._load_sram_compressed(entry.paths.object_dir)
            ct = CompressedTensor.from_packed_artifacts(
                shape=artifacts["shape"],
                assignment_flat=artifacts["assignment_flat"],
                per_device_shard_data={c: dev["shard_bytes"] for c, dev in artifacts["per_device"].items()},
                per_device_assignment_flat={c: dev["assignment_flat"] for c, dev in artifacts["per_device"].items()},
                per_device_shape=artifacts["per_device_shape"],
                device=device,
                memory_config=memory_config,
                assignment_memory_config=assignment_memory_config,
                per_core_allocation=per_core_allocation,
                mesh_mapper_config=mesh_mapper_config,
                tile_hw=tile_hw,
                min_shard_bytes=min_shard_bytes,
            )
            logger.debug(
                "SRAM cache hit for {} ({}) — hydrated in {:.3f}s",
                logical,
                artifact_id[:12],
                time.perf_counter() - t0,
            )
            return ct
        if isinstance(entry, CorruptCacheEntry):
            logger.warning(
                "Corrupt SRAM cache entry for {} ({}); rebuilding from cold pack",
                logical,
                artifact_id[:12],
            )
            import shutil

            shutil.rmtree(entry.paths.object_dir, ignore_errors=True)
        else:
            assert isinstance(entry, AbsentCacheEntry)

    # ----- Cold path (build + persist) ---------------------------------
    t0 = time.perf_counter()
    ct = _build_cold(
        weight_provider=weight_provider,
        assigner=assigner,
        assignment=assignment,
        device=device,
        memory_config=memory_config,
        assignment_memory_config=assignment_memory_config,
        per_core_allocation=per_core_allocation,
        mesh_mapper_config=mesh_mapper_config,
        tile_hw=tile_hw,
        min_shard_bytes=min_shard_bytes,
        keep_packed_data=True,
    )
    cold_elapsed = time.perf_counter() - t0

    artifacts = ct.extract_packed_artifacts(drop=True)
    cache._store_sram_compressed(artifact_id, fingerprint, artifacts)
    logger.info(
        "SRAM cache miss for {} ({}) — built+persisted in {:.3f}s",
        logical,
        artifact_id[:12],
        cold_elapsed,
    )
    return ct


def _build_cold(
    *,
    weight_provider: Callable[[], torch.Tensor],
    assigner: "CompressedTensorAssigner | None",
    assignment: np.ndarray | None,
    device,
    memory_config,
    assignment_memory_config,
    per_core_allocation: bool,
    mesh_mapper_config,
    tile_hw: int,
    min_shard_bytes: int,
    keep_packed_data: bool,
) -> "CompressedTensor":
    """Run the cold pack path (assigner or pre-computed BSPM).

    Centralises the layout-kwarg routing so the wrapper's ephemeral
    fast-path and the cache-miss path share one construction site.
    """
    from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import CompressedTensor

    weight = weight_provider()
    if assigner is not None:
        return CompressedTensor.from_torch(
            weight.float(),
            assigner,
            device=device,
            memory_config=memory_config,
            assignment_memory_config=assignment_memory_config,
            per_core_allocation=per_core_allocation,
            mesh_mapper_config=mesh_mapper_config,
            keep_packed_data=keep_packed_data,
        )
    # Pre-computed BSPM assignment: ``from_bspm`` doesn't expose the layout
    # kwargs (legacy TP=1 single-device path).  Drop down to ``__init__``
    # whenever per-core / mesh layout matters so SRAM BSPM mode lands on
    # the same allocator semantics as the assigner mode.
    if mesh_mapper_config is not None or per_core_allocation:
        return CompressedTensor(
            weight.float(),
            assignment,
            device=device,
            memory_config=memory_config,
            assignment_memory_config=assignment_memory_config,
            tile_hw=tile_hw,
            min_shard_bytes=min_shard_bytes,
            per_core_allocation=per_core_allocation,
            mesh_mapper_config=mesh_mapper_config,
            keep_packed_data=keep_packed_data,
        )
    return CompressedTensor.from_bspm(
        weight.float(),
        assignment,
        device=device,
        memory_config=memory_config,
        assignment_memory_config=assignment_memory_config,
        tile_hw=tile_hw,
        min_shard_bytes=min_shard_bytes,
        keep_packed_data=keep_packed_data,
    )
