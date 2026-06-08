# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed tensor cache for preprocessed weight artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from models.demos.deepseek_v3_b1.weights.cache.cache import EphemeralTensorCache, TensorCache, TensorCacheProtocol
from models.demos.deepseek_v3_b1.weights.overlap.spec import OverlappedTensorSpec
from models.demos.deepseek_v3_b1.weights.cache.types import (
    ArtifactTarget,
    BspmVariant,
    CacheContext,
    CompressedTensorBuildInputs,
    CompressedTensorTarget,
    Fingerprint,
    FusionGroupSpec,
    MeshMapperConfig,
    ReplicateMeshMapper,
    RegionSpec,
    Shard2dMeshMapper,
    ShardMeshMapper,
    SourceTensorSelection,
    SramCompressedTensorTarget,
    TensorTarget,
)


def create_overlapped_tensor(*args, **kwargs):
    """Lazy wrapper — defers heavy fuse imports until first call."""
    from models.demos.deepseek_v3_b1.weights.cache.fuse import create_overlapped_tensor as _cot

    return _cot(*args, **kwargs)


def get_or_create_bspm_expert(*args, **kwargs):
    """Lazy wrapper — defers bspm_expert_cache imports until first call."""
    from models.demos.deepseek_v3_b1.weights.cache.bspm_expert_cache import get_or_create_bspm_expert as _fn

    return _fn(*args, **kwargs)


def get_or_create_sram_compressed_expert(*args, **kwargs):
    """Lazy wrapper — defers sram_compressed_cache imports until first call."""
    from models.demos.deepseek_v3_b1.weights.cache.sram_compressed_cache import (
        get_or_create_sram_compressed_expert as _fn,
    )

    return _fn(*args, **kwargs)


def get_or_create_bspm_expert_tp8(*args, **kwargs):
    """Lazy wrapper — defers bspm_expert_cache imports until first call."""
    from models.demos.deepseek_v3_b1.weights.cache.bspm_expert_cache import get_or_create_bspm_expert_tp8 as _fn

    return _fn(*args, **kwargs)


def __getattr__(name: str):
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class CacheConfig:
    """Bundles a TensorCache (or EphemeralTensorCache) and its CacheContext for prepare functions."""

    cache: TensorCacheProtocol
    context: CacheContext

    @classmethod
    def ephemeral(cls, *, move_to_device: bool = True, mesh_shape: tuple[int, int] = (1, 1)) -> CacheConfig:
        """Config with in-memory cache only (no disk); used when callers omit ``cache_config``.

        ``mesh_shape`` must match the actual device mesh — the TP8 cache helpers
        (e.g. :func:`get_or_create_bspm_expert_tp8`) cross-check ``Fingerprint.mesh_shape``
        against the runtime device mesh, so leaving the default ``(1, 1)`` on a
        multi-device run will raise.
        """
        return cls(
            cache=EphemeralTensorCache(move_to_device=move_to_device),
            context=CacheContext(
                schema_version=0,
                hf_model_id="ephemeral",
                hf_revision="ephemeral",
                mesh_shape=mesh_shape,
            ),
        )


__all__ = [
    "ArtifactTarget",
    "BspmVariant",
    "CacheConfig",
    "CompressedTensorBuildInputs",
    "CompressedTensorTarget",
    "EphemeralTensorCache",
    "CacheContext",
    "Fingerprint",
    "FusionGroupSpec",
    "MeshMapperConfig",
    "ReplicateMeshMapper",
    "RegionSpec",
    "Shard2dMeshMapper",
    "ShardMeshMapper",
    "SourceTensorSelection",
    "SramCompressedTensorTarget",
    "OverlappedTensorSpec",
    "TensorCache",
    "TensorCacheProtocol",
    "TensorTarget",
    "create_overlapped_tensor",
    "get_or_create_bspm_expert",
    "get_or_create_bspm_expert_tp8",
    "get_or_create_sram_compressed_expert",
]
