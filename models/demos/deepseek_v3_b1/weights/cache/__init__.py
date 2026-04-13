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
    CacheContext,
    Fingerprint,
    FusionGroupSpec,
    MeshMapperConfig,
    ReplicateMeshMapper,
    RegionSpec,
    Shard2dMeshMapper,
    ShardMeshMapper,
    SourceTensorSelection,
    TensorTarget,
)


def __getattr__(name: str):
    """Lazy export of ``create_overlapped_tensor`` to avoid import cycles."""
    if name == "create_overlapped_tensor":
        from models.demos.deepseek_v3_b1.weights.cache.fuse import create_overlapped_tensor as _cot

        return _cot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class CacheConfig:
    """Bundles a TensorCache (or EphemeralTensorCache) and its CacheContext for prepare functions."""

    cache: TensorCacheProtocol
    context: CacheContext

    @classmethod
    def ephemeral(cls, *, move_to_device: bool = True) -> CacheConfig:
        """Config with in-memory cache only (no disk); used when callers omit ``cache_config``."""
        return cls(
            cache=EphemeralTensorCache(move_to_device=move_to_device),
            context=CacheContext(
                schema_version=0,
                hf_model_id="ephemeral",
                hf_revision="ephemeral",
                mesh_shape=(1, 1),
            ),
        )


__all__ = [
    "ArtifactTarget",
    "CacheConfig",
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
    "OverlappedTensorSpec",
    "TensorCache",
    "TensorCacheProtocol",
    "TensorTarget",
    "create_overlapped_tensor",
]
