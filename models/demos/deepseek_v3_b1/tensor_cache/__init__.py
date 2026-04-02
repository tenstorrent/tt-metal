# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed tensor cache for preprocessed weight artifacts."""

from dataclasses import dataclass

from models.demos.deepseek_v3_b1.tensor_cache.cache import EphemeralTensorCache, TensorCache
from models.demos.deepseek_v3_b1.tensor_cache.types import (
    ArtifactTarget,
    CacheContext,
    Fingerprint,
    FusionGroupSpec,
    MeshMapperConfig,
    OverlappedViewMeta,
    ReplicateMeshMapper,
    RegionSpec,
    Shard2dMeshMapper,
    ShardMeshMapper,
    SourceTensorSelection,
    SubTensorSpec,
    TensorTarget,
)


def __getattr__(name: str):
    """Lazy export of ``create_overlapped_tensor`` to avoid import cycles with ``blitz_decode_weights``."""
    if name == "create_overlapped_tensor":
        from models.demos.deepseek_v3_b1.tensor_cache.fuse import create_overlapped_tensor as _cot

        return _cot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class CacheConfig:
    """Bundles a TensorCache (or EphemeralTensorCache) and its CacheContext for prepare functions."""

    cache: TensorCache | EphemeralTensorCache
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
                transform_version=0,
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
    "OverlappedViewMeta",
    "ReplicateMeshMapper",
    "RegionSpec",
    "Shard2dMeshMapper",
    "ShardMeshMapper",
    "SourceTensorSelection",
    "SubTensorSpec",
    "TensorCache",
    "TensorTarget",
    "create_overlapped_tensor",
]
