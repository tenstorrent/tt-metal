# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed tensor cache for preprocessed weight artifacts."""

from dataclasses import dataclass

from models.demos.deepseek_v3_b1.tensor_cache.cache import TensorCache
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
    """Bundles a TensorCache and its CacheContext for passing to prepare functions."""

    cache: TensorCache
    context: CacheContext


__all__ = [
    "ArtifactTarget",
    "CacheConfig",
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
