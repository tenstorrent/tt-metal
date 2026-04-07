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
    MeshMapperConfig,
    ReplicateMeshMapper,
    Shard2dMeshMapper,
    ShardMeshMapper,
    SourceTensorSelection,
    TensorTarget,
)


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
    "MeshMapperConfig",
    "ReplicateMeshMapper",
    "Shard2dMeshMapper",
    "ShardMeshMapper",
    "SourceTensorSelection",
    "TensorCache",
    "TensorTarget",
]
