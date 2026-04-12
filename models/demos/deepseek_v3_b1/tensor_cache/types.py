# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed tensor cache data model.

Frozen dataclasses for fingerprinting and tensor target specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

import ttnn


@dataclass(frozen=True)
class SourceTensorSelection:
    """Which HF state dict tensors feed into this artifact."""

    names: tuple[str, ...]


@dataclass(frozen=True)
class ReplicateMeshMapper:
    """Replicate the tensor on every device in the mesh."""

    strategy: Literal["replicate"] = "replicate"


@dataclass(frozen=True)
class ShardMeshMapper:
    """Shard the tensor along a single dimension across the mesh."""

    dim: int
    strategy: Literal["shard"] = "shard"


@dataclass(frozen=True)
class Shard2dMeshMapper:
    """Shard the tensor along two dimensions across a 2D mesh."""

    dims: tuple[int | None, int | None]
    strategy: Literal["shard_2d"] = "shard_2d"


MeshMapperConfig = Union[ReplicateMeshMapper, ShardMeshMapper, Shard2dMeshMapper]


@dataclass(frozen=True)
class TensorTarget:
    """Complete specification for a single (non-fused) cached tensor artifact.

    Contains every parameter needed for ttnn.from_torch on the miss path.
    All fields participate in the fingerprint hash.
    """

    kind: Literal["tensor"] = "tensor"
    name: str = ""
    dtype: ttnn.DataType = ttnn.bfloat16
    layout: ttnn.Layout = ttnn.TILE_LAYOUT
    memory_config: ttnn.MemoryConfig = field(default_factory=lambda: ttnn.DRAM_MEMORY_CONFIG)
    tile_shape: tuple[int, int] = (32, 32)
    mesh_mapper_config: MeshMapperConfig = field(default_factory=ReplicateMeshMapper)


ArtifactTarget = TensorTarget  # Will become TensorTarget | FusionGroupSpec in Phase 2


@dataclass(frozen=True)
class Fingerprint:
    """Cache key. All fields are known before any computation runs."""

    schema_version: int
    source: SourceTensorSelection
    hf_model_id: str
    hf_revision: str
    transform_version: int
    mesh_shape: tuple[int, int]
    target: ArtifactTarget


@dataclass(frozen=True)
class CacheContext:
    """Bundles the common cache key fields shared across all tensors in a model.

    Prepare functions accept this to avoid repeating hf_model_id, hf_revision, etc.
    for every standalone tensor they cache.
    """

    schema_version: int
    hf_model_id: str
    hf_revision: str
    transform_version: int
    mesh_shape: tuple[int, int]

    def fingerprint(self, *, source: SourceTensorSelection, target: ArtifactTarget) -> Fingerprint:
        return Fingerprint(
            schema_version=self.schema_version,
            source=source,
            hf_model_id=self.hf_model_id,
            hf_revision=self.hf_revision,
            transform_version=self.transform_version,
            mesh_shape=self.mesh_shape,
            target=target,
        )
