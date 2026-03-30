# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed tensor cache data model.

Frozen dataclasses for fingerprinting, artifact metadata, and CAS storage paths.
See ARCHITECTURE.md for the full design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import ttnn


@dataclass(frozen=True)
class SourceTensorSelection:
    """Which HF state dict tensors feed into this artifact."""

    names: tuple[str, ...]


@dataclass(frozen=True)
class MeshMapperConfig:
    """Declarative mesh mapping strategy (serializable, device-independent).

    At runtime, TensorCache reconstructs the ttnn mesh_mapper object from this
    config and the target device.

    Strategies:
        "replicate"  -> ReplicateTensorToMesh(device)
        "shard"      -> ShardTensorToMesh(device, dim=dim)
        "shard_2d"   -> ShardTensor2dMesh(device, mesh_shape=device.shape, dims=dims)
    """

    strategy: Literal["replicate", "shard", "shard_2d"]
    dim: int | None = None  # for "shard"
    dims: tuple[int | None, int | None] | None = None  # for "shard_2d"


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
    mesh_mapper_config: MeshMapperConfig = field(default_factory=lambda: MeshMapperConfig("replicate"))


@dataclass(frozen=True)
class Fingerprint:
    """Cache key. All fields are known before any computation runs."""

    schema_version: int
    source: SourceTensorSelection
    hf_model_id: str
    hf_revision: str
    transform_version: int
    mesh_shape: tuple[int, int]
    target: TensorTarget


@dataclass(frozen=True)
class FingerprintContext:
    """Bundles the common fingerprint fields shared across all tensors in a model.

    Prepare functions accept this to avoid repeating hf_model_id, hf_revision, etc.
    for every standalone tensor they cache.
    """

    schema_version: int
    hf_model_id: str
    hf_revision: str
    transform_version: int
    mesh_shape: tuple[int, int]

    def fingerprint(self, *, source: SourceTensorSelection, target: TensorTarget) -> Fingerprint:
        return Fingerprint(
            schema_version=self.schema_version,
            source=source,
            hf_model_id=self.hf_model_id,
            hf_revision=self.hf_revision,
            transform_version=self.transform_version,
            mesh_shape=self.mesh_shape,
            target=target,
        )


@dataclass(frozen=True)
class Manifest:
    """Stored in manifest.json inside each artifact directory."""

    fingerprint: Fingerprint
    logical_name: str | None = None


@dataclass(frozen=True)
class Metadata:
    """Stored in metadata.json inside each artifact directory."""

    artifact_id: str
    content_hash: str
    size_bytes: int
    created_at: str


@dataclass(frozen=True)
class CasPaths:
    """Filesystem paths for one CAS object."""

    object_dir: Path
    data_path: Path


class CacheEntryState(Enum):
    ABSENT = "absent"
    PRESENT = "present"
    CORRUPT = "corrupt"


@dataclass(frozen=True)
class CacheEntry:
    """Result of a CAS lookup."""

    artifact_id: str
    state: CacheEntryState
    paths: CasPaths | None
