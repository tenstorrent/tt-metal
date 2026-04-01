# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint canonicalization and artifact ID computation.

Deterministic JSON serialization of a Fingerprint, then SHA-256 hash to produce
the content-addressed artifact ID.
"""

from __future__ import annotations

import hashlib
import json

import ttnn
from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlappedTensorSpec
from models.demos.deepseek_v3_b1.tensor_cache.types import (
    Fingerprint,
    FusionGroupSpec,
    RegionSpec,
    Shard2dMeshMapper,
    ShardMeshMapper,
    TensorTarget,
)


def _canonical_mesh_mapper(mapper_config) -> dict:
    """Serialize a MeshMapperConfig variant to a dict compatible with existing fingerprint hashes."""
    if isinstance(mapper_config, ShardMeshMapper):
        return {"strategy": "shard", "dim": mapper_config.dim, "dims": None}
    if isinstance(mapper_config, Shard2dMeshMapper):
        return {"strategy": "shard_2d", "dim": None, "dims": list(mapper_config.dims)}
    return {"strategy": "replicate", "dim": None, "dims": None}


def _canonical_core_range_set(crs: ttnn.CoreRangeSet) -> list:
    """Deterministic JSON-serializable representation of CoreRangeSet."""
    ranges = list(crs.ranges())
    ranges.sort(key=lambda r: (r.start.x, r.start.y, r.end.x, r.end.y))
    out = []
    for r in ranges:
        out.append([[r.start.x, r.start.y], [r.end.x, r.end.y]])
    return out


def _canonical_subtensor(st: OverlappedTensorSpec) -> dict:
    return {
        "name": st.name,
        "tensor_shape": list(st.raw_tensor_shape),
        "dtype": st.dtype.name,
        "tile_shape": [st.tile_h, st.tile_w],
    }


def _canonical_region(region: RegionSpec) -> dict:
    return {
        "core_range_set": _canonical_core_range_set(region.core_range_set),
        "subtensors": [_canonical_subtensor(st) for st in region.subtensors],
    }


def _canonical_fusion_group(target: FusionGroupSpec) -> dict:
    return {
        "kind": "fusion_group",
        "name": target.name,
        "regions": [_canonical_region(r) for r in target.regions],
        "sharding_strategy": target.sharding_strategy.name,
        "mesh_mapper_config": _canonical_mesh_mapper(target.mesh_mapper_config),
    }


def canonical(fingerprint: Fingerprint) -> dict:
    """Produce a deterministic, JSON-serializable dict from a Fingerprint."""
    target = fingerprint.target
    if isinstance(target, TensorTarget):
        target_dict = {
            "kind": "tensor",
            "name": target.name,
            "dtype": target.dtype.name,
            "layout": target.layout.name,
            "memory_config": json.loads(target.memory_config.to_json()),
            "tile_shape": list(target.tile_shape),
            "mesh_mapper_config": _canonical_mesh_mapper(target.mesh_mapper_config),
        }
    elif isinstance(target, FusionGroupSpec):
        target_dict = _canonical_fusion_group(target)
    else:
        raise TypeError(f"Unsupported target type: {type(target)}")
    return {
        "schema_version": fingerprint.schema_version,
        "source": sorted(fingerprint.source.names),
        "hf_model_id": fingerprint.hf_model_id,
        "hf_revision": fingerprint.hf_revision,
        "transform_version": fingerprint.transform_version,
        "mesh_shape": list(fingerprint.mesh_shape),
        "target": target_dict,
    }


def compute_artifact_id(fingerprint: Fingerprint) -> str:
    """SHA-256 hex digest of the canonical JSON representation."""
    blob = json.dumps(canonical(fingerprint), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
