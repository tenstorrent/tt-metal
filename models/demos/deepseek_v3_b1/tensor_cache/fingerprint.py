# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint canonicalization and artifact ID computation.

Deterministic JSON serialization of a Fingerprint, then SHA-256 hash to produce
the content-addressed artifact ID.
"""

from __future__ import annotations

import hashlib
import json

from models.demos.deepseek_v3_b1.tensor_cache.types import Fingerprint, Shard2dMeshMapper, ShardMeshMapper, TensorTarget


def _canonical_mesh_mapper(mapper_config) -> dict:
    """Serialize a MeshMapperConfig variant to a dict compatible with existing fingerprint hashes."""
    if isinstance(mapper_config, ShardMeshMapper):
        return {"strategy": "shard", "dim": mapper_config.dim, "dims": None}
    if isinstance(mapper_config, Shard2dMeshMapper):
        return {"strategy": "shard_2d", "dim": None, "dims": list(mapper_config.dims)}
    return {"strategy": "replicate", "dim": None, "dims": None}


def canonical(fingerprint: Fingerprint) -> dict:
    """Produce a deterministic, JSON-serializable dict from a Fingerprint.

    Only ArtifactTarget variants with kind='tensor' are supported; FusionGroupSpec will be added later.
    """
    target = fingerprint.target
    if not isinstance(target, TensorTarget):
        raise TypeError(f"Unsupported target type: {type(target)}")
    target_dict = {
        "kind": "tensor",
        "name": target.name,
        "dtype": target.dtype.name,
        "layout": target.layout.name,
        "memory_config": json.loads(target.memory_config.to_json()),
        "tile_shape": list(target.tile_shape),
        "mesh_mapper_config": _canonical_mesh_mapper(target.mesh_mapper_config),
    }
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
