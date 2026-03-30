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

from models.demos.deepseek_v3_b1.tensor_cache.types import Fingerprint, TensorTarget


def canonical(fp: Fingerprint) -> dict:
    """Produce a deterministic, JSON-serializable dict from a Fingerprint.

    Only TensorTarget is supported; FusionGroupSpec will be added later.
    """
    target = fp.target
    if not isinstance(target, TensorTarget):
        raise TypeError(f"Unsupported target type: {type(target)}")
    cfg = target.mesh_mapper_config
    target_dict = {
        "kind": "tensor",
        "name": target.name,
        "dtype": target.dtype.name,
        "layout": target.layout.name,
        "memory_config": json.loads(target.memory_config.to_json()),
        "tile_shape": list(target.tile_shape),
        "mesh_mapper_config": {
            "strategy": cfg.strategy,
            "dim": cfg.dim,
            "dims": list(cfg.dims) if cfg.dims is not None else None,
        },
    }
    return {
        "schema_version": fp.schema_version,
        "source": sorted(fp.source.names),
        "hf_model_id": fp.hf_model_id,
        "hf_revision": fp.hf_revision,
        "transform_version": fp.transform_version,
        "mesh_shape": list(fp.mesh_shape),
        "target": target_dict,
    }


def compute_artifact_id(fp: Fingerprint) -> str:
    """SHA-256 hex digest of the canonical JSON representation."""
    blob = json.dumps(canonical(fp), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
