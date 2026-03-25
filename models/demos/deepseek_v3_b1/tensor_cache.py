# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed lazy tensor cache for weight tensors.

The cache is transparent: callers provide a fingerprint (cache key), a
creation callback, and a device reference.  On a hit the cached artifact
is loaded directly; on a miss the callback is invoked, the result is
stored, and then returned.

Three storage formats are supported:

* **Overlapped tensors** (``get_or_create``) — fusion groups serialized
  via the C++ FlatBuffer path (``dump_overlapped_tensors`` /
  ``load_overlapped_tensors``).
* **Standalone tensors** (``get_or_create_tensor``) — single
  ``ttnn.Tensor`` objects serialized via ``ttnn.dump_tensor`` /
  ``ttnn.load_tensor``.
* **Routed expert lists** (``get_or_create_routed_experts``) — per-expert
  ``ttnn.Tensor`` files (768 per MoE layer) loaded in projection-first
  order to guarantee contiguous DRAM allocation.

Storage uses content-addressed directories keyed by
``sha256(fingerprint)`` under a configurable root.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlappedTensor


@dataclass(frozen=True)
class Fingerprint:
    """Cache key for a single fusion group.

    All fields are known before any computation runs.  The artifact ID
    is derived deterministically from them so the same inputs always
    map to the same cache entry.

    ``group_name`` identifies the fusion group (e.g. ``"q_ab_kv_a"``).
    ``layer_idx`` distinguishes per-layer weight data.
    ``spec_fingerprints`` captures the exact layout spec hashes so that
    any spec change (dtype, core grid, tile dims) invalidates the entry.
    """

    schema_version: int
    hf_model_id: str
    hf_revision: str
    transform_version: int
    mesh_shape: tuple[int, int]
    group_name: str
    layer_idx: int
    spec_fingerprints: tuple[str, ...]

    def artifact_id(self) -> str:
        """Deterministic SHA-256 of all fingerprint fields."""
        canonical = {
            "schema_version": self.schema_version,
            "hf_model_id": self.hf_model_id,
            "hf_revision": self.hf_revision,
            "transform_version": self.transform_version,
            "mesh_shape": list(self.mesh_shape),
            "group_name": self.group_name,
            "layer_idx": self.layer_idx,
            "spec_fingerprints": list(self.spec_fingerprints),
        }
        blob = json.dumps(canonical, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


_TRANSFORM_VERSION = 1


class TensorCache:
    """Content-addressed cache for weight tensors (overlapped, standalone, and routed experts).

    Storage layout::

        <root>/objects/<prefix>/<artifact_id>/data.overlappedtensorbin
        <root>/objects/<prefix>/<artifact_id>/data.tensorbin
        <root>/objects/<prefix>/<artifact_id>/experts/e_NNN/{gate,up,down}_proj.tensorbin
        <root>/objects/<prefix>/<artifact_id>/_complete   (sentinel for routed experts)
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def _artifact_dir(self, artifact_id: str) -> Path:
        prefix = artifact_id[:2]
        return self._root / "objects" / prefix / artifact_id

    def get_or_create(
        self,
        fingerprint: Fingerprint,
        *,
        fuse: Callable[[], dict[str, OverlappedTensor]],
        device: ttnn.Device,
    ) -> dict[str, OverlappedTensor]:
        """Return cached overlapped views on hit, or fuse + store + return on miss.

        Args:
            fingerprint: The cache key identifying this fusion group.
            fuse: A callable that produces the ``dict[str, OverlappedTensor]``
                when invoked (only called on a cache miss).
            device: The mesh device to load onto (hit path).
        """
        artifact_id = fingerprint.artifact_id()
        path = self._artifact_dir(artifact_id) / "data.overlappedtensorbin"

        if path.exists():
            logger.debug(
                "Cache hit for {} layer {} ({})", fingerprint.group_name, fingerprint.layer_idx, artifact_id[:12]
            )
            return ttnn._ttnn.tensor.load_overlapped_tensors(str(path), device=device)

        logger.debug("Cache miss for {} layer {} — fusing", fingerprint.group_name, fingerprint.layer_idx)
        views = fuse()
        path.parent.mkdir(parents=True, exist_ok=True)
        ttnn._ttnn.tensor.dump_overlapped_tensors(str(path), views)
        return views

    def get_or_create_tensor(
        self,
        fingerprint: Fingerprint,
        *,
        create: Callable[[], ttnn.Tensor],
        device: ttnn.Device,
    ) -> ttnn.Tensor:
        """Return cached standalone tensor on hit, or create + store + return on miss.

        Uses ``ttnn.dump_tensor`` / ``ttnn.load_tensor`` for serialization.

        Args:
            fingerprint: The cache key identifying this tensor.
            create: A callable that produces the ``ttnn.Tensor``
                when invoked (only called on a cache miss).
            device: The mesh device to load onto (hit path).
        """
        artifact_id = fingerprint.artifact_id()
        path = self._artifact_dir(artifact_id) / "data.tensorbin"

        if path.exists():
            logger.debug(
                "Cache hit for {} layer {} ({})", fingerprint.group_name, fingerprint.layer_idx, artifact_id[:12]
            )
            return ttnn.load_tensor(str(path), device=device)

        logger.debug("Cache miss for {} layer {} — creating", fingerprint.group_name, fingerprint.layer_idx)
        tensor = create()
        path.parent.mkdir(parents=True, exist_ok=True)
        ttnn.dump_tensor(str(path), tensor)
        return tensor

    _COMPLETE_MARKER = "_complete"

    def get_or_create_routed_experts(
        self,
        fingerprint: Fingerprint,
        *,
        create: Callable,
        device,
        num_experts: int,
    ):
        """Return cached MoE routed expert weights on hit, or create + store on miss.

        Stores 768 files (``num_experts`` x 3 projections) under
        ``experts/e_NNN/{gate,up,down}_proj.tensorbin``.  A ``_complete``
        sentinel file is written after all expert files to guard against
        partial writes.

        On load, experts are read in projection-first order (all gates,
        then all ups, then all downs) to guarantee contiguous DRAM
        allocation required by ``DRAMStreamingMatmul``.

        Args:
            fingerprint: Cache key for this layer's routed experts.
            create: Callable returning a ``MoERoutedExpertWeights`` on miss.
            device: Device to load onto (hit path).
            num_experts: Expected number of experts (e.g. 256).
        """
        from models.demos.deepseek_v3_b1.prepare_weights import MoERoutedExpertWeights

        artifact_id = fingerprint.artifact_id()
        artifact_dir = self._artifact_dir(artifact_id)
        marker = artifact_dir / self._COMPLETE_MARKER

        if marker.exists():
            logger.debug(
                "Cache hit for {} layer {} ({})", fingerprint.group_name, fingerprint.layer_idx, artifact_id[:12]
            )
            experts_dir = artifact_dir / "experts"
            routed_gate_proj = []
            routed_up_proj = []
            routed_down_proj = []
            for e in range(num_experts):
                expert_dir = experts_dir / f"e_{e:03d}"
                routed_gate_proj.append(ttnn.load_tensor(str(expert_dir / "gate_proj.tensorbin"), device=device))
            for e in range(num_experts):
                expert_dir = experts_dir / f"e_{e:03d}"
                routed_up_proj.append(ttnn.load_tensor(str(expert_dir / "up_proj.tensorbin"), device=device))
            for e in range(num_experts):
                expert_dir = experts_dir / f"e_{e:03d}"
                routed_down_proj.append(ttnn.load_tensor(str(expert_dir / "down_proj.tensorbin"), device=device))
            result = MoERoutedExpertWeights(
                routed_gate_proj=routed_gate_proj,
                routed_up_proj=routed_up_proj,
                routed_down_proj=routed_down_proj,
            )
            result.validate_contiguous_dram()
            return result

        logger.debug("Cache miss for {} layer {} — creating", fingerprint.group_name, fingerprint.layer_idx)
        result = create()
        experts_dir = artifact_dir / "experts"
        for e in range(num_experts):
            expert_dir = experts_dir / f"e_{e:03d}"
            expert_dir.mkdir(parents=True, exist_ok=True)
            ttnn.dump_tensor(str(expert_dir / "gate_proj.tensorbin"), result.routed_gate_proj[e])
            ttnn.dump_tensor(str(expert_dir / "up_proj.tensorbin"), result.routed_up_proj[e])
            ttnn.dump_tensor(str(expert_dir / "down_proj.tensorbin"), result.routed_down_proj[e])
        marker.touch()
        return result


@dataclass(frozen=True)
class CacheConfig:
    """Bundle of cache + model identity for passing through prepare_* functions."""

    cache: TensorCache
    hf_model_id: str
    hf_revision: str
