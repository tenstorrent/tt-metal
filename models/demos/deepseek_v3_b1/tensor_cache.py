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
* **Tensor lists** (``get_or_create_tensor_list``) — ordered lists of
  ``ttnn.Tensor`` objects, stored as individual ``.tensorbin`` files and
  loaded sequentially.  Atomicity is guaranteed via a ``_complete``
  sentinel: either all tensors are present or the entry is treated as a
  miss.  Sequential loading preserves allocation order, which is
  critical for contiguous DRAM placement (e.g. routed MoE experts).

All write paths are crash-safe: single-file artifacts are written to a
temporary file and atomically renamed (``os.rename``); list artifacts
use a ``_complete`` sentinel that is only created after all items are
flushed.

Storage uses content-addressed directories keyed by
``sha256(fingerprint)`` under a configurable root.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlappedTensor

_T = TypeVar("_T")


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


class TensorCache:
    """Content-addressed cache for weight tensors (overlapped, standalone, and lists).

    Storage layout::

        <root>/objects/<prefix>/<artifact_id>/data.overlappedtensorbin
        <root>/objects/<prefix>/<artifact_id>/data.tensorbin
        <root>/objects/<prefix>/<artifact_id>/list/NNN.tensorbin + _complete
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def _artifact_dir(self, artifact_id: str) -> Path:
        prefix = artifact_id[:2]
        return self._root / "objects" / prefix / artifact_id

    _COMPLETE_MARKER = "_complete"

    # ------------------------------------------------------------------
    # Existence checks
    # ------------------------------------------------------------------

    def has_overlapped(self, fingerprint: Fingerprint) -> bool:
        """Return True if the overlapped-tensor artifact exists."""
        return (self._artifact_dir(fingerprint.artifact_id()) / "data.overlappedtensorbin").exists()

    def has_tensor(self, fingerprint: Fingerprint) -> bool:
        """Return True if the standalone-tensor artifact exists."""
        return (self._artifact_dir(fingerprint.artifact_id()) / "data.tensorbin").exists()

    def has_tensor_list(self, fingerprint: Fingerprint) -> bool:
        """Return True if the tensor-list sentinel exists."""
        return (self._artifact_dir(fingerprint.artifact_id()) / "list" / self._COMPLETE_MARKER).exists()

    # ------------------------------------------------------------------
    # Load-only methods
    # ------------------------------------------------------------------

    def load_overlapped(self, fingerprint: Fingerprint, *, device: ttnn.Device) -> dict[str, OverlappedTensor]:
        """Load cached overlapped views. Raises if artifact is missing."""
        artifact_id = fingerprint.artifact_id()
        path = self._artifact_dir(artifact_id) / "data.overlappedtensorbin"
        return ttnn._ttnn.tensor.load_overlapped_tensors(str(path), device=device)

    def load_tensor(self, fingerprint: Fingerprint, *, device: ttnn.Device) -> ttnn.Tensor:
        """Load cached standalone tensor. Raises if artifact is missing."""
        artifact_id = fingerprint.artifact_id()
        path = self._artifact_dir(artifact_id) / "data.tensorbin"
        return ttnn.load_tensor(str(path), device=device)

    def load_tensor_list(self, fingerprint: Fingerprint, *, device: ttnn.Device) -> list[ttnn.Tensor]:
        """Load cached tensor list. Raises if sentinel or directory is missing."""
        artifact_id = fingerprint.artifact_id()
        list_dir = self._artifact_dir(artifact_id) / "list"
        if not (list_dir / self._COMPLETE_MARKER).exists():
            raise FileNotFoundError(
                f"Tensor list cache entry incomplete or missing: {list_dir} "
                f"(group={fingerprint.group_name}, layer={fingerprint.layer_idx})"
            )
        paths = sorted(list_dir.glob("*.tensorbin"), key=lambda p: int(p.stem))
        return [ttnn.load_tensor(str(p), device=device) for p in paths]

    # ------------------------------------------------------------------
    # Get-or-create methods (create on miss, load on hit)
    # ------------------------------------------------------------------

    def _get_or_create_file(
        self,
        fingerprint: Fingerprint,
        filename: str,
        *,
        create: Callable[[], _T],
        load: Callable[[str], _T],
        dump: Callable[[str, _T], None],
        miss_label: str,
    ) -> _T:
        """Shared logic for single-file cache entries (overlapped and standalone).

        On miss, writes to a ``.tmp`` sibling and atomically renames into
        place so a crash never leaves a half-written artifact.
        """
        artifact_id = fingerprint.artifact_id()
        path = self._artifact_dir(artifact_id) / filename

        if path.exists():
            logger.debug(
                "Cache hit for {} layer {} ({})", fingerprint.group_name, fingerprint.layer_idx, artifact_id[:12]
            )
            return load(str(path))

        logger.debug("Cache miss for {} layer {} — {}", fingerprint.group_name, fingerprint.layer_idx, miss_label)
        result = create()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        dump(str(tmp), result)
        os.rename(str(tmp), str(path))
        return result

    def get_or_create(
        self,
        fingerprint: Fingerprint,
        *,
        fuse: Callable[[], dict[str, OverlappedTensor]],
        device: ttnn.Device,
    ) -> dict[str, OverlappedTensor]:
        """Return cached overlapped views on hit, or fuse + store + return on miss."""
        return self._get_or_create_file(
            fingerprint,
            "data.overlappedtensorbin",
            create=fuse,
            load=lambda p: ttnn._ttnn.tensor.load_overlapped_tensors(p, device=device),
            dump=ttnn._ttnn.tensor.dump_overlapped_tensors,
            miss_label="fusing",
        )

    def get_or_create_tensor(
        self,
        fingerprint: Fingerprint,
        *,
        create: Callable[[], ttnn.Tensor],
        device: ttnn.Device,
    ) -> ttnn.Tensor:
        """Return cached standalone tensor on hit, or create + store + return on miss."""
        return self._get_or_create_file(
            fingerprint,
            "data.tensorbin",
            create=create,
            load=lambda p: ttnn.load_tensor(p, device=device),
            dump=ttnn.dump_tensor,
            miss_label="creating",
        )

    def get_or_create_tensor_list(
        self,
        fingerprint: Fingerprint,
        *,
        create: Callable[[], list[ttnn.Tensor]],
        device: ttnn.Device,
    ) -> list[ttnn.Tensor]:
        """Return cached tensor list on hit, or create + store + return on miss.

        All-or-nothing: a ``_complete`` sentinel guards against partial
        writes.  On hit every tensor is loaded sequentially so the device
        allocator preserves the original allocation order (important for
        contiguous DRAM placement).
        """
        artifact_id = fingerprint.artifact_id()
        list_dir = self._artifact_dir(artifact_id) / "list"
        marker = list_dir / self._COMPLETE_MARKER

        if marker.exists():
            logger.debug(
                "Cache hit for {} layer {} ({})", fingerprint.group_name, fingerprint.layer_idx, artifact_id[:12]
            )
            paths = sorted(list_dir.glob("*.tensorbin"), key=lambda p: int(p.stem))
            return [ttnn.load_tensor(str(p), device=device) for p in paths]

        logger.debug("Cache miss for {} layer {} — creating list", fingerprint.group_name, fingerprint.layer_idx)
        tensors = create()
        list_dir.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(tensors):
            ttnn.dump_tensor(str(list_dir / f"{i:04d}.tensorbin"), t)
        marker.touch()
        return tensors


@dataclass(frozen=True)
class CacheConfig:
    """Bundle of cache + model identity for passing through prepare_* functions."""

    cache: TensorCache
    hf_model_id: str
    hf_revision: str
