# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed TensorCache for standalone tensors and fusion groups.

Manages the full lifecycle: fingerprint -> CAS lookup -> on miss: preprocess,
serialize to NVMe, load to device -> on hit: load from NVMe to device.

Use :meth:`get_or_create` for both standalone tensors (:class:`TensorTarget`) and
fusion groups (:class:`FusionGroupSpec`); return type follows ``fingerprint.target``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, Union, runtime_checkable

if TYPE_CHECKING:
    from models.demos.deepseek_v3_b1.blitz_decode_weights import OverlappedTensor

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.tensor_cache.fingerprint import canonical, compute_artifact_id
from models.demos.deepseek_v3_b1.tensor_cache.overlapped_metadata import (
    overlapped_tensor_from_view_dict,
    views_dict_from_overlapped,
)
from models.demos.deepseek_v3_b1.tensor_cache.types import (
    Fingerprint,
    FusionGroupSpec,
    ReplicateMeshMapper,
    Shard2dMeshMapper,
    ShardMeshMapper,
    TensorTarget,
)


@dataclass(frozen=True)
class ContentAddressedStoragePaths:
    """Filesystem paths for one content-addressed storage object."""

    object_dir: Path
    data_path: Path


@dataclass(frozen=True)
class AbsentCacheEntry:
    """CAS lookup result: no object directory exists for this artifact."""

    artifact_id: str


@dataclass(frozen=True)
class PresentCacheEntry:
    """CAS lookup result: object directory and data file both exist."""

    artifact_id: str
    paths: ContentAddressedStoragePaths


@dataclass(frozen=True)
class CorruptCacheEntry:
    """CAS lookup result: object directory exists but data file is missing."""

    artifact_id: str
    paths: ContentAddressedStoragePaths


CacheEntry = Union[AbsentCacheEntry, PresentCacheEntry, CorruptCacheEntry]


@runtime_checkable
class TensorCacheProtocol(Protocol):
    """Structural interface shared by :class:`TensorCache` and :class:`EphemeralTensorCache`."""

    def get_or_create(
        self,
        fingerprint: Fingerprint,
        device,
        *,
        preprocess: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        raw_tensors: Callable[[], dict[str, torch.Tensor]] | dict[str, torch.Tensor],
    ) -> ttnn.Tensor | dict[str, OverlappedTensor]:
        ...


def build_mesh_mapper_for_target(target: TensorTarget, device):
    """Reconstruct the runtime mesh_mapper from the declarative config + device."""
    mapper_config = target.mesh_mapper_config
    if isinstance(mapper_config, ReplicateMeshMapper):
        return ttnn.ReplicateTensorToMesh(device)
    if isinstance(mapper_config, ShardMeshMapper):
        return ttnn.ShardTensorToMesh(device, dim=mapper_config.dim)
    if isinstance(mapper_config, Shard2dMeshMapper):
        mesh_shape = (device.shape[0], device.shape[1])
        return ttnn.ShardTensor2dMesh(device, mesh_shape=mesh_shape, dims=mapper_config.dims)
    raise TypeError(f"Unknown mesh mapper config type: {type(mapper_config)}")


def _create_overlapped_tensor_fused(
    spec: FusionGroupSpec,
    preprocessed: dict[str, torch.Tensor],
    device,
    *,
    move_to_device: bool = False,
) -> tuple[ttnn.Tensor, dict[str, OverlappedTensor]]:
    """Lazy import of :mod:`fuse` to avoid circular import with ``blitz_decode_weights``."""
    from models.demos.deepseek_v3_b1.tensor_cache.fuse import create_overlapped_tensor

    return create_overlapped_tensor(spec, preprocessed, device, move_to_device=move_to_device)


def _logical_name_from_fingerprint(fingerprint: Fingerprint) -> str:
    return fingerprint.target.name


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class TensorCache:
    """Content-addressed cache backed by a local filesystem (NVMe).

    ``local_root`` is created with ``mkdir(parents=True, exist_ok=True)`` if it does not exist.

    Storage layout::

        {local_root}/
          objects/{id[:2]}/{id}/
            manifest.json
            metadata.json
            data.tensorbin

    Note: current implementation writes directly into the final object directory
    (no separate staging directory yet).
    """

    def __init__(self, local_root: Path):
        self.local_root = Path(local_root)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self._objects_dir = self.local_root / "objects"

    def _content_addressed_paths(self, artifact_id: str) -> ContentAddressedStoragePaths:
        object_dir = self._objects_dir / artifact_id[:2] / artifact_id
        return ContentAddressedStoragePaths(object_dir=object_dir, data_path=object_dir / "data.tensorbin")

    def _lookup(self, artifact_id: str) -> CacheEntry:
        paths = self._content_addressed_paths(artifact_id)
        if not paths.object_dir.exists():
            return AbsentCacheEntry(artifact_id=artifact_id)
        if paths.data_path.is_file():
            return PresentCacheEntry(artifact_id=artifact_id, paths=paths)
        return CorruptCacheEntry(artifact_id=artifact_id, paths=paths)

    def _write_artifact_blob_and_manifest(
        self,
        artifact_id: str,
        fingerprint: Fingerprint,
        tensor_host: ttnn.Tensor,
    ) -> tuple[ContentAddressedStoragePaths, str, int]:
        """Write ``data.tensorbin`` and ``manifest.json``; return paths and stats for ``metadata.json``.

        TODO: Publish atomically (stage under a unique temporary directory, then rename
        into ``objects/``) for crash safety and to avoid concurrent writers corrupting
        the same artifact.
        """
        dest = self._content_addressed_paths(artifact_id)
        dest.object_dir.mkdir(parents=True, exist_ok=True)

        data_path = dest.object_dir / "data.tensorbin"
        ttnn.dump_tensor(data_path, tensor_host, mode=ttnn.DumpTensorMode.LOCAL)

        content_hash = _sha256_file(data_path)
        size_bytes = data_path.stat().st_size

        manifest_dict = {
            "fingerprint": canonical(fingerprint),
            "logical_name": _logical_name_from_fingerprint(fingerprint),
        }
        with open(dest.object_dir / "manifest.json", "w") as f:
            json.dump(manifest_dict, f, indent=2, sort_keys=True)

        return dest, content_hash, size_bytes

    def _store(
        self,
        artifact_id: str,
        fingerprint: Fingerprint,
        tensor_host: ttnn.Tensor,
    ) -> ContentAddressedStoragePaths:
        dest, content_hash, size_bytes = self._write_artifact_blob_and_manifest(artifact_id, fingerprint, tensor_host)
        metadata_dict = {
            "artifact_id": artifact_id,
            "content_hash": content_hash,
            "size_bytes": size_bytes,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(dest.object_dir / "metadata.json", "w") as f:
            json.dump(metadata_dict, f, indent=2, sort_keys=True)
        return dest

    def _load(self, paths: ContentAddressedStoragePaths, device) -> ttnn.Tensor:
        return ttnn.load_tensor(paths.data_path, device=device)

    def _store_fused(
        self,
        artifact_id: str,
        fingerprint: Fingerprint,
        fused_host: ttnn.Tensor,
        views: dict[str, OverlappedTensor],
    ) -> ContentAddressedStoragePaths:
        """Persist fused host tensor and per-view metadata (OverlappedTensor, without device)."""
        dest, content_hash, size_bytes = self._write_artifact_blob_and_manifest(artifact_id, fingerprint, fused_host)
        metadata_dict = {
            "artifact_id": artifact_id,
            "artifact_kind": "fusion_group",
            "content_hash": content_hash,
            "size_bytes": size_bytes,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "views": views_dict_from_overlapped(views),
        }
        with open(dest.object_dir / "metadata.json", "w") as f:
            json.dump(metadata_dict, f, indent=2, sort_keys=True)
        return dest

    def _load_fused(
        self,
        paths: ContentAddressedStoragePaths,
        device,
        *,
        meta: dict | None = None,
    ) -> dict[str, OverlappedTensor]:
        fused = ttnn.load_tensor(paths.data_path, device=device)
        if meta is None:
            with open(paths.object_dir / "metadata.json") as f:
                meta = json.load(f)
        views_raw = meta.get("views")
        if not views_raw:
            raise ValueError(f"Missing views in metadata for fusion artifact: {paths.object_dir}")
        out: dict[str, OverlappedTensor] = {}
        for name, d in views_raw.items():
            out[name] = overlapped_tensor_from_view_dict(fused, d)
        return out

    def get_or_create(
        self,
        fingerprint: Fingerprint,
        device,
        *,
        preprocess: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        raw_tensors: Callable[[], dict[str, torch.Tensor]] | dict[str, torch.Tensor],
    ) -> ttnn.Tensor | dict[str, OverlappedTensor]:
        """Load from cache or build, then return a device tensor or overlapped views.

        Dispatches on ``fingerprint.target``:

        * :class:`TensorTarget` — returns ``ttnn.Tensor`` (standalone artifact).
        * :class:`FusionGroupSpec` — returns ``dict[str, OverlappedTensor]`` (fused buffer + views).

        On hit: load ``data.tensorbin`` from NVMe; fusion entries also read ``views`` from
        ``metadata.json``.

        On miss: ``raw_tensors()`` (optional lazy) → ``preprocess()`` → build host tensor(s),
        persist to disk, load to device.

        ``raw_tensors`` may be a callable so HF weights are not read on a warm cache.
        """
        target = fingerprint.target
        if not isinstance(target, (TensorTarget, FusionGroupSpec)):
            raise TypeError(
                f"TensorCache.get_or_create requires TensorTarget or FusionGroupSpec target, got {type(target)}"
            )

        artifact_id = compute_artifact_id(fingerprint)
        entry = self._lookup(artifact_id)
        logical = _logical_name_from_fingerprint(fingerprint)

        if isinstance(entry, PresentCacheEntry):
            if isinstance(target, TensorTarget):
                return self._load(entry.paths, device)
            meta_path = entry.paths.object_dir / "metadata.json"
            if meta_path.is_file():
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("views"):
                    return self._load_fused(entry.paths, device, meta=meta)
            logger.warning(
                "Present cache entry for fused {} ({}) missing fusion metadata; rebuilding",
                logical,
                artifact_id[:12],
            )
            shutil.rmtree(entry.paths.object_dir, ignore_errors=True)

        if isinstance(entry, CorruptCacheEntry):
            logger.warning("Corrupt cache entry for {} ({}), rebuilding", logical, artifact_id[:12])
            shutil.rmtree(entry.paths.object_dir, ignore_errors=True)

        t0 = time.perf_counter()
        tensors = raw_tensors() if callable(raw_tensors) else raw_tensors
        preprocessed = preprocess(tensors)

        if isinstance(target, TensorTarget):
            torch_tensor = preprocessed[target.name]
            from_torch_kwargs: dict = {
                "dtype": target.dtype,
                "layout": target.layout,
                "device": None,
                "memory_config": target.memory_config,
                "mesh_mapper": build_mesh_mapper_for_target(target, device),
            }
            if target.layout == ttnn.TILE_LAYOUT:
                from_torch_kwargs["tile"] = ttnn.Tile(target.tile_shape)
            tensor_host = ttnn.from_torch(torch_tensor, **from_torch_kwargs)
            paths = self._store(artifact_id, fingerprint, tensor_host)
            elapsed = time.perf_counter() - t0
            logger.info("Cache miss for {} resolved in {:.3f}s, stored as {}", logical, elapsed, artifact_id[:12])
            return self._load(paths, device)

        spec = target
        fused_host, views = _create_overlapped_tensor_fused(
            spec,
            preprocessed,
            device,
            move_to_device=False,
        )
        paths = self._store_fused(artifact_id, fingerprint, fused_host, views)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Cache miss (fused) for {} resolved in {:.3f}s, stored as {}",
            logical,
            elapsed,
            artifact_id[:12],
        )
        return self._load_fused(paths, device, meta={"views": views_dict_from_overlapped(views)})


class EphemeralTensorCache:
    """In-memory passthrough implementing the same ``get_or_create`` API as :class:`TensorCache`.

    Builds tensors directly without disk persistence. Used when ``cache_config`` is omitted
    in prepare functions so there is a single code path.
    """

    def __init__(self, *, move_to_device: bool = True):
        self._move_to_device = move_to_device

    def get_or_create(
        self,
        fingerprint: Fingerprint,
        device,
        *,
        preprocess: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        raw_tensors: Callable[[], dict[str, torch.Tensor]] | dict[str, torch.Tensor],
    ) -> ttnn.Tensor | dict[str, OverlappedTensor]:
        target = fingerprint.target
        if not isinstance(target, (TensorTarget, FusionGroupSpec)):
            raise TypeError(
                f"EphemeralTensorCache.get_or_create requires TensorTarget or FusionGroupSpec target, "
                f"got {type(target)}"
            )

        tensors = raw_tensors() if callable(raw_tensors) else raw_tensors
        preprocessed = preprocess(tensors)

        if isinstance(target, TensorTarget):
            torch_tensor = preprocessed[target.name]
            from_torch_kwargs: dict = {
                "dtype": target.dtype,
                "layout": target.layout,
                "device": device if self._move_to_device else None,
                "memory_config": target.memory_config,
                "mesh_mapper": build_mesh_mapper_for_target(target, device),
            }
            if target.layout == ttnn.TILE_LAYOUT:
                from_torch_kwargs["tile"] = ttnn.Tile(target.tile_shape)
            return ttnn.from_torch(torch_tensor, **from_torch_kwargs)

        _fused, views = _create_overlapped_tensor_fused(
            target,
            preprocessed,
            device,
            move_to_device=self._move_to_device,
        )
        return views
