# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed TensorCache for standalone (non-fused) tensors.

Manages the full lifecycle: fingerprint -> CAS lookup -> on miss: preprocess,
serialize to NVMe, load to device -> on hit: load from NVMe to device.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Union

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.tensor_cache.fingerprint import canonical, compute_artifact_id
from models.demos.deepseek_v3_b1.tensor_cache.types import (
    Fingerprint,
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

    Storage layout::

        {local_root}/
          objects/{id[:2]}/{id}/
            manifest.json
            metadata.json
            data.tensorbin
          tmp/
            {id}_{pid}/
    """

    def __init__(self, local_root: Path):
        self.local_root = Path(local_root)
        self._objects_dir = self.local_root / "objects"
        self._tmp_dir = self.local_root / "tmp"

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

    def _store(
        self,
        artifact_id: str,
        fingerprint: Fingerprint,
        tensor_host: ttnn.Tensor,
    ) -> ContentAddressedStoragePaths:
        tmp = self._tmp_dir / f"{artifact_id}_{os.getpid()}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            data_path = tmp / "data.tensorbin"
            ttnn.dump_tensor(data_path, tensor_host)

            content_hash = _sha256_file(data_path)
            size_bytes = data_path.stat().st_size

            manifest_dict = {
                "fingerprint": canonical(fingerprint),
                "logical_name": fingerprint.target.name if isinstance(fingerprint.target, TensorTarget) else None,
            }
            with open(tmp / "manifest.json", "w") as f:
                json.dump(manifest_dict, f, indent=2, sort_keys=True)

            metadata_dict = {
                "artifact_id": artifact_id,
                "content_hash": content_hash,
                "size_bytes": size_bytes,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            with open(tmp / "metadata.json", "w") as f:
                json.dump(metadata_dict, f, indent=2, sort_keys=True)

            dest = self._content_addressed_paths(artifact_id)
            dest.object_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(str(tmp), str(dest.object_dir))
            except OSError:
                # Another process wrote the same artifact concurrently -- that's fine,
                # same fingerprint guarantees same content.
                shutil.rmtree(tmp, ignore_errors=True)
            return dest
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def _load(self, paths: ContentAddressedStoragePaths, device) -> ttnn.Tensor:
        return ttnn.load_tensor(paths.data_path, device=device)

    @staticmethod
    def _build_mesh_mapper(target: TensorTarget, device):
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

    def get_or_create(
        self,
        fingerprint: Fingerprint,
        device,
        *,
        preprocess: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        raw_tensors: Callable[[], dict[str, torch.Tensor]] | dict[str, torch.Tensor],
    ) -> ttnn.Tensor:
        """Return a device tensor, loading from cache on hit or building on miss.

        On hit:  load from local NVMe directly to device.
        On miss: raw_tensors() -> preprocess() -> from_torch(host) -> store -> load to device.

        raw_tensors can be a callable (lazy) so HF safetensors are never read when
        the cache is warm.
        """
        target = fingerprint.target
        if not isinstance(target, TensorTarget):
            raise TypeError(f"TensorCache only supports TensorTarget, got {type(target)}")

        artifact_id = compute_artifact_id(fingerprint)
        entry = self._lookup(artifact_id)

        if isinstance(entry, PresentCacheEntry):
            logger.debug("Cache hit for {} ({})", target.name, artifact_id[:12])
            return self._load(entry.paths, device)

        if isinstance(entry, CorruptCacheEntry):
            logger.warning("Corrupt cache entry for {} ({}), rebuilding", target.name, artifact_id[:12])
            shutil.rmtree(entry.paths.object_dir, ignore_errors=True)

        logger.info("Cache miss for {} ({}), preprocessing...", target.name, artifact_id[:12])
        t0 = time.perf_counter()

        tensors = raw_tensors() if callable(raw_tensors) else raw_tensors
        preprocessed = preprocess(tensors)

        torch_tensor = preprocessed[target.name]

        from_torch_kwargs: dict = {
            "dtype": target.dtype,
            "layout": target.layout,
            "device": None,
            "memory_config": target.memory_config,
            "mesh_mapper": self._build_mesh_mapper(target, device),
        }
        if target.layout == ttnn.TILE_LAYOUT:
            from_torch_kwargs["tile"] = ttnn.Tile(target.tile_shape)
        tensor_host = ttnn.from_torch(torch_tensor, **from_torch_kwargs)

        paths = self._store(artifact_id, fingerprint, tensor_host)
        elapsed = time.perf_counter() - t0
        logger.info("Cache miss for {} resolved in {:.3f}s, stored as {}", target.name, elapsed, artifact_id[:12])

        return self._load(paths, device)
