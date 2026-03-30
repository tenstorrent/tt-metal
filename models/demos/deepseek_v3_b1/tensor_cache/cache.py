# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed TensorCache for standalone (non-fused) tensors.

Manages the full lifecycle: fingerprint -> CAS lookup -> on miss: preprocess,
serialize to NVMe, load to device -> on hit: load from NVMe to device.

See ARCHITECTURE.md for the full design.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.tensor_cache.fingerprint import canonical, compute_artifact_id
from models.demos.deepseek_v3_b1.tensor_cache.types import (
    CacheEntry,
    CacheEntryState,
    CasPaths,
    Fingerprint,
    TensorTarget,
)


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

    def _cas_paths(self, artifact_id: str) -> CasPaths:
        obj_dir = self._objects_dir / artifact_id[:2] / artifact_id
        return CasPaths(object_dir=obj_dir, data_path=obj_dir / "data.tensorbin")

    def _lookup(self, artifact_id: str) -> CacheEntry:
        paths = self._cas_paths(artifact_id)
        if not paths.object_dir.exists():
            return CacheEntry(artifact_id=artifact_id, state=CacheEntryState.ABSENT, paths=None)
        if paths.data_path.is_file():
            return CacheEntry(artifact_id=artifact_id, state=CacheEntryState.PRESENT, paths=paths)
        return CacheEntry(artifact_id=artifact_id, state=CacheEntryState.CORRUPT, paths=paths)

    def _store(
        self,
        artifact_id: str,
        fingerprint: Fingerprint,
        tensor_host: ttnn.Tensor,
    ) -> CasPaths:
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

            dest = self._cas_paths(artifact_id)
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

    def _load(self, paths: CasPaths, device) -> ttnn.Tensor:
        return ttnn.load_tensor(paths.data_path, device=device)

    @staticmethod
    def _build_mesh_mapper(target: TensorTarget, device):
        """Reconstruct the runtime mesh_mapper from the declarative config + device."""
        cfg = target.mesh_mapper_config
        if cfg.strategy == "replicate":
            return ttnn.ReplicateTensorToMesh(device)
        if cfg.strategy == "shard":
            if cfg.dim is None:
                raise ValueError("MeshMapperConfig with strategy='shard' requires dim")
            return ttnn.ShardTensorToMesh(device, dim=cfg.dim)
        if cfg.strategy == "shard_2d":
            if cfg.dims is None:
                raise ValueError("MeshMapperConfig with strategy='shard_2d' requires dims")
            mesh_shape = (device.shape[0], device.shape[1])
            return ttnn.ShardTensor2dMesh(device, mesh_shape=mesh_shape, dims=cfg.dims)
        raise ValueError(f"Unknown mesh mapper strategy: {cfg.strategy!r}")

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

        if entry.state is CacheEntryState.PRESENT:
            logger.debug("Cache hit for {} ({})", target.name, artifact_id[:12])
            return self._load(entry.paths, device)

        if entry.state is CacheEntryState.CORRUPT:
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
