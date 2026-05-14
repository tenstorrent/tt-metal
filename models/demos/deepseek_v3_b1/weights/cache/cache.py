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

import base64
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, Union, runtime_checkable

if TYPE_CHECKING:
    from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import CompressedTensor
    from models.demos.deepseek_v3_b1.weights.overlap.packing import OverlappedTensor

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.weights.cache.fingerprint import canonical, compute_artifact_id
from models.demos.deepseek_v3_b1.weights.cache.overlapped_metadata import (
    overlapped_tensor_from_view_dict,
    views_dict_from_overlapped,
)
from models.demos.deepseek_v3_b1.weights.cache.types import (
    CompressedTensorBuildInputs,
    CompressedTensorTarget,
    Fingerprint,
    FusionGroupSpec,
    ReplicateMeshMapper,
    Shard2dMeshMapper,
    ShardMeshMapper,
    SramCompressedTensorTarget,
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
        move_to_device: bool = True,
        preprocess: Callable[[dict[str, torch.Tensor]], dict],
        raw_tensors: Callable[[], dict[str, torch.Tensor]] | dict[str, torch.Tensor],
        reconstruct: Callable[["CompressedTensorBuildInputs", object], "CompressedTensor"] | None = None,
    ) -> "ttnn.Tensor | dict[str, OverlappedTensor] | CompressedTensor":
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
) -> tuple[ttnn.Tensor, dict[str, "OverlappedTensor"]]:
    """Lazy import of :mod:`fuse` to keep top-level import graph acyclic."""
    from models.demos.deepseek_v3_b1.weights.cache.fuse import create_overlapped_tensor

    return create_overlapped_tensor(spec, preprocessed, device, move_to_device=move_to_device)


def _logical_name_from_fingerprint(fingerprint: Fingerprint) -> str:
    return fingerprint.target.name


def _coord_as_list(coord) -> list[int]:
    """Convert a ``ttnn.MeshCoordinate`` (or list/tuple) into a plain ``list[int]``.

    Used by the SRAM cache writer to serialize per-device metadata; warm
    load reverses the conversion via ``ttnn.MeshCoordinate(list(...))``.
    """
    if isinstance(coord, (list, tuple)):
        return [int(v) for v in coord]
    # ttnn.MeshCoordinate exposes ``dims()`` + ``__getitem__``.
    if hasattr(coord, "dims") and hasattr(coord, "__getitem__"):
        return [int(coord[i]) for i in range(coord.dims())]
    raise TypeError(f"Cannot convert coordinate {coord!r} ({type(coord)}) to list[int]")


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

    def _load(self, paths: ContentAddressedStoragePaths, device, *, move_to_device: bool = True) -> ttnn.Tensor:
        return ttnn.load_tensor(paths.data_path, device=device if move_to_device else None)

    def _store_fused(
        self,
        artifact_id: str,
        fingerprint: Fingerprint,
        fused_host: ttnn.Tensor,
        views: dict[str, "OverlappedTensor"],
    ) -> ContentAddressedStoragePaths:
        """Persist fused host tensor and per-view metadata (OverlappedTensor, without device).

        The per-core allocation flag is round-tripped through the tensor
        flatbuffer schema itself (see ``MemoryConfig.per_core_allocation`` in
        ``tensor_spec.fbs``), so no external hint is required here.
        """
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
        move_to_device: bool = True,
        meta: dict | None = None,
    ) -> dict[str, "OverlappedTensor"]:
        if meta is None:
            with open(paths.object_dir / "metadata.json") as f:
                meta = json.load(f)
        # The per-core allocation flag is round-tripped through the tensor
        # flatbuffer schema (``MemoryConfig.per_core_allocation`` in
        # ``tensor_spec.fbs``), so a plain ``ttnn.load_tensor(device=...)``
        # reconstructs the correct allocator semantics with no Python-side
        # patching required.
        fused = ttnn.load_tensor(paths.data_path, device=device if move_to_device else None)
        # If the loaded fused buffer ended up per-core allocated, verify that its
        # L1 base address is the same on every core it spans.  Downstream kernel
        # setup (e.g. attention_block/op.py::_fused_base_addr) queries a single
        # core and broadcasts that address to all cores via CB setup; non-uniform
        # addresses would silently corrupt matmul loads rather than fail loudly.
        # ``overlap_tensors`` runs this same guard on the cold pack path; this
        # check covers the warm-load reconstitution above.
        if fused.is_per_core_allocated():
            from models.demos.deepseek_v3_b1.weights.overlap.packing import assert_uniform_per_core_addresses
            from models.demos.deepseek_v3_b1.weights.overlap.spec import _core_list

            cores = _core_list(fused.memory_config().shard_spec.grid)
            assert_uniform_per_core_addresses(fused, cores)
        views_raw = meta.get("views")
        if not views_raw:
            raise ValueError(f"Missing views in metadata for fusion artifact: {paths.object_dir}")
        out: dict[str, "OverlappedTensor"] = {}
        for name, d in views_raw.items():
            out[name] = overlapped_tensor_from_view_dict(fused, d)
        return out

    # ------------------------------------------------------------------
    # Compressed-tensor helpers (compact BSPM disk format)
    # ------------------------------------------------------------------

    def _lookup_compressed(self, artifact_id: str) -> "CacheEntry":
        """Check for compact BSPM cache entry (tiles.bin + assignment.npy)."""
        obj_dir = self._objects_dir / artifact_id[:2] / artifact_id
        if not obj_dir.exists():
            return AbsentCacheEntry(artifact_id=artifact_id)
        tiles_path = obj_dir / "tiles.bin"
        assignment_path = obj_dir / "assignment.npy"
        if tiles_path.is_file() and assignment_path.is_file():
            return PresentCacheEntry(
                artifact_id=artifact_id,
                paths=ContentAddressedStoragePaths(object_dir=obj_dir, data_path=tiles_path),
            )
        return CorruptCacheEntry(
            artifact_id=artifact_id,
            paths=ContentAddressedStoragePaths(object_dir=obj_dir, data_path=tiles_path),
        )

    def _store_compressed(
        self,
        artifact_id: str,
        fingerprint: Fingerprint,
        inputs: "CompressedTensorBuildInputs",
    ) -> Path:
        """Write compact BSPM tiles to the CAS object directory.

        Layout::

            objects/{id[:2]}/{id}/
                tiles.bin        — compact packed tile bytes, DRAM-shuffled order
                assignment.npy   — (tiles_h, tiles_w) int8 tile format codes, DRAM-shuffled order
                metadata.json    — K, N_padded, num_banks, …
                manifest.json    — fingerprint + logical_name
        """
        from models.demos.deepseek_v3_b1.compressed_tensor.compact_io import pack_compact_tiles

        assert isinstance(
            fingerprint.target, CompressedTensorTarget
        ), f"_store_compressed requires CompressedTensorTarget, got {type(fingerprint.target)}"
        target: CompressedTensorTarget = fingerprint.target
        obj_dir = self._objects_dir / artifact_id[:2] / artifact_id
        obj_dir.mkdir(parents=True, exist_ok=True)

        # 1. Compact tile bytes (DRAM-shuffled order, variable length)
        tiles_path = obj_dir / "tiles.bin"
        compact_bytes = pack_compact_tiles(inputs.w, inputs.assignment)
        tiles_path.write_bytes(compact_bytes)

        # 2. Assignment array
        assignment_path = obj_dir / "assignment.npy"
        np.save(str(assignment_path), inputs.assignment.astype(np.int8))

        # 3. Metadata — everything needed to reconstruct the memory config at load time
        tiles_h, tiles_w = inputs.assignment.shape
        metadata_dict = {
            "artifact_id": artifact_id,
            "artifact_kind": "compressed_tensor",
            "K": target.K,
            "N_padded": target.N_padded,
            "num_banks": target.num_banks,
            "tiles_h": tiles_h,
            "tiles_w": tiles_w,
            "compact_bytes": len(compact_bytes),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(obj_dir / "metadata.json", "w") as f:
            json.dump(metadata_dict, f, indent=2, sort_keys=True)

        # 4. Manifest (fingerprint + logical name)
        manifest_dict = {
            "fingerprint": canonical(fingerprint),
            "logical_name": _logical_name_from_fingerprint(fingerprint),
        }
        with open(obj_dir / "manifest.json", "w") as f:
            json.dump(manifest_dict, f, indent=2, sort_keys=True)

        return obj_dir

    def _load_compressed(
        self,
        obj_dir: Path,
        target: "CompressedTensorTarget",
    ) -> "CompressedTensorBuildInputs":
        """Read a compact CAS object and return DRAM-shuffled build inputs.

        Tiles are stored in DRAM-shuffled order (written by :meth:`_store_compressed`
        after :func:`bspm_expert_cache.get_or_create_bspm_expert` applies the shuffle).
        The caller's ``reconstruct`` callback handles device upload.
        """
        from models.demos.deepseek_v3_b1.compressed_tensor.compact_io import unpack_compact_tiles

        assignment = np.load(str(obj_dir / "assignment.npy"))
        compact_bytes = (obj_dir / "tiles.bin").read_bytes()
        w = unpack_compact_tiles(compact_bytes, assignment)  # (K, N_padded) float32, DRAM-shuffled
        return CompressedTensorBuildInputs(w=w, assignment=assignment)

    # ------------------------------------------------------------------
    # SRAM hot-expert compressed-tensor helpers (post-pack disk format)
    # ------------------------------------------------------------------

    _SRAM_SCHEMA_VERSION = 1

    def _sram_compressed_paths(self, artifact_id: str) -> ContentAddressedStoragePaths:
        obj_dir = self._objects_dir / artifact_id[:2] / artifact_id
        return ContentAddressedStoragePaths(object_dir=obj_dir, data_path=obj_dir / "shards.bin")

    def _lookup_sram_compressed(self, artifact_id: str) -> "CacheEntry":
        """Check for a present SRAM-compressed CAS entry (shards.bin + metadata.json)."""
        paths = self._sram_compressed_paths(artifact_id)
        if not paths.object_dir.exists():
            return AbsentCacheEntry(artifact_id=artifact_id)
        if paths.data_path.is_file() and (paths.object_dir / "metadata.json").is_file():
            return PresentCacheEntry(artifact_id=artifact_id, paths=paths)
        return CorruptCacheEntry(artifact_id=artifact_id, paths=paths)

    def _store_sram_compressed(
        self,
        artifact_id: str,
        fingerprint: Fingerprint,
        artifacts: dict,
    ) -> Path:
        """Persist post-pack SRAM CompressedTensor artifacts to a CAS object directory.

        Layout::

            objects/{id[:2]}/{id}/
                shards.bin       — concatenation of all per-device per-shard packed bytes
                                  (no inter-shard padding; offsets recorded in metadata)
                metadata.json    — shape, tile_hw, per-device shard sizes/offsets,
                                  full + per-device assignment arrays (base64 int8)
                manifest.json    — fingerprint + logical name

        The single ``shards.bin`` file keeps the artifact one-blob per cache
        entry — matching the existing fusion-group / tensor cache layout —
        while ``metadata.json`` records the byte offsets the warm-load path
        needs to recover ``{coord: [shard0_bytes, ...]}`` byte-equivalent
        to a fresh cold pack.
        """
        assert isinstance(
            fingerprint.target, SramCompressedTensorTarget
        ), f"_store_sram_compressed requires SramCompressedTensorTarget, got {type(fingerprint.target)}"
        paths = self._sram_compressed_paths(artifact_id)
        paths.object_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Concatenate all per-device per-shard bytes into one blob ---
        # Iterate device coords in a deterministic order so warm-load can
        # rebuild offsets without an explicit "device order" field.
        per_device_meta: list[dict] = []
        cursor = 0
        with open(paths.data_path, "wb") as blob:
            for coord_key, dev in artifacts["per_device"].items():
                shard_sizes = list(int(s) for s in dev["shard_sizes"])
                dev_start = cursor
                for shard_bytes, expected in zip(dev["shard_bytes"], shard_sizes):
                    arr = np.asarray(shard_bytes, dtype=np.uint8).ravel()
                    if expected == 0:
                        if arr.size != 0:
                            raise ValueError(
                                f"SRAM cache write: empty shard expected at coord {coord_key} but got {arr.size} bytes"
                            )
                        continue
                    if arr.size != expected:
                        raise ValueError(
                            f"SRAM cache write: shard byte count mismatch at coord {coord_key} "
                            f"(got {arr.size}, expected {expected})"
                        )
                    blob.write(arr.tobytes())
                    cursor += arr.size
                dev_length = cursor - dev_start
                per_device_meta.append(
                    {
                        "coord": _coord_as_list(coord_key),
                        "shard_sizes": shard_sizes,
                        "data_offset": dev_start,
                        "data_length": dev_length,
                        "assignment_flat_b64": base64.b64encode(
                            np.asarray(dev["assignment_flat"], dtype=np.int8).tobytes()
                        ).decode("ascii"),
                    }
                )

        # --- 2. Metadata sidecar ---
        metadata_dict = {
            "artifact_id": artifact_id,
            "artifact_kind": "sram_compressed_tensor",
            "schema_version": self._SRAM_SCHEMA_VERSION,
            "tensor_shape": list(artifacts["shape"]),
            "tile_hw": int(artifacts["tile_hw"]),
            "per_device_shape": list(artifacts["per_device_shape"]),
            "mesh_shape": list(artifacts["mesh_shape"]) if artifacts["mesh_shape"] is not None else None,
            "assignment_flat_b64": base64.b64encode(
                np.asarray(artifacts["assignment_flat"], dtype=np.int8).tobytes()
            ).decode("ascii"),
            "per_device": per_device_meta,
            "data_size_bytes": cursor,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(paths.object_dir / "metadata.json", "w") as f:
            json.dump(metadata_dict, f, indent=2, sort_keys=True)

        # --- 3. Manifest (fingerprint + logical name) ---
        manifest_dict = {
            "fingerprint": canonical(fingerprint),
            "logical_name": _logical_name_from_fingerprint(fingerprint),
        }
        with open(paths.object_dir / "manifest.json", "w") as f:
            json.dump(manifest_dict, f, indent=2, sort_keys=True)

        return paths.object_dir

    def _load_sram_compressed(self, obj_dir: Path) -> dict:
        """Read a SRAM-compressed CAS object and return artifacts ready for ``from_packed_artifacts``.

        The returned dict mirrors the structure produced by
        :meth:`CompressedTensor.extract_packed_artifacts` so the warm-load
        site can plug it straight into
        :meth:`CompressedTensor.from_packed_artifacts`.
        """
        with open(obj_dir / "metadata.json") as f:
            meta = json.load(f)
        assert (
            meta.get("artifact_kind") == "sram_compressed_tensor"
        ), f"Unexpected artifact_kind {meta.get('artifact_kind')!r} for SRAM cache load at {obj_dir}"
        # Slurp the whole blob — even at 50-100 MB per artifact this is faster
        # (and simpler) than pread'ing per-device slices, and the OS page cache
        # warms the rest of the layer for free.
        blob = (obj_dir / "shards.bin").read_bytes()
        if len(blob) != int(meta["data_size_bytes"]):
            raise ValueError(
                f"SRAM cache load: shards.bin size {len(blob)} != metadata.data_size_bytes {meta['data_size_bytes']}"
            )

        full_assignment = np.frombuffer(base64.b64decode(meta["assignment_flat_b64"]), dtype=np.int8).copy()
        per_device_artifacts: dict = {}
        for dev in meta["per_device"]:
            data_offset = int(dev["data_offset"])
            shard_sizes = list(int(s) for s in dev["shard_sizes"])
            shards: list[np.ndarray] = []
            local_cursor = data_offset
            for sz in shard_sizes:
                if sz == 0:
                    shards.append(np.empty(0, dtype=np.uint8))
                else:
                    shards.append(np.frombuffer(blob, dtype=np.uint8, count=sz, offset=local_cursor).copy())
                    local_cursor += sz
            assert local_cursor - data_offset == int(
                dev["data_length"]
            ), f"SRAM cache load: device data_length mismatch at coord {dev['coord']}"
            assignment = np.frombuffer(base64.b64decode(dev["assignment_flat_b64"]), dtype=np.int8).copy()
            coord = ttnn.MeshCoordinate([int(c) for c in dev["coord"]])
            per_device_artifacts[coord] = {
                "shard_bytes": shards,
                "shard_sizes": shard_sizes,
                "assignment_flat": assignment,
            }

        return {
            "shape": tuple(int(d) for d in meta["tensor_shape"]),
            "tile_hw": int(meta["tile_hw"]),
            "assignment_flat": full_assignment,
            "per_device_shape": tuple(int(d) for d in meta["per_device_shape"]),
            "mesh_shape": tuple(int(d) for d in meta["mesh_shape"]) if meta["mesh_shape"] is not None else None,
            "per_device": per_device_artifacts,
        }

    def get_or_create(
        self,
        fingerprint: Fingerprint,
        device,
        *,
        move_to_device: bool = True,
        preprocess: Callable[[dict[str, torch.Tensor]], dict],
        raw_tensors: Callable[[], dict[str, torch.Tensor]] | dict[str, torch.Tensor],
        reconstruct: Callable[["CompressedTensorBuildInputs", object], "CompressedTensor"] | None = None,
    ) -> "ttnn.Tensor | dict[str, OverlappedTensor] | CompressedTensor":
        """Load from cache or build, then return a device tensor or overlapped views."""
        target = fingerprint.target
        if isinstance(target, SramCompressedTensorTarget):
            raise TypeError(
                "TensorCache.get_or_create does not handle SramCompressedTensorTarget; "
                "use weights.cache.get_or_create_sram_compressed_expert() instead "
                "(post-pack-byte cache writes/reads diverge from the build-inputs flow)."
            )
        if not isinstance(target, (TensorTarget, FusionGroupSpec, CompressedTensorTarget)):
            raise TypeError(
                f"TensorCache.get_or_create requires TensorTarget, FusionGroupSpec, or CompressedTensorTarget, got {type(target)}"
            )

        artifact_id = compute_artifact_id(fingerprint)
        logical = _logical_name_from_fingerprint(fingerprint)

        # --- CompressedTensorTarget: use compact tiles.bin layout ---
        if isinstance(target, CompressedTensorTarget):
            if reconstruct is None:
                raise TypeError(
                    "TensorCache.get_or_create with CompressedTensorTarget requires a 'reconstruct' callback; "
                    "use get_or_create_bspm_expert() instead of calling get_or_create() directly."
                )
            entry = self._lookup_compressed(artifact_id)
            if isinstance(entry, PresentCacheEntry):
                logger.debug("Cache hit (compressed) for {} ({})", logical, artifact_id[:12])
                inputs = self._load_compressed(entry.paths.object_dir, target)
                return reconstruct(inputs, device if move_to_device else None)
            if isinstance(entry, CorruptCacheEntry):
                logger.warning("Corrupt compressed cache entry for {} ({}), rebuilding", logical, artifact_id[:12])
                shutil.rmtree(entry.paths.object_dir, ignore_errors=True)
            t0 = time.perf_counter()
            tensors = raw_tensors() if callable(raw_tensors) else raw_tensors
            preprocessed = preprocess(tensors)
            inputs = preprocessed[target.name]
            self._store_compressed(artifact_id, fingerprint, inputs)
            elapsed = time.perf_counter() - t0
            logger.info(
                "Cache miss (compressed) for {} resolved in {:.3f}s, stored as {}",
                logical,
                elapsed,
                artifact_id[:12],
            )
            return reconstruct(inputs, device if move_to_device else None)

        # --- TensorTarget / FusionGroupSpec: original path ---
        entry = self._lookup(artifact_id)

        if isinstance(entry, PresentCacheEntry):
            if isinstance(target, TensorTarget):
                return self._load(entry.paths, device, move_to_device=move_to_device)
            meta_path = entry.paths.object_dir / "metadata.json"
            if meta_path.is_file():
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("views"):
                    return self._load_fused(entry.paths, device, move_to_device=move_to_device, meta=meta)
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
            return self._load(paths, device, move_to_device=move_to_device)

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
        return self._load_fused(paths, device, move_to_device=move_to_device)


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
        move_to_device: bool | None = None,
        preprocess: Callable[[dict[str, torch.Tensor]], dict],
        raw_tensors: Callable[[], dict[str, torch.Tensor]] | dict[str, torch.Tensor],
        reconstruct: Callable[["CompressedTensorBuildInputs", object], "CompressedTensor"] | None = None,
    ) -> "ttnn.Tensor | dict[str, OverlappedTensor] | CompressedTensor":
        effective_move_to_device = self._move_to_device if move_to_device is None else move_to_device
        target = fingerprint.target
        if isinstance(target, SramCompressedTensorTarget):
            raise TypeError(
                "EphemeralTensorCache.get_or_create does not handle SramCompressedTensorTarget; "
                "use weights.cache.get_or_create_sram_compressed_expert() instead."
            )
        if not isinstance(target, (TensorTarget, FusionGroupSpec, CompressedTensorTarget)):
            raise TypeError(
                f"EphemeralTensorCache.get_or_create requires TensorTarget, FusionGroupSpec, or CompressedTensorTarget, "
                f"got {type(target)}"
            )

        tensors = raw_tensors() if callable(raw_tensors) else raw_tensors
        preprocessed = preprocess(tensors)

        if isinstance(target, CompressedTensorTarget):
            if reconstruct is None:
                raise TypeError(
                    "EphemeralTensorCache.get_or_create with CompressedTensorTarget requires a 'reconstruct' callback; "
                    "use get_or_create_bspm_expert() instead of calling get_or_create() directly."
                )
            inputs: CompressedTensorBuildInputs = preprocessed[target.name]
            return reconstruct(inputs, device if effective_move_to_device else None)

        if isinstance(target, TensorTarget):
            torch_tensor = preprocessed[target.name]
            from_torch_kwargs: dict = {
                "dtype": target.dtype,
                "layout": target.layout,
                "device": device if effective_move_to_device else None,
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
            move_to_device=effective_move_to_device,
        )
        return views
