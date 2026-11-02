# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Live construction of the ordinary Qwen3.8 decode on an already-open mesh.

The CPU preparation binds the checkpoint, the cache roots and the runtime identity into the builder provenance; the
live construction creates the exact :class:`Qwen38TTNNBuilder` on the caller's ``(1, 4)`` mesh and requires all 48
backbone BF4 records plus MTP layer 0, which ``Qwen38BF4Cache`` converts on the first start.  This module never
discovers, opens, closes or resets a device and never configures fabric.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    CHECKPOINT_FILE_MANIFEST_SHA256,
    INDEX_SHA256,
    Qwen38Checkpoint,
)
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256, Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import Qwen38BF4Cache
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    BACKBONE_LAYERS,
    RESIDENT_MAX_QSA_CACHE_CAPACITY,
    RESIDENT_QSA_CACHE_CAPACITIES,
    Qwen38BuildProvenance,
    Qwen38CacheRoots,
    Qwen38TTNNBuilder,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import Qwen38MeshContract
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    PINNED_CHECKPOINT_REVISION,
    PINNED_TENSOR_MANIFEST_SHA256,
)

EXPECTED_MESH_SHAPE = (1, 4)
EXPECTED_SLOTS = tuple(("backbone", index) for index in range(BACKBONE_LAYERS)) + (("mtp", 0),)

Marker = Callable[[str], None]
FileSignature = tuple[int, int, int, int, int]


def _require_hex(value: Any, length: int, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase {length}-hex")
    return value


class LiveDecodeConstructionError(RuntimeError):
    """The live construction differs from its prepared inputs."""


def _mark(marker: Marker | None, phase: str) -> None:
    if marker is not None:
        marker(phase)


def _signature(metadata: os.stat_result) -> FileSignature:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stream_sha256(path: Path, *, expected_bytes: int | None = None) -> str:
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (expected_bytes is not None and before.st_size != expected_bytes):
            raise LiveDecodeConstructionError(f"hash input identity/size differs: {path}")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 8 << 20):
            digest.update(block)
        after = os.fstat(descriptor)
        if _signature(before) != _signature(after) or _signature(after) != _signature(path.lstat()):
            raise LiveDecodeConstructionError(f"hash input changed during verification: {path}")
        return digest.hexdigest()
    except OSError as error:
        raise LiveDecodeConstructionError(f"cannot hash pinned file: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True)
class Qwen38PreparedLiveDecode:
    """All CPU-verified inputs needed at the already-open mesh boundary."""

    checkpoint: Qwen38Checkpoint
    placement: Qwen38Placement
    mesh_contract: Qwen38MeshContract
    provenance: Qwen38BuildProvenance
    cache_roots: Qwen38CacheRoots
    runtime_extension: Path
    runtime_sha256: str
    tt_metal_sha: str
    qsa_cache_capacity: int
    expert_residency: str = "resident"

    @property
    def runtime_profile(self) -> str:
        return f"resident-c{self.qsa_cache_capacity}"

    def summary(self) -> dict[str, Any]:
        return {
            "checkpoint_root": str(self.checkpoint.root),
            "tt_metal_sha": self.tt_metal_sha,
            "runtime_extension": str(self.runtime_extension),
            "runtime_sha256": self.runtime_sha256,
            "runtime_profile": self.runtime_profile,
            "expert_residency": self.expert_residency,
            "qsa_cache_capacity": self.qsa_cache_capacity,
            "builder_provenance_key": self.provenance.key,
            "mesh_shape": list(self.mesh_contract.mesh_shape),
            "physical_ids": list(self.mesh_contract.physical_ids),
        }


def prepare_live_decode(
    *,
    checkpoint_root: Path,
    component_cache_root: Path,
    routed_bf4_scratch_root: Path,
    model_io_cache_root: Path,
    tt_metal_sha: str,
    runtime_extension: Path,
    runtime_sha256: str,
    allocated_context: int = RESIDENT_MAX_QSA_CACHE_CAPACITY,
    physical_ids: tuple[int, int, int, int],
    marker: Marker | None = None,
) -> Qwen38PreparedLiveDecode:
    """Bind every software input before the caller opens its mesh.

    ``tt_metal_sha`` / ``runtime_extension`` / ``runtime_sha256`` are the admitted runtime identity (the checkout's
    head and its extension); the extension must be the one this interpreter loaded.  ``physical_ids`` is the order
    the host's mesh opens in (the hardware profile's route).  The BF4 cache under ``routed_bf4_scratch_root`` is
    filled on the first start.
    """

    _require_hex(tt_metal_sha, 40, "tt_metal_sha")
    _require_hex(runtime_sha256, 64, "runtime_sha256")
    if (
        not isinstance(physical_ids, tuple)
        or len(physical_ids) != 4
        or any(isinstance(value, bool) or type(value) is not int or value < 0 for value in physical_ids)
        or len(set(physical_ids)) != 4
    ):
        raise ValueError(f"physical_ids must be four distinct non-negative ints, got {physical_ids!r}")
    if allocated_context not in RESIDENT_QSA_CACHE_CAPACITIES:
        raise ValueError(f"allocated_context must be one of {RESIDENT_QSA_CACHE_CAPACITIES}, got {allocated_context!r}")

    runtime_path = Path(runtime_extension).resolve(strict=True)
    loaded_extension = Path(ttnn._ttnn.__file__).resolve(strict=True)
    if runtime_path != loaded_extension:
        raise LiveDecodeConstructionError(f"runtime extension {runtime_path} is not the loaded one {loaded_extension}")
    _mark(marker, "before-runtime-sha256")
    actual_runtime_sha = _stream_sha256(runtime_path)
    _mark(marker, "after-runtime-sha256")
    if actual_runtime_sha != runtime_sha256:
        raise LiveDecodeConstructionError(f"runtime digest differs: {actual_runtime_sha} != {runtime_sha256}")

    checkpoint = Qwen38Checkpoint(Path(checkpoint_root).resolve(strict=True))
    placement = Qwen38Placement(checkpoint.config, mesh_shape=EXPECTED_MESH_SHAPE, physical_ids=physical_ids)
    mesh_contract = Qwen38MeshContract(physical_ids)
    provenance = Qwen38BuildProvenance(
        checkpoint_revision=PINNED_CHECKPOINT_REVISION,
        checkpoint_index_sha256=INDEX_SHA256,
        checkpoint_config_sha256=CONFIG_SHA256,
        checkpoint_file_manifest_sha256=CHECKPOINT_FILE_MANIFEST_SHA256,
        checkpoint_hash_manifest_sha256=PINNED_TENSOR_MANIFEST_SHA256,
        tt_metal_sha=tt_metal_sha,
        ttnn_runtime_sha256=runtime_sha256,
    )
    cache_roots = Qwen38CacheRoots(
        component_weights=component_cache_root,
        routed_bf4=routed_bf4_scratch_root,
        model_io=model_io_cache_root,
    )
    return Qwen38PreparedLiveDecode(
        checkpoint=checkpoint,
        placement=placement,
        mesh_contract=mesh_contract,
        provenance=provenance,
        cache_roots=cache_roots,
        runtime_extension=runtime_path,
        runtime_sha256=runtime_sha256,
        tt_metal_sha=tt_metal_sha,
        qsa_cache_capacity=allocated_context,
    )


@dataclass(frozen=True)
class Qwen38LiveDecodeConstruction:
    """A real live builder stopped before target/state/inference construction."""

    prepared: Qwen38PreparedLiveDecode
    builder: Qwen38TTNNBuilder
    bf4_cache: Qwen38BF4Cache
    expert_streamer: Any
    staged_bf4_layers: tuple[tuple[str, int], ...] = ()
    bf4_admission: dict[str, Any] | None = None  # the one-expert byte check of the cache, if it held a layer


def missing_bf4_layers(builder: Qwen38TTNNBuilder) -> tuple[tuple[str, int], ...]:
    """The slots the builder's BF4 cache does not hold yet (each ``verify_layer`` hashes a present record)."""

    return tuple(slot for slot in EXPECTED_SLOTS if builder.bf4_cache.verify_layer(*slot) is None)


def construct_live_decode(
    prepared: Qwen38PreparedLiveDecode,
    *,
    mesh_device,
    collective_topology,
    marker: Marker | None = None,
    bf4_stage_limit: int | None = None,
    require_complete: bool = True,
) -> Qwen38LiveDecodeConstruction:
    """Construct and attach the real live builder, then stop before target build.

    Missing BF4 layers are converted from the checkpoint (at most ``bf4_stage_limit`` of them this run, for hosts that
    bound a job's wall time) and the construction refuses while any is still missing (``require_complete``; a
    prepare-only run returns the builder as is).
    """

    if type(prepared) is not Qwen38PreparedLiveDecode:
        raise TypeError("prepared must be Qwen38PreparedLiveDecode")
    prepared.mesh_contract.validate_mesh(mesh_device)
    loaded_extension = Path(ttnn._ttnn.__file__).resolve(strict=True)
    if loaded_extension != prepared.runtime_extension or _stream_sha256(loaded_extension) != prepared.runtime_sha256:
        raise LiveDecodeConstructionError("loaded TTNN extension differs from CPU-prepared runtime identity")

    _mark(marker, "before-live-builder-construction")
    builder = Qwen38TTNNBuilder(
        checkpoint=prepared.checkpoint,
        placement=prepared.placement,
        mesh_device=mesh_device,
        mesh_contract=prepared.mesh_contract,
        provenance=prepared.provenance,
        cache_roots=prepared.cache_roots,
        collective_topology=collective_topology,
        expert_residency=prepared.expert_residency,
        qsa_cache_capacity=prepared.qsa_cache_capacity,
    )
    _mark(marker, "after-live-builder-construction")
    bf4_cache = builder.bf4_cache
    staged: list[tuple[str, int]] = []
    admission = None
    _mark(marker, "before-bf4-cache-inventory")
    missing = missing_bf4_layers(builder)
    _mark(marker, f"after-bf4-cache-inventory-missing-{len(missing)}")
    # The cache pins the converter's sources, not the runtime: prove the cached bytes are this converter's output
    # on one expert of the first cached layer before anything is loaded or converted next to them.
    cached_slot = next((slot for slot in EXPECTED_SLOTS if slot not in missing), None)
    if cached_slot is not None:
        _mark(marker, f"before-bf4-cache-admission-{cached_slot[0]}-{cached_slot[1]:02d}")
        admission = bf4_cache.admit_converted_bytes(
            prepared.checkpoint, prepared.placement, namespace=cached_slot[0], layer_index=cached_slot[1]
        )
        _mark(marker, f"after-bf4-cache-admission-expert-{admission['expert']}-{admission['seconds']}s")
    if missing:
        for namespace, layer_index in missing[: len(missing) if bf4_stage_limit is None else bf4_stage_limit]:
            _mark(marker, f"before-bf4-stage-{namespace}-{layer_index:02d}")
            builder.stage_bf4_layer(namespace=namespace, layer_index=layer_index)
            ttnn.synchronize_device(mesh_device)
            staged.append((namespace, layer_index))
            _mark(marker, f"after-bf4-stage-{namespace}-{layer_index:02d}")
        missing = missing_bf4_layers(builder)
    if missing and require_complete:
        raise LiveDecodeConstructionError(
            f"the BF4 cache under {bf4_cache.root} lacks {len(missing)} of {len(EXPECTED_SLOTS)} layers "
            f"({', '.join(f'{namespace}:{index}' for namespace, index in missing[:8])}"
            f"{'...' if len(missing) > 8 else ''}); {len(staged)} staged this run"
        )

    if require_complete:
        _mark(marker, "before-bf4-record-attachment")
        backbone = builder.require_complete_backbone_bf4()
        mtp = builder.require_bf4_layers(namespace="mtp", layer_indices=(0,))
        if len(backbone) != BACKBONE_LAYERS or len(mtp) != 1:
            raise LiveDecodeConstructionError(
                f"real builder attached {len(backbone)} backbone and {len(mtp)} MTP records, expected 48/1"
            )
        _mark(marker, "after-bf4-record-attachment")
    return Qwen38LiveDecodeConstruction(
        prepared=prepared,
        builder=builder,
        bf4_cache=bf4_cache,
        expert_streamer=builder.expert_streamer,
        staged_bf4_layers=tuple(staged),
        bf4_admission=admission,
    )


__all__ = [
    "LiveDecodeConstructionError",
    "Qwen38LiveDecodeConstruction",
    "Qwen38PreparedLiveDecode",
    "construct_live_decode",
    "missing_bf4_layers",
    "prepare_live_decode",
]
