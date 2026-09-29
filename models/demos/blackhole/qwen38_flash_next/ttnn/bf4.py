# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""BF4_B conversion, cache qualification, and upload for routed experts.

The fused ``moe_compute`` kernel consumes a ring-specific packed layout, not a
plain BF4 quantization of checkpoint matrices.  Cache identity therefore binds
the checkpoint, the converter's sources (this module, the ``moe_compute`` layout
packer and tt-metal's BFP4 packer), exact 1x4 topology, Blackhole DRAM ring
size, expert ranges, packer layout version, and output file hashes.  The
tt-metal revision is not part of it: the packed bytes do not depend on the rest
of the runtime, so a rebuilt runtime keeps the cache.  Every start re-packs one
routed expert of one cached layer from the checkpoint and compares the bytes
(``Qwen38BF4Cache.admit_converted_bytes``).

Conversion is deliberately performed one layer at a time.  It is weight
conversion, not CPU inference.  The first qualified device run creates the
multi-device cache with an explicit two-dimensional mesh mapper; later runs
load that exact topology and fail closed if any qualifier differs.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import operator
import os
import socket
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
import ttnn.experimental.moe_compute_utils as moe_compute_utils
from ttnn.experimental.moe_compute_utils import (
    W2_TILES_PER_A2A_ITER_W,
    _block_tiles_h,
    _shard_tiles,
    _tiles_per_txn,
    _w0_w1_compact_layout,
    _w2_core_blocks_per_expert,
    _w2_last_a2a_iter_half,
    _w2_num_a2a_iters,
    _w2_shard_tiles,
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w2_tensor_for_moe_compute,
)

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    CHECKPOINT_FILE_MANIFEST_SHA256,
    CHECKPOINT_TENSOR_MANIFEST_SHA256,
    PINNED_CHECKPOINT_REVISION,
    Qwen38Checkpoint,
)
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256, Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tt.moe import Qwen38MoEWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    MESH_SHAPE,
    Qwen38MeshContract,
    Qwen38TensorBackingIdentity,
    TensorPlacement,
    qwen38_tensor_backing_identity,
)

FORMAT_VERSION = 3
# Adopted in place by the first format-3 run: 1 was keyed by the tt-metal revision, 2 by the first die's DRAM
# bank-to-worker coordinates.
LEGACY_FORMAT_VERSIONS = (1, 2)
PACKER = "ttnn.experimental.moe_compute_utils"
DTYPE = "BFLOAT4_B"
LAYOUT = "TILE"
# The packed expert layout the cache holds, part of the identity: 1 = the per-core stride layout (every ring core
# stored the uniform even column stride, K padded to 7-tile blocks); 2 = moe_compute's compact owned-column layout
# with the per-shape DRAM transaction size (20-tile transactions and a half-width last W2 iteration on the 8-bank
# ring, 14-tile transactions on the 7-bank ring).
LAYOUT_VERSION = 2
BF4_TILE_BYTES = 576
BACKBONE_LAYERS = 48
MTP_LAYERS = 1
BLACKHOLE_RING_SIZES = (7, 8)
CONVERSION_LOCK_TIMEOUT_SECONDS = 60.0
TENSORBIN_HEADER_PREFIX_BYTES = 8
TENSORBIN_HEADER_ALIGNMENT = 8
MAX_TENSORBIN_HEADER_BYTES = 16 << 20
REPO_ROOT = Path(__file__).resolve().parents[5]
# The sources that determine the packed bytes: this module (the layer walk), the moe_compute layout packer, and
# tt-metal's BFP4 tile packer.  The cache identity pins their digests instead of the tt-metal revision.
CONVERTER_SOURCES = (
    Path(__file__).resolve(),
    Path(moe_compute_utils.__file__).resolve(),
    REPO_ROOT / "tt_metal/impl/data_format/bfloat4.cpp",
    REPO_ROOT / "tt_metal/impl/data_format/blockfloat_common.hpp",
    REPO_ROOT / "tt_metal/impl/data_format/blockfloat_common.cpp",
)
ADMISSION_EXPERT = 511  # the last routed expert: the last mesh shard, the last block of every ring bank


def validate_bf4_layer_request(namespace: str, layer_index: int) -> None:
    """Validate one exact routed-expert cache key without depending on a cache adapter."""

    if type(namespace) is not str or namespace not in {"backbone", "mtp"}:
        raise ValueError(f"unsupported MoE namespace {namespace!r}")
    if type(layer_index) is not int:
        raise TypeError(f"layer index must be an exact integer, got {layer_index!r}")
    layer_count = BACKBONE_LAYERS if namespace == "backbone" else MTP_LAYERS
    if not 0 <= layer_index < layer_count:
        raise ValueError(f"{namespace} layer index must be in [0,{layer_count}), got {layer_index}")


def bf4_converter_source_identity() -> tuple[tuple[str, str], ...]:
    """``(repository-relative path, sha256)`` of every converter source, in path order."""

    identity = []
    for path in CONVERTER_SOURCES:
        try:
            relative = path.relative_to(REPO_ROOT).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError) as error:
            raise RuntimeError(f"BF4 converter source is unavailable under {REPO_ROOT}: {path}") from error
        identity.append((relative, digest))
    return tuple(sorted(identity))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_fd(descriptor: int, chunk_size: int = 8 << 20) -> str:
    """Hash an open artifact without changing its shared file offset."""

    digest = hashlib.sha256()
    offset = 0
    while chunk := os.pread(descriptor, chunk_size, offset):
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _tensorbin_path(base: Path) -> Path:
    return Path(f"{base}_dtype_{DTYPE}_layout_{LAYOUT}.tensorbin")


def _artifact_stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    """Detect post-verification replacement/writes without re-hashing payloads."""

    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"BF4 artifact must be a regular file, got {path}")
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _artifact_fd_signature(descriptor: int) -> tuple[int, int, int, int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("BF4 artifact descriptor target must be a regular file")
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


@dataclass(frozen=True)
class BF4TensorCleanupOutcome:
    """One input slot's deallocation result, retaining uncertain ownership."""

    slot: int
    tensor: Any | None
    release_attempted: bool
    released: bool
    release_error: BaseException | None


def _tensor_identity(tensor: Any) -> Qwen38TensorBackingIdentity:
    """Return exact per-card backing buffers; wrapper/graph IDs are insufficient."""

    try:
        return qwen38_tensor_backing_identity(tensor)
    except RuntimeError as error:
        raise RuntimeError("BF4 tensor lacks an exact 1x4 device-backing identity") from error


class BF4CleanupError(RuntimeError):
    """One BF4 operation failed and one or more owned tensors did not release cleanly."""

    def __init__(
        self,
        context: str,
        *,
        primary_error: BaseException | None,
        cleanup_errors: tuple[BaseException, ...],
        tensor_cleanup_outcomes: tuple[BF4TensorCleanupOutcome, ...] = (),
    ) -> None:
        if not cleanup_errors:
            raise ValueError("BF4CleanupError requires at least one cleanup failure")
        primary = "none" if primary_error is None else f"{type(primary_error).__name__}: {primary_error}"
        cleanup = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
        super().__init__(f"{context}; primary={primary}; tensor cleanup failure(s)={cleanup}")
        self.primary_error = primary_error
        self.cleanup_errors = cleanup_errors
        self.tensor_cleanup_outcomes = tensor_cleanup_outcomes
        unreleased: list[Any] = []
        identities: set[Qwen38TensorBackingIdentity] = set()
        for outcome in tensor_cleanup_outcomes:
            tensor = outcome.tensor
            if tensor is None or outcome.released:
                continue
            try:
                identity = _tensor_identity(tensor)
            except RuntimeError:
                pass
            else:
                if identity in identities:
                    continue
                identities.add(identity)
            unreleased.append(tensor)
        self.unreleased_tensors = tuple(unreleased)


def _deallocate_all(tensors: tuple[Any, ...]) -> tuple[BF4TensorCleanupOutcome, ...]:
    """Try every unique task-owned release and retain its exact input slot."""

    outcomes: list[BF4TensorCleanupOutcome] = []
    prior: dict[Qwen38TensorBackingIdentity, BF4TensorCleanupOutcome] = {}
    for slot, tensor in enumerate(tensors):
        if tensor is None:
            outcomes.append(BF4TensorCleanupOutcome(slot, None, False, False, None))
            continue
        try:
            identity = _tensor_identity(tensor)
        except RuntimeError as error:
            outcomes.append(BF4TensorCleanupOutcome(slot, tensor, False, False, error))
            continue
        if identity in prior:
            original = prior[identity]
            outcomes.append(
                BF4TensorCleanupOutcome(
                    slot,
                    tensor if not original.released else None,
                    original.release_attempted,
                    original.released,
                    original.release_error,
                )
            )
            continue
        try:
            ttnn.deallocate(tensor)
        except BaseException as error:  # cleanup must continue across every owned allocation
            outcome = BF4TensorCleanupOutcome(slot, tensor, True, False, error)
        else:
            outcome = BF4TensorCleanupOutcome(slot, None, True, True, None)
        outcomes.append(outcome)
        prior[identity] = outcome
    return tuple(outcomes)


def _raise_if_cleanup_failed(
    tensors: tuple[Any, ...],
    *,
    context: str,
    primary_error: BaseException | None,
) -> None:
    outcomes = _deallocate_all(tensors)
    cleanup_errors = tuple(
        {
            id(outcome.release_error): outcome.release_error
            for outcome in outcomes
            if outcome.release_error is not None
        }.values()
    )
    if cleanup_errors:
        aggregate = BF4CleanupError(
            context,
            primary_error=primary_error,
            cleanup_errors=cleanup_errors,
            tensor_cleanup_outcomes=outcomes,
        )
        if primary_error is not None:
            raise aggregate from primary_error
        raise aggregate from cleanup_errors[0]


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _json_normalized(value: Any) -> Any:
    """Return the exact representation JSON will preserve on disk."""

    return json.loads(json.dumps(value, sort_keys=True))


def _same_exact_typed_tree(actual: Any, expected: Any) -> bool:
    """Compare a JSON-shaped value without bool/int or int/float aliases."""

    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return set(actual) == set(expected) and all(
            _same_exact_typed_tree(actual[key], expected[key]) for key in expected
        )
    if type(actual) in {list, tuple}:
        return len(actual) == len(expected) and all(
            _same_exact_typed_tree(actual_item, expected_item) for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def _native_integer_shape(shape: Any, *, label: str) -> tuple[int, ...]:
    """Normalize integer-like dimensions while rejecting bool/float/string aliases."""

    try:
        dimensions = tuple(shape)
    except TypeError as error:
        raise RuntimeError(f"{label} shape is not iterable") from error
    normalized = []
    for dimension in dimensions:
        if isinstance(dimension, bool):
            raise RuntimeError(f"{label} shape contains a Boolean dimension")
        try:
            value = operator.index(dimension)
        except TypeError as error:
            raise RuntimeError(f"{label} shape contains a non-integer dimension: {dimension!r}") from error
        if type(value) is not int:
            raise RuntimeError(f"{label} shape did not normalize to exact Python integers")
        normalized.append(value)
    return tuple(normalized)


@contextmanager
def _exclusive_file_lock(path: Path, *, timeout_seconds: float = CONVERSION_LOCK_TIMEOUT_SECONDS):
    """Hold a bounded process lock around one cache publication transaction."""

    if timeout_seconds <= 0:
        raise ValueError("lock timeout must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring BF4 cache lock {path}")
                time.sleep(0.25)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _core_xy(core) -> tuple[int, int]:
    return int(core.x), int(core.y)


def dram_bank_ring_order(signature: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    """The DRAM bank ids in ring position order: the banks sorted by their worker core's ``(y, x)``, descending.

    The sort of ``get_weight_core_shard_maps`` (the moe_compute layout packer), which keeps only the resulting bank
    order; ``tests/test_ttnn_bf4_static.py`` pins the two against each other.  ``signature[bank]`` is the worker
    ``(x, y)`` that serves ``bank``.
    """

    return tuple(sorted(range(len(signature)), key=lambda bank: (signature[bank][1], signature[bank][0]), reverse=True))


def qualify_live_bf4_ring(mesh_device) -> tuple[int, ...]:
    """Require one cache-compatible DRAM ring on all four devices; return its bank ids in ring order.

    ``moe_compute`` builds every die's program from the MeshDevice-level bank-to-worker assignment, which is the
    reference (first) device's: the packed shards are laid out by ring position, the placement grid is bank ids, and
    the packed bytes never see worker coordinates.  Dies harvested differently (a QuietBox 2 was observed with one
    die serving its banks from worker column 5, the other three from column 6) are therefore compatible when every
    die has the same bank count and the same bank order: the odd die's ring workers sit one column from their banks
    and read the same bank ids in the same order (adjacency lost there, not correctness).  The public mesh-level
    query returns only the first device's assignment, so this guard queries each device; dies that differ in ring
    size or bank order are refused with their raw coordinates in the message.  The result is the
    :class:`BF4CacheIdentity` field.
    """

    physical_ids = tuple(int(item) for item in mesh_device.get_device_ids())
    if len(physical_ids) != 4:
        raise RuntimeError(f"BF4 conversion requires four local devices, got {len(physical_ids)}")
    if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
        raise RuntimeError("BF4 conversion requires a Blackhole mesh")

    signatures: list[tuple[tuple[int, int], ...]] = []
    for index in range(4):
        coordinate = ttnn.MeshCoordinate(0, index)
        coordinate_device_id = int(mesh_device.get_device_id(coordinate))
        if coordinate_device_id != physical_ids[index]:
            raise RuntimeError(
                f"mesh coordinate {coordinate} reports physical ID {coordinate_device_id}, "
                f"expected {physical_ids[index]}"
            )
        assignment = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment_at_mesh_coordinate(
            mesh_device, ttnn.NOC.RISCV_0_default, coordinate
        )
        signature = tuple(_core_xy(core) for core in assignment)
        if len(signature) not in BLACKHOLE_RING_SIZES or len(set(signature)) != len(signature):
            raise RuntimeError(f"physical device {physical_ids[index]} has unsupported DRAM worker order {signature}")
        signatures.append(signature)

    if len({len(signature) for signature in signatures}) != 1:
        details = {physical_ids[index]: len(signature) for index, signature in enumerate(signatures)}
        raise RuntimeError(f"mixed Blackhole DRAM ring sizes are not cache-compatible: {details}")
    ring_orders = [dram_bank_ring_order(signature) for signature in signatures]
    if len(set(ring_orders)) != 1:
        details = {physical_ids[index]: (signature, ring_orders[index]) for index, signature in enumerate(signatures)}
        raise RuntimeError(f"mixed Blackhole DRAM ring orders are not cache-compatible: {details}")
    return ring_orders[0]


@dataclass(frozen=True)
class BF4Artifact:
    name: str
    relative_path: str
    sha256: str
    bytes: int
    logical_shape: tuple[int, ...]
    dtype: str = DTYPE
    layout: str = LAYOUT


@dataclass(frozen=True)
class _VerifiedBF4ArtifactFD:
    descriptor: int
    proc_path: Path
    signature: tuple[int, int, int, int, int]


def _retained_artifact_identity(descriptor: int, path: Path) -> tuple[int, int, int, int, int]:
    """Bind an open descriptor to the still-canonical cache pathname."""

    descriptor_signature = _artifact_fd_signature(descriptor)
    pathname_signature = _artifact_stat_signature(path)
    proc_path = Path(f"/proc/self/fd/{descriptor}")
    descriptor_target = os.readlink(proc_path)
    if descriptor_signature != pathname_signature or descriptor_target != str(path):
        raise RuntimeError(f"BF4 artifact descriptor and canonical pathname identity differ: {path}")
    return descriptor_signature


def _pread_exact(descriptor: int, size: int, offset: int) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        try:
            block = os.pread(descriptor, size - len(payload), offset + len(payload))
        except InterruptedError:
            continue
        except OSError as error:
            raise RuntimeError("BF4 tensorbin header pread failed") from error
        if not block:
            break
        payload.extend(block)
    if len(payload) != size:
        raise RuntimeError(f"BF4 tensorbin is truncated before its {size}-byte header prefix: got {len(payload)} bytes")
    return bytes(payload)


def _validate_tensorbin_payload_fd(
    descriptor: int,
    *,
    signature: tuple[int, int, int, int, int],
    expected_payload_bytes: int,
) -> int:
    """Bind the serialized header and exact data region to one retained FD."""

    if type(expected_payload_bytes) is not int or expected_payload_bytes <= 0:
        raise RuntimeError("BF4 tensorbin expected payload byte count is invalid")
    before = _artifact_fd_signature(descriptor)
    if before != signature:
        raise RuntimeError("BF4 tensorbin identity changed before header validation")
    prefix = _pread_exact(descriptor, TENSORBIN_HEADER_PREFIX_BYTES, 0)
    header_size = int.from_bytes(prefix, byteorder="little", signed=False)
    file_size = before[2]
    if (
        header_size == 0
        or header_size > MAX_TENSORBIN_HEADER_BYTES
        or header_size % TENSORBIN_HEADER_ALIGNMENT
        or TENSORBIN_HEADER_PREFIX_BYTES + header_size > file_size
    ):
        raise RuntimeError(f"BF4 tensorbin header size/alignment is invalid: header={header_size}, file={file_size}")
    payload_bytes = file_size - TENSORBIN_HEADER_PREFIX_BYTES - header_size
    if payload_bytes != expected_payload_bytes:
        raise RuntimeError(
            f"BF4 tensorbin payload byte count {payload_bytes} differs from canonical {expected_payload_bytes}"
        )
    if _artifact_fd_signature(descriptor) != before:
        raise RuntimeError("BF4 tensorbin identity changed during header validation")
    return header_size


@contextmanager
def _verified_artifact_fd(
    path: Path,
    artifact: BF4Artifact,
    *,
    expected_signature: tuple[int, int, int, int, int] | None = None,
):
    """Retain one O_NOFOLLOW descriptor, hashing only to establish session trust."""

    descriptor = None
    try:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            if descriptor == 0:
                positive_descriptor = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 3)
                os.close(descriptor)
                descriptor = positive_descriptor
            before = _retained_artifact_identity(descriptor, path)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(f"BF4 artifact is unavailable or invalid: {path}") from error
        if before[2] != artifact.bytes:
            raise RuntimeError(f"BF4 artifact failed size validation: {path}")
        _validate_tensorbin_payload_fd(
            descriptor,
            signature=before,
            expected_payload_bytes=_packed_payload_bytes(artifact.logical_shape),
        )
        if expected_signature is None:
            digest = _sha256_fd(descriptor)
            verified_signature = _retained_artifact_identity(descriptor, path)
            if verified_signature != before:
                raise RuntimeError(f"BF4 artifact changed while its descriptor was being verified: {path}")
            if digest != artifact.sha256:
                raise RuntimeError(f"BF4 artifact failed hash validation: {path}")
        else:
            if before != expected_signature:
                raise RuntimeError(f"BF4 artifact differs from its session-verified identity: {path}")
            verified_signature = before

        verified = _VerifiedBF4ArtifactFD(
            descriptor=descriptor,
            proc_path=Path(f"/proc/self/fd/{descriptor}"),
            signature=verified_signature,
        )
        try:
            yield verified
        finally:
            try:
                retained = _retained_artifact_identity(descriptor, path)
            except (OSError, RuntimeError) as error:
                raise RuntimeError(
                    f"BF4 artifact identity changed while its descriptor was retained: {path}"
                ) from error
            if retained != verified_signature:
                raise RuntimeError(f"BF4 artifact changed while its descriptor was retained: {path}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True)
class BF4LayerRecord:
    namespace: str
    layer_index: int
    expert_ranges: tuple[tuple[int, int], ...]
    ring_size: int
    w0_w1: BF4Artifact
    w2: BF4Artifact


@dataclass(frozen=True)
class BF4CacheIdentity:
    checkpoint_revision: str
    checkpoint_config_sha256: str
    checkpoint_file_manifest_sha256: str
    checkpoint_hash_manifest_sha256: str
    converter_sources: tuple[tuple[str, str], ...]
    mesh_shape: tuple[int, int]
    physical_ids: tuple[int, int, int, int]
    ring_size: int
    dram_bank_ring_order: tuple[int, ...]
    hidden_size: int = 2560
    intermediate_size: int = 640
    routed_experts: int = 512
    experts_per_device: int = 128
    dtype: str = DTYPE
    packer: str = PACKER
    format_version: int = FORMAT_VERSION
    layout_version: int = LAYOUT_VERSION

    def __post_init__(self) -> None:
        string_fields = (
            "checkpoint_revision",
            "checkpoint_config_sha256",
            "checkpoint_file_manifest_sha256",
            "checkpoint_hash_manifest_sha256",
            "dtype",
            "packer",
        )
        if any(type(getattr(self, name)) is not str for name in string_fields):
            raise ValueError("BF4 cache identity string fields must be exact strings")
        integer_fields = (
            "ring_size",
            "hidden_size",
            "intermediate_size",
            "routed_experts",
            "experts_per_device",
            "format_version",
            "layout_version",
        )
        if any(type(getattr(self, name)) is not int for name in integer_fields):
            raise ValueError("BF4 cache identity scalar fields must be exact integers")
        if (
            type(self.mesh_shape) is not tuple
            or any(type(value) is not int for value in self.mesh_shape)
            or type(self.physical_ids) is not tuple
            or any(type(value) is not int for value in self.physical_ids)
            or type(self.dram_bank_ring_order) is not tuple
            or any(type(bank) is not int for bank in self.dram_bank_ring_order)
        ):
            raise ValueError("BF4 cache identity topology fields must be exact integer tuples")
        pinned = {
            "checkpoint_revision": PINNED_CHECKPOINT_REVISION,
            "checkpoint_config_sha256": CONFIG_SHA256,
            "checkpoint_file_manifest_sha256": CHECKPOINT_FILE_MANIFEST_SHA256,
            "checkpoint_hash_manifest_sha256": CHECKPOINT_TENSOR_MANIFEST_SHA256,
        }
        for name, expected in pinned.items():
            actual = getattr(self, name)
            if actual != expected:
                raise ValueError(f"BF4 cache {name} must be the pinned value {expected}, got {actual}")
        if (
            type(self.converter_sources) is not tuple
            or not self.converter_sources
            or any(
                type(source) is not tuple
                or len(source) != 2
                or type(source[0]) is not str
                or not source[0]
                or type(source[1]) is not str
                or len(source[1]) != 64
                or any(character not in "0123456789abcdef" for character in source[1])
                for source in self.converter_sources
            )
            or [source[0] for source in self.converter_sources]
            != sorted({source[0] for source in self.converter_sources})
        ):
            raise ValueError("BF4 cache requires the converter sources as sorted (path, lowercase sha256) pairs")
        if self.mesh_shape != MESH_SHAPE:
            raise ValueError(f"BF4 cache requires mesh {MESH_SHAPE}, got {self.mesh_shape}")
        if (
            len(self.physical_ids) != 4
            or len(set(self.physical_ids)) != 4
            or any(device_id < 0 for device_id in self.physical_ids)
        ):
            raise ValueError(f"BF4 cache requires four distinct physical IDs, got {self.physical_ids}")
        if self.ring_size not in BLACKHOLE_RING_SIZES:
            raise ValueError(f"Blackhole ring size must be one of {BLACKHOLE_RING_SIZES}, got {self.ring_size}")
        if sorted(self.dram_bank_ring_order) != list(range(self.ring_size)):
            raise ValueError("BF4 cache identity must list every live DRAM bank once in ring order")
        if (self.hidden_size, self.intermediate_size, self.routed_experts, self.experts_per_device) != (
            2560,
            640,
            512,
            128,
        ):
            raise ValueError("BF4 cache identity does not match the pinned Qwen3.8-Flash-Next routed shape")
        if (self.dtype, self.packer, self.format_version, self.layout_version) != (
            DTYPE,
            PACKER,
            FORMAT_VERSION,
            LAYOUT_VERSION,
        ):
            raise ValueError("BF4 cache encoding identity differs from the released packer contract")

    @property
    def key(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def ring_shard_maps(hidden_size: int, intermediate_size: int, ring_size: int):
    if ring_size not in BLACKHOLE_RING_SIZES:
        raise ValueError(f"unsupported Blackhole ring size {ring_size}")
    if hidden_size % ttnn.TILE_SIZE or intermediate_size % ttnn.TILE_SIZE:
        raise ValueError("expert dimensions must be whole TT tiles")
    hidden_tiles = hidden_size // ttnn.TILE_SIZE
    intermediate_tiles = intermediate_size // ttnn.TILE_SIZE
    w0_w1 = [_shard_tiles(intermediate_tiles, core, ring_size) for core in range(ring_size)]
    max_w2_tiles = (hidden_tiles + ring_size - 1) // ring_size
    groups_per_core = (max_w2_tiles + W2_TILES_PER_A2A_ITER_W - 1) // W2_TILES_PER_A2A_ITER_W
    w2 = []
    for core in range(ring_size):
        tiles = _w2_shard_tiles(hidden_tiles, core, intermediate_tiles, ring_size)
        last_group_tiles = tiles - (groups_per_core - 1) * W2_TILES_PER_A2A_ITER_W
        last_group_pad = groups_per_core * W2_TILES_PER_A2A_ITER_W - tiles
        w2.append((last_group_tiles, last_group_pad))
    return w0_w1, w2


def _packed_block_dims(*, ring_size: int) -> dict[str, tuple[int, int]]:
    """Dimensions 3 and 4 -- ``(blocks, rows per block)`` -- of each packed tensor of the pinned 2560/640 expert, from
    the layout helpers of ``ttnn.experimental.moe_compute_utils`` (the python mirror of the op's
    ``moe_ring_common.h``).  W0/W1: each ring core stores only its own gate/up columns and the (layer, expert) stream
    is cut into ``ring_size`` equal bank pieces of whole blocks.  W2: one slice per ring core; with a half-width last
    all-to-all iteration (the 8-bank ring's 20-tile transactions) it is stored as blocks, otherwise the packer keeps
    its grouped ``(iterations, K padded to whole blocks)`` form.  ``tests/test_ttnn_bf4_static.py`` pins both against
    the packers' output."""

    if ring_size not in BLACKHOLE_RING_SIZES:
        raise ValueError(f"unsupported Blackhole ring size {ring_size}")
    hidden_tiles, intermediate_tiles = 2560 // ttnn.TILE_SIZE, 640 // ttnn.TILE_SIZE
    tiles_per_txn = _tiles_per_txn(hidden_tiles, intermediate_tiles, False, ring_size)
    block_tiles_h = _block_tiles_h(tiles_per_txn)
    block_rows = block_tiles_h * ttnn.TILE_SIZE
    w0_w1 = _w0_w1_compact_layout(hidden_tiles, intermediate_tiles, ring_size, ring_size, tiles_per_txn)
    if w0_w1["uniform"]:
        w0_w1_dims = (w0_w1["stored_cols"][0] // 2, w0_w1["blocks_per_col"] * block_rows)
    else:
        w0_w1_dims = (w0_w1["bank_blocks_per_expert"], block_rows)
    if _w2_last_a2a_iter_half(hidden_tiles, ring_size, tiles_per_txn):
        w2_dims = (_w2_core_blocks_per_expert(hidden_tiles, intermediate_tiles, ring_size, tiles_per_txn), block_rows)
    else:
        w2_dims = (_w2_num_a2a_iters(hidden_tiles, ring_size), -(-intermediate_tiles // block_tiles_h) * block_rows)
    return {"w0_w1": w0_w1_dims, "w2": w2_dims}


def canonical_packed_shapes(*, ring_size: int) -> dict[str, tuple[int, ...]]:
    """Exact global tensor shapes serialized for one four-device cache slot: ``(ring, 1, 512 experts, blocks, rows,
    4 tiles)`` per tensor (``_packed_block_dims``)."""

    dims = _packed_block_dims(ring_size=ring_size)
    return {name: (ring_size, 1, 512, *dims[name], 4 * ttnn.TILE_SIZE) for name in ("w0_w1", "w2")}


def packed_bf4_bytes_per_device(*, ring_size: int, experts_per_device: int = 128) -> tuple[int, int]:
    """Exact packed W0/W1 and W2 BF4 payload bytes for one device (``experts_per_device`` experts of the canonical
    slot shape; a BF4 tile is 576 bytes): the pre-allocation memory gate before any checkpoint tensor is read."""

    shapes = canonical_packed_shapes(ring_size=ring_size)
    w0_w1, w2 = (
        _packed_payload_bytes((*shapes[name][:2], experts_per_device, *shapes[name][3:])) for name in ("w0_w1", "w2")
    )
    return w0_w1, w2


def _packed_payload_bytes(logical_shape: tuple[int, ...]) -> int:
    """Return BF4_B packed bytes represented by one exact tiled global shape."""

    if (
        type(logical_shape) is not tuple
        or len(logical_shape) != 6
        or any(type(item) is not int or item <= 0 for item in logical_shape)
    ):
        raise RuntimeError(f"BF4 artifact logical shape is invalid: {logical_shape!r}")
    elements = 1
    for item in logical_shape:
        elements *= item
    tile_elements = ttnn.TILE_SIZE * ttnn.TILE_SIZE
    if elements % tile_elements:
        raise RuntimeError(f"BF4 artifact logical shape is not whole-tile packed: {logical_shape!r}")
    return elements // tile_elements * BF4_TILE_BYTES


def _prepare_routed_layer_host_tensors(
    weights: Qwen38MoEWeights,
    *,
    ring_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one complete EP4 layer without retaining four shard-sized copies.

    ``prepare_*_tensor_for_moe_compute`` consumes one device's 128 experts at
    a time.  The original layer conversion accumulated all four prepared pairs
    and then ``torch.cat`` allocated a second complete layer.  Besides the raw
    checkpoint slice, that transiently retained two copies of both packed host
    tensors.  A first-time layer-0 stage is the exact case where this matters.

    Allocate the canonical 512-expert destinations once and copy each prepared
    expert range directly into dimension 2.  This preserves the byte order
    consumed by ``ShardTensor2dMesh(..., dims=(None, 2))`` while bounding live
    prepared-shard storage to one device.  Nothing is published here; the
    caller continues to use the existing temporary tensorbin + manifest path.
    """

    expected_ranges = tuple((device * 128, (device + 1) * 128) for device in range(4))
    if tuple(tuple(pair) for pair in weights.expert_ranges) != expected_ranges:
        raise RuntimeError(
            f"layer host packing requires canonical EP4 expert ranges {expected_ranges}, "
            f"got {tuple(weights.expert_ranges)}"
        )

    canonical_shapes = canonical_packed_shapes(ring_size=ring_size)
    w01_map, w2_map = ring_shard_maps(2560, 640, ring_size)
    torch_w01 = torch.empty(canonical_shapes["w0_w1"], dtype=torch.bfloat16)
    torch_w2 = torch.empty(canonical_shapes["w2"], dtype=torch.bfloat16)

    for device_index, (start, end) in enumerate(expected_ranges):
        shard = weights.routed_device_shard(device_index)
        if tuple(shard.expert_range) != (start, end):
            raise RuntimeError(
                f"checkpoint routed shard {device_index} owns {tuple(shard.expert_range)}, expected {(start, end)}"
            )
        gate, up = torch.split(shard.gate_up, 640, dim=-1)
        prepared_w01 = prepare_w0_w1_tensor_for_moe_compute(
            gate,
            up,
            1,
            128,
            2560,
            640,
            w01_map,
        )
        prepared_w2 = prepare_w2_tensor_for_moe_compute(
            shard.down,
            1,
            128,
            640,
            2560,
            w2_map,
            w01_map,
        )
        expected_w01_shape = canonical_shapes["w0_w1"][:2] + (end - start,) + canonical_shapes["w0_w1"][3:]
        expected_w2_shape = canonical_shapes["w2"][:2] + (end - start,) + canonical_shapes["w2"][3:]
        if tuple(prepared_w01.shape) != expected_w01_shape or prepared_w01.dtype != torch.bfloat16:
            raise RuntimeError(
                f"prepared W0/W1 shard {device_index} is {prepared_w01.dtype} {tuple(prepared_w01.shape)}, "
                f"expected torch.bfloat16 {expected_w01_shape}"
            )
        if tuple(prepared_w2.shape) != expected_w2_shape or prepared_w2.dtype != torch.bfloat16:
            raise RuntimeError(
                f"prepared W2 shard {device_index} is {prepared_w2.dtype} {tuple(prepared_w2.shape)}, "
                f"expected torch.bfloat16 {expected_w2_shape}"
            )
        torch_w01.narrow(2, start, end - start).copy_(prepared_w01)
        torch_w2.narrow(2, start, end - start).copy_(prepared_w2)
        del shard, gate, up, prepared_w01, prepared_w2

    return torch_w01, torch_w2


def _expert_byte_ranges(
    logical_shape: tuple[int, ...],
    expert: int,
    physical_ids: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    """``(offset, size)`` of one expert's tiles in the tensorbin payload, one contiguous run per ring bank.

    The payload is the mesh shards in coordinate order, each a tiled ``(ring, 1, experts_per_device, blocks, rows,
    cols)`` tensor: tiles run over the leading dims in row-major order, so an expert's blocks (W0/W1: the bank piece
    stored there; W2: the ring core's slice) are one run per ring bank.
    """

    ring_size, _, experts, blocks, rows, cols = logical_shape
    experts_per_device = experts // len(physical_ids)
    run = blocks * (rows // ttnn.TILE_SIZE) * (cols // ttnn.TILE_SIZE) * BF4_TILE_BYTES
    shard = experts_per_device * ring_size * run
    device_index, local = divmod(expert, experts_per_device)
    return tuple(
        (device_index * shard + (ring_bank * experts_per_device + local) * run, run) for ring_bank in range(ring_size)
    )


def _fresh_expert_bf4_bytes(
    checkpoint: Qwen38Checkpoint,
    placement: Qwen38Placement,
    *,
    namespace: str,
    layer_index: int,
    expert: int,
    ring_size: int,
    scratch: Path,
) -> dict[str, bytes]:
    """Pack one routed expert exactly as the layer conversion does (``E = 1``): the packed tile bytes per tensor."""

    if namespace == "backbone":
        weights = Qwen38MoEWeights(checkpoint, placement, layer_index=layer_index)
    else:
        weights = Qwen38MoEWeights(checkpoint, placement, mtp_layer_index=layer_index)
    routed = weights.expert(expert)
    gate_up = routed.gate_up.transpose(0, 1)[None, None].contiguous()
    down = routed.down.transpose(0, 1)[None, None].contiguous()
    w01_map, w2_map = ring_shard_maps(2560, 640, ring_size)
    gate, up = torch.split(gate_up, 640, dim=-1)
    prepared = {
        "w0_w1": prepare_w0_w1_tensor_for_moe_compute(gate, up, 1, 1, 2560, 640, w01_map),
        "w2": prepare_w2_tensor_for_moe_compute(down, 1, 1, 640, 2560, w2_map, w01_map),
    }
    packed = {}
    for name, host_tensor in prepared.items():
        path = scratch / f"{name}.tensorbin"
        ttnn.dump_tensor(path, ttnn.from_torch(host_tensor, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT))
        raw = path.read_bytes()
        header_size = int.from_bytes(raw[:TENSORBIN_HEADER_PREFIX_BYTES], byteorder="little", signed=False)
        packed[name] = raw[TENSORBIN_HEADER_PREFIX_BYTES + header_size :]
    return packed


def packed_bf4_model_bytes_per_device(*, ring_size: int, moe_layers: int = 49) -> int:
    """Packed routed bytes for 48 backbone layers plus the one MTP layer."""

    if moe_layers != 49:
        raise ValueError(f"the exact target contains 49 MoE layers, got {moe_layers}")
    return sum(packed_bf4_bytes_per_device(ring_size=ring_size)) * moe_layers


class Qwen38BF4Cache:
    """Create and load exact multi-device routed-expert cache tensors."""

    def __init__(
        self,
        root: str | Path,
        identity: BF4CacheIdentity,
        mesh_contract: Qwen38MeshContract,
    ) -> None:
        self.root = Path(root).resolve() / identity.key
        self.identity = identity
        self.mesh_contract = mesh_contract
        if mesh_contract.physical_ids != identity.physical_ids:
            raise ValueError("mesh contract and BF4 cache physical ordering differ")
        self.manifest_path = self.root / "manifest.json"
        self._verified_layers: dict[
            tuple[str, int],
            tuple[BF4LayerRecord, tuple[tuple[int, int, int, int, int], ...]],
        ] = {}
        if not self.manifest_path.exists():
            self._adopt_legacy_slot(Path(root).resolve())

    def _adopt_legacy_slot(self, cache_root: Path) -> None:
        """Move a sibling slot that holds this identity's bytes under this identity's key.

        A format-1 slot was keyed by the tt-metal revision, a format-2 slot by the first die's DRAM bank-to-worker
        coordinates (and by this module's digest among the converter sources, which the format change itself moves).
        A slot whose other identity fields equal ours (format 2: the packers' digests too) and whose stored coordinates
        derive to our ring order is rewritten as format 3 and renamed; a format-3 slot that already carries our key
        under another name (an interrupted adoption) is only renamed.  Nothing is reconverted: the admission probe
        checks the bytes at every start, so a slot converted by different code is still refused.
        """

        expected = _json_normalized(asdict(self.identity))
        del expected["dram_bank_ring_order"], expected["format_version"]
        this_module = CONVERTER_SOURCES[0].relative_to(REPO_ROOT).as_posix()
        packers = [source for source in expected.pop("converter_sources") if source[0] != this_module]
        candidates = []
        for manifest_path in cache_root.glob("*/manifest.json"):
            if manifest_path.parent == self.root:
                continue
            try:
                document = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if type(document) is not dict or type(document.get("identity")) is not dict:
                continue
            legacy_format = document.get("format_version")
            if legacy_format == FORMAT_VERSION:
                adoptable = document.get("identity_key") == self.identity.key
            elif legacy_format in LEGACY_FORMAT_VERSIONS:
                identity = dict(document["identity"])
                identity.pop("format_version", None)
                coordinates = identity.pop("dram_bank_worker_order", None)
                if legacy_format == 1:
                    comparable = type(identity.pop("tt_metal_revision", None)) is str
                else:
                    sources = identity.pop("converter_sources", None)
                    comparable = type(sources) is list and packers == [
                        source for source in sources if not (type(source) is list and source[:1] == [this_module])
                    ]
                adoptable = (
                    comparable
                    and _same_exact_typed_tree(identity, expected)
                    and type(coordinates) is list
                    and len(coordinates) == self.identity.ring_size
                    and all(
                        type(coordinate) is list and len(coordinate) == 2 and all(type(v) is int for v in coordinate)
                        for coordinate in coordinates
                    )
                    and len(set(map(tuple, coordinates))) == self.identity.ring_size
                    and dram_bank_ring_order(tuple(map(tuple, coordinates))) == self.identity.dram_bank_ring_order
                )
            else:
                adoptable = False
            if adoptable and type(document.get("updated_utc")) is str:
                candidates.append((document["updated_utc"], manifest_path, document))
        if not candidates:
            return
        _, manifest_path, document = max(candidates, key=lambda candidate: candidate[0])
        with _exclusive_file_lock(manifest_path.parent / ".conversion.lock"):
            if self.manifest_path.exists() or not manifest_path.exists():
                return
            document["format_version"] = FORMAT_VERSION
            document["identity"] = _json_normalized(asdict(self.identity))
            document["identity_key"] = self.identity.key
            document["updated_utc"] = _utc_now()
            _atomic_json(manifest_path, document)
            os.rename(manifest_path.parent, self.root)

    @staticmethod
    def _validate_layer_request(namespace: str, layer_index: int) -> None:
        validate_bf4_layer_request(namespace, layer_index)

    def _base(self, namespace: str, layer_index: int, name: str) -> Path:
        self._validate_layer_request(namespace, layer_index)
        return self.root / namespace / f"layer-{layer_index:02d}" / name

    def _validate_record(
        self,
        record: BF4LayerRecord,
        *,
        namespace: str,
        layer_index: int,
    ) -> tuple[Path, Path]:
        """Bind a manifest record to its exact requested slot and cache files."""

        self._validate_layer_request(namespace, layer_index)
        if type(record.namespace) is not str or type(record.layer_index) is not int:
            raise RuntimeError("BF4 layer record namespace/index types are invalid")
        if (record.namespace, record.layer_index) != (namespace, layer_index):
            raise RuntimeError(
                f"BF4 layer record identity {(record.namespace, record.layer_index)} differs from "
                f"requested {(namespace, layer_index)}"
            )
        expected_ranges = tuple(
            (
                device_index * self.identity.experts_per_device,
                (device_index + 1) * self.identity.experts_per_device,
            )
            for device_index in range(len(self.identity.physical_ids))
        )
        if (
            any(type(value) is not int for pair in record.expert_ranges for value in pair)
            or record.expert_ranges != expected_ranges
        ):
            raise RuntimeError(
                f"BF4 layer expert ownership {record.expert_ranges} differs from expected {expected_ranges}"
            )
        if type(record.ring_size) is not int or record.ring_size != self.identity.ring_size:
            raise RuntimeError(
                f"BF4 layer ring size {record.ring_size!r} differs from cache identity {self.identity.ring_size}"
            )

        canonical_shapes = canonical_packed_shapes(ring_size=self.identity.ring_size)
        packed_per_device = dict(
            zip(
                ("w0_w1", "w2"),
                packed_bf4_bytes_per_device(
                    ring_size=self.identity.ring_size,
                    experts_per_device=self.identity.experts_per_device,
                ),
            )
        )
        paths = []
        for name, artifact in (("w0_w1", record.w0_w1), ("w2", record.w2)):
            if type(artifact) is not BF4Artifact:
                raise RuntimeError(f"BF4 artifact {name} schema is invalid")
            expected_path = _tensorbin_path(self._base(namespace, layer_index, name))
            expected_relative_path = str(expected_path.relative_to(self.root))
            if (
                type(artifact.name) is not str
                or type(artifact.relative_path) is not str
                or type(artifact.sha256) is not str
                or len(artifact.sha256) != 64
                or any(character not in "0123456789abcdef" for character in artifact.sha256)
                or type(artifact.bytes) is not int
                or artifact.bytes <= 0
                or type(artifact.dtype) is not str
                or type(artifact.layout) is not str
            ):
                raise RuntimeError(f"BF4 artifact {name} scalar schema is invalid")
            if (artifact.name, artifact.dtype, artifact.layout) != (name, DTYPE, LAYOUT):
                raise RuntimeError(f"BF4 artifact {name} encoding identity is invalid")
            if artifact.relative_path != expected_relative_path:
                raise RuntimeError(
                    f"BF4 artifact {name} path {artifact.relative_path!r} differs from requested "
                    f"slot {expected_relative_path!r}"
                )
            expected_shape = canonical_shapes[name]
            if (
                type(artifact.logical_shape) is not tuple
                or any(type(item) is not int for item in artifact.logical_shape)
                or artifact.logical_shape != expected_shape
            ):
                raise RuntimeError(
                    f"BF4 artifact {name} logical shape {artifact.logical_shape!r} differs from "
                    f"canonical slot shape {expected_shape!r}"
                )
            expected_global_payload = packed_per_device[name] * len(self.identity.physical_ids)
            if _packed_payload_bytes(artifact.logical_shape) != expected_global_payload:
                raise RuntimeError(f"BF4 artifact {name} packed byte size differs from the canonical slot")
            paths.append(expected_path)
        return paths[0], paths[1]

    def _validate_mesh_shapes(self, tt_w01: Any, tt_w2: Any, *, label: str) -> None:
        """A mesh tensor's shape is the coordinate-local shard: the slot's experts (dim 2) split over the mesh
        columns, 128 per device; the rest of the canonical slot shape is per device already."""

        canonical_shapes = canonical_packed_shapes(ring_size=self.identity.ring_size)
        for name, tensor in (("w0_w1", tt_w01), ("w2", tt_w2)):
            slot_shape = canonical_shapes[name]
            expected = (*slot_shape[:2], self.identity.experts_per_device, *slot_shape[3:])
            loaded_shape = _native_integer_shape(tensor.shape, label=f"{label} {name} loaded")
            if loaded_shape != expected:
                raise RuntimeError(
                    f"{label} {name} loaded shape {loaded_shape} differs from the coordinate-local slot shape "
                    f"{expected} (slot {slot_shape})"
                )

    def _read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if type(document) is not dict:
            raise RuntimeError("BF4 cache manifest root is invalid")
        manifest_keys = {
            "format_version",
            "identity_key",
            "identity",
            "created_utc",
            "created_host",
            "updated_utc",
            "layers",
        }
        if set(document) != manifest_keys:
            raise RuntimeError("BF4 cache manifest top-level schema is invalid")
        if type(document.get("format_version")) is not int or document["format_version"] != FORMAT_VERSION:
            raise RuntimeError("BF4 cache manifest format is invalid")
        expected_identity = _json_normalized(asdict(self.identity))
        if not _same_exact_typed_tree(document.get("identity"), expected_identity):
            raise RuntimeError("BF4 cache manifest identity does not match this run")
        if type(document.get("identity_key")) is not str or document["identity_key"] != self.identity.key:
            raise RuntimeError("BF4 cache identity digest is corrupt")
        if any(
            type(document.get(name)) is not str or not document[name]
            for name in ("created_utc", "created_host", "updated_utc")
        ):
            raise RuntimeError("BF4 cache manifest creation/update fields are invalid")
        if type(document.get("layers")) is not dict:
            raise RuntimeError("BF4 cache manifest layer table is invalid")
        if any(type(key) is not str or type(value) is not dict for key, value in document["layers"].items()):
            raise RuntimeError("BF4 cache manifest layer table entries are invalid")
        return document

    def _record(self, record: BF4LayerRecord) -> None:
        self._validate_record(record, namespace=record.namespace, layer_index=record.layer_index)
        document = self._read_manifest()
        if document is None:
            document = {
                "format_version": FORMAT_VERSION,
                "identity_key": self.identity.key,
                "identity": _json_normalized(asdict(self.identity)),
                "created_utc": _utc_now(),
                "created_host": socket.gethostname(),
                "layers": {},
            }
        key = f"{record.namespace}:{record.layer_index}"
        serialized = _json_normalized(asdict(record))
        previous = document["layers"].get(key)
        if previous is not None and previous != serialized:
            raise RuntimeError(f"refusing to overwrite non-identical BF4 cache record {key}")
        document["layers"][key] = serialized
        document["updated_utc"] = _utc_now()
        _atomic_json(self.manifest_path, document)

    def verify_layer(self, namespace: str, layer_index: int) -> BF4LayerRecord | None:
        """Validate one exact manifest slot, hashing each stable artifact once per cache owner.

        The manifest identity and requested record are parsed on every call.  A
        verified payload is reused only while its exact record and both live
        filesystem signatures remain unchanged; a new cache object establishes
        its own trust by hashing the files again.
        """

        self._validate_layer_request(namespace, layer_index)
        document = self._read_manifest()
        if document is None:
            return None
        raw = document["layers"].get(f"{namespace}:{layer_index}")
        if raw is None:
            return None
        record_keys = {"namespace", "layer_index", "expert_ranges", "ring_size", "w0_w1", "w2"}
        artifact_keys = {"name", "relative_path", "sha256", "bytes", "logical_shape", "dtype", "layout"}
        if type(raw) is not dict or set(raw) != record_keys:
            raise RuntimeError("BF4 layer record schema is invalid")
        artifacts = {}
        for name in ("w0_w1", "w2"):
            if type(raw[name]) is not dict or set(raw[name]) != artifact_keys:
                raise RuntimeError(f"BF4 artifact {name} manifest schema is invalid")
            artifact_data = dict(raw[name])
            if type(artifact_data["logical_shape"]) is not list:
                raise RuntimeError(f"BF4 artifact {name} logical shape manifest type is invalid")
            artifact_data["logical_shape"] = tuple(artifact_data["logical_shape"])
            artifact = BF4Artifact(**artifact_data)
            artifacts[name] = artifact
        if type(raw["expert_ranges"]) is not list or any(type(pair) is not list for pair in raw["expert_ranges"]):
            raise RuntimeError("BF4 layer expert ownership manifest type is invalid")
        record = BF4LayerRecord(
            namespace=raw["namespace"],
            layer_index=raw["layer_index"],
            expert_ranges=tuple(tuple(pair) for pair in raw["expert_ranges"]),
            ring_size=raw["ring_size"],
            **artifacts,
        )
        artifact_paths = self._validate_record(record, namespace=namespace, layer_index=layer_index)
        key = (namespace, layer_index)
        cached = self._verified_layers.get(key)
        if cached is not None and cached[0] == record:
            try:
                current_signatures = tuple(_artifact_stat_signature(path) for path in artifact_paths)
            except (OSError, RuntimeError):
                current_signatures = ()
            if current_signatures == cached[1]:
                return record
        self._verified_layers.pop(key, None)

        verified_signatures = []
        for artifact, path in zip((record.w0_w1, record.w2), artifact_paths):
            with _verified_artifact_fd(path, artifact) as verified:
                verified_signatures.append(verified.signature)
        self._verified_layers[key] = (record, tuple(verified_signatures))
        return record

    def _load_verified_tensors(
        self,
        mesh_device,
        *,
        record: BF4LayerRecord,
        w01_path: Path,
        w2_path: Path,
        memory_configs,
    ) -> tuple[Any, Any]:
        """Load and validate both artifacts while their verified FDs remain open."""

        cached = self._verified_layers.get((record.namespace, record.layer_index))
        if cached is None or cached[0] != record or len(cached[1]) != 2:
            raise RuntimeError("BF4 layer must have a session-verified artifact identity before loading")
        w01_signature, w2_signature = cached[1]
        tt_w01 = None
        tt_w2 = None
        try:
            with (
                _verified_artifact_fd(
                    w01_path,
                    record.w0_w1,
                    expected_signature=w01_signature,
                ) as retained_w01,
                _verified_artifact_fd(
                    w2_path,
                    record.w2,
                    expected_signature=w2_signature,
                ) as retained_w2,
            ):
                tt_w01 = ttnn.load_tensor(retained_w01.proc_path, device=mesh_device)
                tt_w2 = ttnn.load_tensor(retained_w2.proc_path, device=mesh_device)
                self._validate_mesh_shapes(tt_w01, tt_w2, label="BF4 cache")
                if tt_w01.memory_config() != memory_configs.w0_w1 or tt_w2.memory_config() != memory_configs.w2:
                    raise RuntimeError("BF4 cache memory configuration differs from the live Blackhole ring")
                self.mesh_contract.validate_tensor(tt_w01, placement=TensorPlacement.EXPERT_SHARDED, shard_dim=2)
                self.mesh_contract.validate_tensor(tt_w2, placement=TensorPlacement.EXPERT_SHARDED, shard_dim=2)
                if tt_w01.dtype != ttnn.bfloat4_b or tt_w2.dtype != ttnn.bfloat4_b:
                    raise RuntimeError("routed expert cache did not load as BFLOAT4_B")
        except BaseException as error:
            _raise_if_cleanup_failed(
                (tt_w01, tt_w2),
                context="BF4 cache load failed and owned tensor cleanup was incomplete",
                primary_error=error,
            )
            raise
        return tt_w01, tt_w2

    def convert_and_upload(
        self,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        mesh_device,
        *,
        layer_index: int,
        namespace: str = "backbone",
    ) -> tuple[Any, Any]:
        """Serialize cache creation for a layer and publish only verified files."""

        self._validate_layer_request(namespace, layer_index)
        lock_path = self.root / ".conversion.lock"
        with _exclusive_file_lock(lock_path):
            return self._convert_and_upload_locked(
                checkpoint,
                placement,
                mesh_device,
                layer_index=layer_index,
                namespace=namespace,
            )

    def _convert_and_upload_locked(
        self,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        mesh_device,
        *,
        layer_index: int,
        namespace: str = "backbone",
    ) -> tuple[Any, Any]:
        """Build/cache one layer and return packed BF4_B tensors on the mesh.

        Cache creation uses ``ShardTensor2dMesh(..., dims=(None, 2))`` so the
        serialized tensor records a real ``(1,4)`` topology with 128 experts per
        column.  A cache hit is topology-checked immediately after load.
        """

        self.mesh_contract.validate_mesh(mesh_device)
        if tuple(mesh_device.shape) != self.identity.mesh_shape:
            raise RuntimeError("mesh shape changed after BF4 cache construction")
        live_ring_order = qualify_live_bf4_ring(mesh_device)
        if len(live_ring_order) != self.identity.ring_size:
            raise RuntimeError(
                f"live ring has {len(live_ring_order)} banks, cache identity requires {self.identity.ring_size}"
            )
        if live_ring_order != self.identity.dram_bank_ring_order:
            raise RuntimeError(
                "live DRAM bank ring order differs from the cache identity: "
                f"live={live_ring_order} cache={self.identity.dram_bank_ring_order}"
            )
        existing = self.verify_layer(namespace, layer_index)
        w01_base = self._base(namespace, layer_index, "w0_w1")
        w2_base = self._base(namespace, layer_index, "w2")
        w01_path, w2_path = _tensorbin_path(w01_base), _tensorbin_path(w2_base)
        expected_ranges = tuple(tuple(pair) for pair in placement.expert_ranges)
        if existing is None and (w01_path.exists() or w2_path.exists()):
            raise RuntimeError(
                "refusing to adopt unmanifested BF4 artifact(s); classify and remove only the task-owned "
                f"orphan before retrying: {w01_path}, {w2_path}"
            )
        if existing is not None:
            if existing.expert_ranges != expected_ranges:
                raise RuntimeError("cached MoE expert ownership differs from the requested placement")
            if existing.ring_size != self.identity.ring_size:
                raise RuntimeError("cached MoE ring size differs from the live cache identity")

        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 2))
        memory_configs = ttnn.experimental.get_weight_mem_configs(
            mesh_device,
            num_layers=1,
            experts_per_device=128,
            hidden_size=2560,
            intermediate_size=640,
            has_bias=False,
        )

        if existing is None:
            if namespace == "backbone":
                weights = Qwen38MoEWeights(checkpoint, placement, layer_index=layer_index)
            else:
                weights = Qwen38MoEWeights(checkpoint, placement, mtp_layer_index=layer_index)
            if tuple(weights.expert_ranges) != expected_ranges:
                raise RuntimeError("MoE expert ownership changed during conversion")
            torch_w01, torch_w2 = _prepare_routed_layer_host_tensors(
                weights,
                ring_size=self.identity.ring_size,
            )
            w01_base.parent.mkdir(parents=True, exist_ok=True)
            tt_w01 = None
            tt_w2 = None
            try:
                with tempfile.TemporaryDirectory(prefix=".convert.", dir=w01_base.parent) as temporary_directory:
                    temporary_root = Path(temporary_directory)
                    temporary_w01_base = temporary_root / "w0_w1"
                    temporary_w2_base = temporary_root / "w2"
                    tt_w01 = ttnn.as_tensor(
                        torch_w01,
                        dtype=ttnn.bfloat4_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh_device,
                        memory_config=memory_configs.w0_w1,
                        mesh_mapper=mapper,
                        cache_file_name=temporary_w01_base,
                    )
                    del torch_w01
                    tt_w2 = ttnn.as_tensor(
                        torch_w2,
                        dtype=ttnn.bfloat4_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh_device,
                        memory_config=memory_configs.w2,
                        mesh_mapper=mapper,
                        cache_file_name=temporary_w2_base,
                    )
                    del torch_w2
                    temporary_w01_path = _tensorbin_path(temporary_w01_base)
                    temporary_w2_path = _tensorbin_path(temporary_w2_base)
                    if not temporary_w01_path.is_file() or not temporary_w2_path.is_file():
                        raise RuntimeError("ttnn.as_tensor did not create both staged BF4 tensorbins")
                    # The record carries the slot's global shape (what the host packed: 512 experts on dim 2); the
                    # mesh tensor presents one coordinate's shard.  Both are checked while the files are still
                    # temporary, so a refused record publishes nothing.
                    self._validate_mesh_shapes(tt_w01, tt_w2, label="BF4 conversion")
                    canonical_shapes = canonical_packed_shapes(ring_size=self.identity.ring_size)
                    record = BF4LayerRecord(
                        namespace=namespace,
                        layer_index=layer_index,
                        expert_ranges=expected_ranges,
                        ring_size=self.identity.ring_size,
                        w0_w1=BF4Artifact(
                            name="w0_w1",
                            relative_path=str(w01_path.relative_to(self.root)),
                            sha256=_sha256(temporary_w01_path),
                            bytes=temporary_w01_path.stat().st_size,
                            logical_shape=canonical_shapes["w0_w1"],
                        ),
                        w2=BF4Artifact(
                            name="w2",
                            relative_path=str(w2_path.relative_to(self.root)),
                            sha256=_sha256(temporary_w2_path),
                            bytes=temporary_w2_path.stat().st_size,
                            logical_shape=canonical_shapes["w2"],
                        ),
                    )
                    self._validate_record(record, namespace=namespace, layer_index=layer_index)
                    if w01_path.exists() or w2_path.exists():
                        raise RuntimeError("BF4 destination appeared while the exclusive conversion lock was held")
                    os.replace(temporary_w01_path, w01_path)
                    os.replace(temporary_w2_path, w2_path)
                self._record(record)
                verified = self.verify_layer(namespace, layer_index)
                if verified is None:
                    raise RuntimeError("published BF4 layer is absent from its manifest")
                self.mesh_contract.validate_tensor(tt_w01, placement=TensorPlacement.EXPERT_SHARDED, shard_dim=2)
                self.mesh_contract.validate_tensor(tt_w2, placement=TensorPlacement.EXPERT_SHARDED, shard_dim=2)
                if tt_w01.dtype != ttnn.bfloat4_b or tt_w2.dtype != ttnn.bfloat4_b:
                    raise RuntimeError("routed expert cache did not load as BFLOAT4_B")
            except BaseException as error:
                _raise_if_cleanup_failed(
                    (tt_w01, tt_w2),
                    context="BF4 conversion failed and owned tensor cleanup was incomplete",
                    primary_error=error,
                )
                raise
            return tt_w01, tt_w2
        else:
            return self._load_verified_tensors(
                mesh_device,
                record=existing,
                w01_path=w01_path,
                w2_path=w2_path,
                memory_configs=memory_configs,
            )

    def admit_converted_bytes(
        self,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        *,
        namespace: str = "backbone",
        layer_index: int = 0,
        expert: int = ADMISSION_EXPERT,
    ) -> dict[str, Any]:
        """Re-pack one routed expert of one cached layer from the checkpoint and compare the bytes with the cache.

        The manifest pins the converter's sources, not the runtime that ran them; this is the check that the cached
        bytes are what this runtime's converter produces.  Fails closed naming the layer, expert, tensor and ring bank.
        The host packer needs the cluster open (tt-metal initializes on the first BF4 tensor), so this runs after the
        mesh is open, on ~4 MB of packed bytes.
        """

        started = time.monotonic()
        record = self.verify_layer(namespace, layer_index)
        if record is None:
            raise RuntimeError(f"BF4 cache holds no {namespace} layer {layer_index} to admit")
        if not 0 <= expert < self.identity.routed_experts:
            raise ValueError(f"expert must be in [0, {self.identity.routed_experts}), got {expert}")
        paths = self._validate_record(record, namespace=namespace, layer_index=layer_index)
        with tempfile.TemporaryDirectory(prefix=".admit.", dir=self.root) as scratch:
            fresh = _fresh_expert_bf4_bytes(
                checkpoint,
                placement,
                namespace=namespace,
                layer_index=layer_index,
                expert=expert,
                ring_size=self.identity.ring_size,
                scratch=Path(scratch),
            )
        compared = 0
        for artifact, path in zip((record.w0_w1, record.w2), paths):
            ranges = _expert_byte_ranges(artifact.logical_shape, expert, self.identity.physical_ids)
            if len(fresh[artifact.name]) != len(ranges) * ranges[0][1]:
                raise RuntimeError(
                    f"fresh {artifact.name} packing of expert {expert} is {len(fresh[artifact.name])} bytes, "
                    f"expected {len(ranges) * ranges[0][1]}"
                )
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                header_size = _validate_tensorbin_payload_fd(
                    descriptor,
                    signature=_artifact_fd_signature(descriptor),
                    expected_payload_bytes=_packed_payload_bytes(artifact.logical_shape),
                )
                for ring_bank, (offset, size) in enumerate(ranges):
                    cached = _pread_exact(descriptor, size, TENSORBIN_HEADER_PREFIX_BYTES + header_size + offset)
                    expected = fresh[artifact.name][ring_bank * size : (ring_bank + 1) * size]
                    if cached != expected:
                        first = next(index for index in range(size) if cached[index] != expected[index])
                        raise RuntimeError(
                            f"BF4 cache {self.root} was not produced by this runtime's converter: {namespace} layer "
                            f"{layer_index} expert {expert} ({artifact.name}, ring bank {ring_bank}) differs from a fresh "
                            f"conversion of the checkpoint at byte {first} of {size}; move the slot aside or delete it "
                            "and the next start reconverts"
                        )
                    compared += size
            finally:
                os.close(descriptor)
        return {
            "namespace": namespace,
            "layer_index": layer_index,
            "expert": expert,
            "compared_bytes": compared,
            "seconds": round(time.monotonic() - started, 3),
        }

    def load_layer(self, mesh_device, *, layer_index: int, namespace: str = "backbone") -> tuple[Any, Any]:
        """Load one already-converted layer and revalidate live ring/topology."""

        self._validate_layer_request(namespace, layer_index)
        self.mesh_contract.validate_mesh(mesh_device)
        live_ring_order = qualify_live_bf4_ring(mesh_device)
        if len(live_ring_order) != self.identity.ring_size or live_ring_order != self.identity.dram_bank_ring_order:
            raise RuntimeError(
                "live Blackhole DRAM ring does not match the BF4 cache identity: "
                f"live={live_ring_order} cache={self.identity.dram_bank_ring_order}"
            )
        record = self.verify_layer(namespace, layer_index)
        if record is None:
            raise RuntimeError(f"BF4 cache is missing {namespace} layer {layer_index}")
        if record.ring_size != self.identity.ring_size:
            raise RuntimeError("BF4 layer ring size differs from the cache identity")
        w01_path, w2_path = self._validate_record(record, namespace=namespace, layer_index=layer_index)
        memory_configs = ttnn.experimental.get_weight_mem_configs(
            mesh_device,
            num_layers=1,
            experts_per_device=128,
            hidden_size=2560,
            intermediate_size=640,
            has_bias=False,
        )
        return self._load_verified_tensors(
            mesh_device,
            record=record,
            w01_path=w01_path,
            w2_path=w2_path,
            memory_configs=memory_configs,
        )


class Qwen38BF4Streamer:
    """Single-slot correctness streamer for packed routed expert layers.

    The conservative memory admission budget rejects keeping all 49 packed
    layers resident on either a seven- or eight-bank P150.  This serialized
    context loads exactly one layer, keeps it live for the caller's operation,
    and releases only that task-owned allocation on exit.  Queue overlap and
    double buffering are deferred until the single-queue path is numerically
    proven.
    """

    def __init__(self, cache: Qwen38BF4Cache, mesh_device) -> None:
        cache.mesh_contract.validate_mesh(mesh_device)
        self.cache = cache
        self.mesh_device = mesh_device
        self._active: tuple[str, int] | None = None

    @contextmanager
    def layer(self, layer_index: int, *, namespace: str = "backbone"):
        if self._active is not None:
            raise RuntimeError(f"BF4 streamer already owns active layer {self._active}")
        tensors = self.cache.load_layer(self.mesh_device, layer_index=layer_index, namespace=namespace)
        self._active = (namespace, layer_index)
        try:
            yield tensors
        except BaseException as error:
            self._active = None
            _raise_if_cleanup_failed(
                tensors,
                context="BF4 streamer body failed and owned tensor cleanup was incomplete",
                primary_error=error,
            )
            raise
        else:
            self._active = None
            _raise_if_cleanup_failed(
                tensors,
                context="BF4 streamer owned tensor cleanup was incomplete",
                primary_error=None,
            )


_BACKBONE_RESIDENT_KEYS = tuple(("backbone", layer_index) for layer_index in range(BACKBONE_LAYERS))
_MTP_RESIDENT_KEYS = tuple(("mtp", layer_index) for layer_index in range(MTP_LAYERS))
_ALL_RESIDENT_KEYS = _BACKBONE_RESIDENT_KEYS + _MTP_RESIDENT_KEYS


@dataclass
class _BF4ResidentHandle:
    """Authoritative lifetime state for one object returned by ``load_layer``."""

    tensor: Any
    backing_identity: Qwen38TensorBackingIdentity | None
    first_key: tuple[str, int]
    first_slot: int
    release_attempted: bool = False
    release_error: BaseException | None = None


@dataclass(frozen=True)
class Qwen38BF4ResidentLoadEvent:
    """One immutable cache-I/O outcome in resident preload order."""

    sequence: int
    namespace: str
    layer_index: int
    outcome: Literal["attempt", "success", "failure"]
    error_type: str | None = None


class Qwen38BF4ResidentSet(Qwen38BF4Streamer):
    """One explicit owner for build-time resident backbone and MTP BF4 tensors.

    Residency changes only tensor lifetime.  Every pair is still loaded through
    :class:`Qwen38BF4Cache`, which revalidates the exact cache identity, live
    Blackhole ring, TP4/EP4 placement, shapes, memory configs, and BF4 dtype.
    Decode borrows a retained pair; it cannot fall back to a host cache read or
    deallocate a pair at the end of a layer call.

    ``preload_*`` calls are transactional for newly requested keys and
    idempotent for already resident keys.  Each returned object is registered
    before the pair contract is validated.  A successful release clears its
    handle; a failed release is attempted only once, poisons the owner, and
    remains inspectable.  ``close`` therefore lets target and MTP component
    bundles share one exactly-once release boundary without hiding uncertain
    deallocation outcomes.
    """

    def __init__(self, cache: Qwen38BF4Cache, mesh_device) -> None:
        super().__init__(cache, mesh_device)
        self._slots: dict[tuple[str, int], tuple[_BF4ResidentHandle | None, ...]] = {}
        self._ready: set[tuple[str, int]] = set()
        self._handles: dict[Qwen38TensorBackingIdentity, _BF4ResidentHandle] = {}
        self._load_events: list[Qwen38BF4ResidentLoadEvent] = []
        self._registered_handle_count = 0
        self._preloading = False
        self._poisoned = False
        self._close_started = False
        self._closed = False

    @property
    def resident_layers(self) -> tuple[tuple[str, int], ...]:
        return tuple(key for key in _ALL_RESIDENT_KEYS if key in self._ready)

    @property
    def owned_layers(self) -> tuple[tuple[str, int], ...]:
        """All slots still retaining at least one live or uncertain handle."""

        return tuple(key for key in _ALL_RESIDENT_KEYS if key in self._slots)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def unreleased_tensor_slots(self) -> tuple[tuple[str, int, int, bool, str | None], ...]:
        """Stable ledger of handles that still own or may own device storage."""

        result = []
        for namespace, layer_index in self.owned_layers:
            for slot, handle in enumerate(self._slots[(namespace, layer_index)]):
                if handle is None or handle.tensor is None:
                    continue
                result.append(
                    (
                        namespace,
                        layer_index,
                        slot,
                        handle.release_attempted,
                        None if handle.release_error is None else type(handle.release_error).__name__,
                    )
                )
        return tuple(result)

    @property
    def preload_events(self) -> tuple[Qwen38BF4ResidentLoadEvent, ...]:
        return tuple(self._load_events)

    @property
    def cache_load_attempt_count(self) -> int:
        return sum(event.outcome == "attempt" for event in self._load_events)

    @property
    def cache_load_success_count(self) -> int:
        return sum(event.outcome == "success" for event in self._load_events)

    @property
    def registered_tensor_handle_count(self) -> int:
        return self._registered_handle_count

    @property
    def live_tensor_handle_count(self) -> int:
        handles = {
            id(handle): handle for slot_handles in self._slots.values() for handle in slot_handles if handle is not None
        }
        return sum(handle.tensor is not None for handle in handles.values())

    def _record_load_event(
        self,
        key: tuple[str, int],
        outcome: Literal["attempt", "success", "failure"],
        error: BaseException | None = None,
    ) -> None:
        self._load_events.append(
            Qwen38BF4ResidentLoadEvent(
                sequence=len(self._load_events),
                namespace=key[0],
                layer_index=key[1],
                outcome=outcome,
                error_type=None if error is None else type(error).__name__,
            )
        )

    def _assert_open(self, operation: str) -> None:
        if self._closed:
            raise RuntimeError(f"cannot {operation}: BF4 resident owner is closed")
        if self._poisoned:
            raise RuntimeError(
                f"cannot {operation}: BF4 resident owner is poisoned with "
                f"{len(self.unreleased_tensor_slots)} unreleased tensor slot(s)"
            )

    def _register_returned(
        self,
        key: tuple[str, int],
        returned: Any,
    ) -> tuple[tuple[_BF4ResidentHandle | None, ...], tuple[_BF4ResidentHandle, ...]]:
        """Register every returned object before validating the two-tensor pair."""

        if key in self._slots:
            raise RuntimeError(f"BF4 resident slot {key} is already registered")
        values = tuple(returned) if isinstance(returned, (tuple, list)) else (returned,)
        handles: list[_BF4ResidentHandle | None] = []
        created: list[_BF4ResidentHandle] = []
        for slot, tensor in enumerate(values):
            if tensor is None:
                handles.append(None)
                continue
            try:
                identity = _tensor_identity(tensor)
            except RuntimeError as identity_error:
                # The object is already owned, but without TTNN allocation
                # identity it cannot be safely deduplicated or deallocated.
                # Retain it terminally in the slot ledger and require mesh/
                # process teardown instead of falling back to Python id().
                handle = _BF4ResidentHandle(
                    tensor=tensor,
                    backing_identity=None,
                    first_key=key,
                    first_slot=slot,
                    release_attempted=True,
                    release_error=identity_error,
                )
                self._poisoned = True
                created.append(handle)
                self._registered_handle_count += 1
            else:
                handle = self._handles.get(identity)
                if handle is None:
                    handle = _BF4ResidentHandle(
                        tensor=tensor,
                        backing_identity=identity,
                        first_key=key,
                        first_slot=slot,
                    )
                    self._handles[identity] = handle
                    created.append(handle)
                    self._registered_handle_count += 1
            handles.append(handle)
        registered = tuple(handles)
        self._slots[key] = registered
        return registered, tuple(created)

    def _adopt_failed_cache_load(
        self,
        key: tuple[str, int],
        error: BF4CleanupError,
    ) -> tuple[tuple[_BF4ResidentHandle | None, ...], tuple[_BF4ResidentHandle, ...], BaseException | None,]:
        """Adopt every cache-load tensor whose release outcome is uncertain."""

        outcomes = error.tensor_cleanup_outcomes
        validation_error: BaseException | None = None
        if not outcomes or tuple(outcome.slot for outcome in outcomes) != tuple(range(len(outcomes))):
            validation_error = RuntimeError("BF4 cache cleanup error omitted its ordered tensor-slot ledger")
        # Check every outcome before filtering it.  In particular, a non-None
        # wrapper marked released is contradictory and must remain owned by a
        # poisoned resident ledger rather than disappearing from adoption.
        if validation_error is None:
            for outcome in outcomes:
                if outcome.released:
                    valid = outcome.tensor is None and outcome.release_attempted and outcome.release_error is None
                elif outcome.tensor is None:
                    valid = not outcome.release_attempted and outcome.release_error is None
                elif outcome.release_attempted:
                    valid = outcome.release_error is not None
                else:
                    valid = outcome.release_error is not None
                if not valid:
                    validation_error = RuntimeError("BF4 cache cleanup error contains contradictory release fields")
                    break
                if outcome.tensor is not None and not outcome.release_attempted:
                    try:
                        _tensor_identity(outcome.tensor)
                    except RuntimeError:
                        if outcome.release_error is None:
                            validation_error = RuntimeError(
                                "BF4 cache cleanup identity failure lacks its terminal error"
                            )
                            break
                    else:
                        validation_error = RuntimeError(
                            "BF4 cache cleanup retained an identifiable tensor without a release attempt"
                        )
                        break
        retained = tuple(outcome for outcome in outcomes if outcome.tensor is not None and not outcome.released)
        if validation_error is None:
            expected_unreleased: list[Any] = []
            expected_identities: set[Qwen38TensorBackingIdentity] = set()
            for outcome in retained:
                try:
                    identity = _tensor_identity(outcome.tensor)
                except RuntimeError:
                    pass
                else:
                    if identity in expected_identities:
                        continue
                    expected_identities.add(identity)
                expected_unreleased.append(outcome.tensor)
            if tuple(expected_unreleased) != error.unreleased_tensors:
                validation_error = RuntimeError("BF4 cache cleanup error unreleased-tensor ledger differs")
        # Validate first.  If the error ledger is malformed, retain every
        # non-released object anyway and poison the owner; never raise while
        # leaving device handles only in the exception's transient locals.
        returned = tuple(
            outcome.tensor
            if outcome.tensor is not None and (validation_error is not None or not outcome.released)
            else None
            for outcome in outcomes
        )
        handles, created = self._register_returned(key, returned)
        for outcome, handle in zip(outcomes, handles):
            if handle is None:
                continue
            handle.release_attempted = True
            handle.release_error = (
                outcome.release_error
                or validation_error
                or RuntimeError("BF4 cache cleanup outcome is not authoritative")
            )
        if validation_error is not None:
            self._poisoned = True
        elif not created or tuple(handle.tensor for handle in created) != error.unreleased_tensors:
            validation_error = RuntimeError("BF4 resident owner did not adopt every failed cache-load handle")
            self._poisoned = True
        return handles, created, validation_error

    def _release_handle(self, handle: _BF4ResidentHandle) -> BaseException | None:
        if handle.tensor is None:
            return None
        if handle.release_attempted:
            return handle.release_error
        handle.release_attempted = True
        try:
            ttnn.deallocate(handle.tensor)
        except BaseException as error:
            handle.release_error = error
            return error
        backing_identity = handle.backing_identity
        if backing_identity is None:
            handle.release_attempted = True
            if handle.release_error is None:
                handle.release_error = RuntimeError("BF4 tensor cannot be released without exact backing identity")
            return handle.release_error
        handle.tensor = None
        self._handles.pop(backing_identity, None)
        return None

    def _drop_released_slots(self, keys: tuple[tuple[str, int], ...]) -> None:
        for key in keys:
            handles = self._slots.get(key)
            if handles is not None and all(handle is None or handle.tensor is None for handle in handles):
                self._slots.pop(key, None)
                self._ready.discard(key)

    @staticmethod
    def _handle_cleanup_outcomes(
        handles: tuple[_BF4ResidentHandle, ...] | list[_BF4ResidentHandle],
    ) -> tuple[BF4TensorCleanupOutcome, ...]:
        return tuple(
            BF4TensorCleanupOutcome(
                slot=slot,
                tensor=handle.tensor,
                release_attempted=handle.release_attempted,
                released=handle.release_attempted and handle.release_error is None and handle.tensor is None,
                release_error=handle.release_error,
            )
            for slot, handle in enumerate(handles)
        )

    def _rollback_preload(
        self,
        keys: tuple[tuple[str, int], ...],
        created: tuple[_BF4ResidentHandle, ...],
        *,
        primary_error: BaseException,
    ) -> None:
        errors = tuple(error for handle in reversed(created) if (error := self._release_handle(handle)) is not None)
        created_identities = {id(handle) for handle in created}
        for key in keys:
            handles = self._slots.get(key)
            if handles is None:
                continue
            # A malformed return may alias a previously resident handle.  Drop
            # only the new alias; the earlier ready slot remains authoritative.
            retained = tuple(
                handle
                for handle in handles
                if handle is not None and id(handle) in created_identities and handle.tensor is not None
            )
            if retained:
                self._slots[key] = handles
            else:
                self._slots.pop(key, None)
            self._ready.discard(key)
        self._drop_released_slots(keys)
        if errors:
            self._poisoned = True
            raise BF4CleanupError(
                "BF4 resident preload failed and partial cleanup was incomplete",
                primary_error=primary_error,
                cleanup_errors=errors,
                tensor_cleanup_outcomes=self._handle_cleanup_outcomes(created),
            ) from primary_error

    def _preload(self, requested: tuple[tuple[str, int], ...]) -> None:
        self._assert_open("preload")
        if self._preloading:
            raise RuntimeError("BF4 resident preload is already active")
        if self._active is not None:
            raise RuntimeError(f"cannot preload while BF4 layer {self._active} is active")
        if len(set(requested)) != len(requested):
            raise ValueError("BF4 resident preload contains duplicate layer keys")
        for namespace, layer_index in requested:
            validate_bf4_layer_request(namespace, layer_index)

        missing = tuple(key for key in requested if key not in self._ready)
        if not missing:
            return

        registered_keys: list[tuple[str, int]] = []
        created_handles: list[_BF4ResidentHandle] = []
        self._preloading = True
        try:
            for namespace, layer_index in missing:
                key = (namespace, layer_index)
                self._record_load_event(key, "attempt")
                try:
                    returned = self.cache.load_layer(
                        self.mesh_device,
                        layer_index=layer_index,
                        namespace=namespace,
                    )
                    handles, created = self._register_returned(key, returned)
                    registered_keys.append(key)
                    created_handles.extend(created)
                    if (
                        type(returned) is not tuple
                        or len(handles) != 2
                        or any(handle is None for handle in handles)
                        or handles[0] is handles[1]
                        or len(created) != 2
                    ):
                        raise RuntimeError(f"BF4 resident preload {key} did not return two new distinct packed tensors")
                except BF4CleanupError as load_error:
                    # The cache owns tensors before it can return a pair.  If
                    # its best-effort cleanup fails, transfer those exact
                    # handles and their attempted-release state into this
                    # resident owner's terminal ledger before propagating.
                    _handles, created, adoption_error = self._adopt_failed_cache_load(key, load_error)
                    registered_keys.append(key)
                    created_handles.extend(created)
                    self._record_load_event(key, "failure", load_error)
                    if adoption_error is not None:
                        raise BF4CleanupError(
                            "BF4 cache cleanup ledger was malformed and resident ownership is terminal",
                            primary_error=load_error,
                            cleanup_errors=(adoption_error,),
                            tensor_cleanup_outcomes=self._handle_cleanup_outcomes(created),
                        ) from load_error
                    raise
                except BaseException as load_error:
                    self._record_load_event(key, "failure", load_error)
                    raise
                self._ready.add(key)
                self._record_load_event(key, "success")
        except BaseException as error:
            self._rollback_preload(
                tuple(registered_keys),
                tuple(created_handles),
                primary_error=error,
            )
            raise
        finally:
            self._preloading = False

    def preload_backbone(self) -> None:
        """Load the exact ordered 48-layer ordinary backbone once."""

        self._preload(_BACKBONE_RESIDENT_KEYS)

    def preload_mtp(self) -> None:
        """Load the exact released MTP layer once without reloading backbone."""

        self._preload(_MTP_RESIDENT_KEYS)

    def preload_all(self) -> None:
        """Load all 48 backbone pairs and the released MTP pair once."""

        self._preload(_ALL_RESIDENT_KEYS)

    @contextmanager
    def layer(self, layer_index: int, *, namespace: str = "backbone"):
        self._assert_open("borrow a layer")
        validate_bf4_layer_request(namespace, layer_index)
        if self._active is not None:
            raise RuntimeError(f"BF4 resident owner already has active layer {self._active}")
        key = (namespace, layer_index)
        handles = self._slots.get(key)
        if key not in self._ready or handles is None:
            raise RuntimeError(f"BF4 resident layer {key} was not preloaded; host streaming is disabled")
        if len(handles) != 2 or any(handle is None or handle.tensor is None for handle in handles):
            self._poisoned = True
            raise RuntimeError(f"BF4 resident layer {key} lost an owned tensor handle")
        tensors = tuple(handle.tensor for handle in handles if handle is not None)
        self._active = key
        try:
            yield tensors
        finally:
            self._active = None

    def close(self) -> None:
        """Release every retained pair once after no layer or preload is active."""

        if self._closed:
            return
        if self._active is not None:
            raise RuntimeError(f"cannot close BF4 resident owner while layer {self._active} is active")
        if self._preloading:
            raise RuntimeError("cannot close BF4 resident owner while preload is active")
        self._close_started = True
        ordered_handles: list[_BF4ResidentHandle] = []
        seen: set[int] = set()
        for key in reversed(self.owned_layers):
            for handle in reversed(self._slots[key]):
                if handle is None or id(handle) in seen:
                    continue
                seen.add(id(handle))
                ordered_handles.append(handle)
        new_errors = tuple(
            error
            for handle in ordered_handles
            if not handle.release_attempted and (error := self._release_handle(handle)) is not None
        )
        cleanup_errors = tuple(
            handle.release_error
            for handle in ordered_handles
            if handle.tensor is not None and handle.release_error is not None
        )
        self._drop_released_slots(self.owned_layers)
        if cleanup_errors:
            self._poisoned = True
            raise BF4CleanupError(
                "BF4 resident owner cleanup was incomplete",
                primary_error=None,
                cleanup_errors=cleanup_errors,
                tensor_cleanup_outcomes=self._handle_cleanup_outcomes(ordered_handles),
            ) from (new_errors[0] if new_errors else cleanup_errors[0])
        self._slots.clear()
        self._ready.clear()
        self._handles.clear()
        self._closed = True

    def deallocate(self) -> None:
        """Compatibility alias for explicit static-owner cleanup."""

        self.close()
