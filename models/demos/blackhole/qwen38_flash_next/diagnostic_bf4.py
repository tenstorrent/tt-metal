# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Read-only binding of a CPU-staged BF4 expert corpus (``tools/stage_full_bf4_cpu.py`` + ``tools/verify_full_bf4_cpu.py``).

The binder takes the corpus root and its verification record; the producer identity (the staging run's source
head, runtime and checkpoint digests) is what the record claims, checked against every staging summary and
per-layer evidence, and a caller may pin it (``producer``).  Every tensorbin is streamed through SHA-256 before a
record is exposed.  The production ``Qwen38BF4Cache`` is not involved: a server without a corpus converts the
experts itself on the first start.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

MODE = "diagnostic_non_promoting_read_only_bf4_binding"
STAGING_MODE = "diagnostic_non_promoting_cpu_bf4_staging"
VERIFICATION_MODE = "diagnostic_non_promoting_cpu_bf4_corpus_verification"

EXPECTED_SLOTS = tuple(("backbone", index) for index in range(48)) + (("mtp", 0),)
EXPECTED_MESH_SHAPE = (1, 4)
EXPECTED_MESH_COORDINATES = ((0, 0), (0, 1), (0, 2), (0, 3))
EXPECTED_EXPERT_RANGES = ((0, 128), (128, 256), (256, 384), (384, 512))
EXPECTED_RING_SIZE = 8
PRODUCER_IDENTITY_KEYS = ("source_head", "runtime", "checkpoint")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class _ArtifactSpec:
    filename: str
    bytes: int
    local_shape: tuple[int, ...]
    global_shape: tuple[int, ...]


ARTIFACT_SPECS = MappingProxyType(
    {
        "w0_w1": _ArtifactSpec(
            filename="w0_w1_dtype_BFLOAT4_B_layout_TILE.tensorbin",
            bytes=943_719_648,
            local_shape=(8, 1, 128, 10, 320, 128),
            global_shape=(8, 1, 512, 10, 320, 128),
        ),
        "w2": _ArtifactSpec(
            filename="w2_dtype_BFLOAT4_B_layout_TILE.tensorbin",
            bytes=471_860_448,
            local_shape=(8, 1, 128, 5, 320, 128),
            global_shape=(8, 1, 512, 5, 320, 128),
        ),
    }
)
# The corpus total follows the per-artifact pins: 49 slots x (w0_w1 + w2) = 69,363,424,704 bytes in the compact layout.
# The verification record's corpus.bytes is the on-disk sum the producer saw (the Aug-27 record: 106,819,608,000, its
# pre-compact layout); the binder compares the record to the disk and the pins to every artifact (2026-09-26).
EXPECTED_TOTAL_BYTES = len(EXPECTED_SLOTS) * sum(spec.bytes for spec in ARTIFACT_SPECS.values())


class DiagnosticBF4BindingError(RuntimeError):
    """The retained diagnostic corpus differs from its exact evidence."""


@dataclass(frozen=True)
class DiagnosticBF4Identity:
    identity_key: str
    namespace_key: str
    source_head: str
    checkpoint: Mapping[str, Any]
    runtime: Mapping[str, Any]
    mesh_shape: tuple[int, int]
    physical_ids: tuple[int, int, int, int]
    mesh_coordinates: tuple[tuple[int, int], ...]
    expert_ranges: tuple[tuple[int, int], ...]
    ring_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity_key": self.identity_key,
            "namespace_key": self.namespace_key,
            "source_head": self.source_head,
            "checkpoint": dict(self.checkpoint),
            "runtime": dict(self.runtime),
            "mesh_shape": list(self.mesh_shape),
            "physical_ids": list(self.physical_ids),
            "mesh_coordinates": [list(pair) for pair in self.mesh_coordinates],
            "expert_ranges": [list(pair) for pair in self.expert_ranges],
            "ring_size": self.ring_size,
        }


@dataclass(frozen=True)
class DiagnosticBF4Artifact:
    name: str
    path: Path
    relative_path: str
    sha256: str
    bytes: int
    dtype: str
    layout: str
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


@dataclass(frozen=True)
class DiagnosticBF4Record:
    namespace: str
    layer_index: int
    evidence_path: Path
    expert_ranges: tuple[tuple[int, int], ...]
    mesh_shape: tuple[int, int]
    physical_ids: tuple[int, int, int, int]
    mesh_coordinates: tuple[tuple[int, int], ...]
    ring_size: int
    w0_w1: DiagnosticBF4Artifact
    w2: DiagnosticBF4Artifact

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot": [self.namespace, self.layer_index],
            "evidence_path": str(self.evidence_path),
            "expert_ranges": [list(pair) for pair in self.expert_ranges],
            "mesh_shape": list(self.mesh_shape),
            "physical_ids": list(self.physical_ids),
            "mesh_coordinates": [list(pair) for pair in self.mesh_coordinates],
            "ring_size": self.ring_size,
            "artifacts": {"w0_w1": self.w0_w1.as_dict(), "w2": self.w2.as_dict()},
        }


@dataclass(frozen=True)
class DiagnosticBF4Corpus:
    root: Path
    verification_result: Path
    identity: DiagnosticBF4Identity
    records: Mapping[tuple[str, int], DiagnosticBF4Record]
    total_bytes: int  # the contract's total (the per-artifact pins)
    disk_bytes: int  # what the bound artifacts occupy on disk

    def get(self, namespace: str, layer_index: int) -> DiagnosticBF4Record:
        try:
            return self.records[(namespace, layer_index)]
        except KeyError as error:
            raise DiagnosticBF4BindingError(f"diagnostic BF4 slot is not bound: {(namespace, layer_index)}") from error

    def summary(self) -> dict[str, Any]:
        return {
            "mode": MODE,
            "production_qualification": False,
            "device_opened": False,
            "root": str(self.root),
            "verification_result": str(self.verification_result),
            "identity_key": self.identity.identity_key,
            "bound_record_count": len(self.records),
            "backbone_layers": [index for namespace, index in self.records if namespace == "backbone"],
            "mtp_layers": [index for namespace, index in self.records if namespace == "mtp"],
            "tensorbins": 2 * len(self.records),
            "bytes": self.disk_bytes,
            "contract_bytes": self.total_bytes,
            "all_payload_sha256_verified": True,
        }


@dataclass(frozen=True)
class _BindingContract:
    artifact_root: Path
    verification_result: Path
    slots: tuple[tuple[str, int], ...]
    artifact_specs: Mapping[str, _ArtifactSpec]
    checkpoint: Mapping[str, Any]
    runtime: Mapping[str, Any]
    source_head: str
    mesh_shape: tuple[int, int]
    # the producer's physical order; ``None`` adopts the first evidence record's, every other must agree
    physical_ids: tuple[int, int, int, int] | None
    mesh_coordinates: tuple[tuple[int, int], ...]
    expert_ranges: tuple[tuple[int, int], ...]
    ring_size: int
    total_bytes: int


def producer_identity(verification_result: Path) -> dict[str, Any]:
    """The staging identity a verification record claims (``source_head``, ``runtime``, ``checkpoint``)."""

    verification = _load_json(verification_result, label="corpus verification result")
    identity = verification.get("staging_identity")
    if type(identity) is not dict or set(identity) != set(PRODUCER_IDENTITY_KEYS):
        raise DiagnosticBF4BindingError(f"verification staging identity must carry {PRODUCER_IDENTITY_KEYS}")
    if (
        type(identity["source_head"]) is not str
        or type(identity["runtime"]) is not dict
        or type(identity["checkpoint"]) is not dict
    ):
        raise DiagnosticBF4BindingError("verification staging identity field types differ")
    return identity


def binding_contract(
    artifact_root: Path, verification_result: Path, *, producer: Mapping[str, Any] | None = None
) -> _BindingContract:
    """The corpus's own claim as the contract; ``producer`` (the same three keys) pins it."""

    claimed = producer_identity(verification_result)
    if producer is not None:
        _require_equal(dict(producer), claimed, "BF4 corpus producer identity")
    return _BindingContract(
        artifact_root=artifact_root,
        verification_result=verification_result,
        slots=EXPECTED_SLOTS,
        artifact_specs=ARTIFACT_SPECS,
        checkpoint=MappingProxyType(dict(claimed["checkpoint"])),
        runtime=MappingProxyType(dict(claimed["runtime"])),
        source_head=claimed["source_head"],
        mesh_shape=EXPECTED_MESH_SHAPE,
        physical_ids=None,
        mesh_coordinates=EXPECTED_MESH_COORDINATES,
        expert_ranges=EXPECTED_EXPERT_RANGES,
        ring_size=EXPECTED_RING_SIZE,
        total_bytes=EXPECTED_TOTAL_BYTES,
    )


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_typed_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if not _typed_equal(actual, expected):
        raise DiagnosticBF4BindingError(f"{label} differs: {actual!r} != {expected!r}")


def _canonical_file(path: Path, *, label: str) -> os.stat_result:
    if not path.is_absolute():
        raise DiagnosticBF4BindingError(f"{label} must be absolute: {path}")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DiagnosticBF4BindingError(f"{label} is unavailable: {path}") from error
    if resolved != path or not stat.S_ISREG(metadata.st_mode):
        raise DiagnosticBF4BindingError(f"{label} must be a canonical regular file: {path}")
    return metadata


def _stream_sha256(path: Path, *, expected_bytes: int | None = None, chunk_size: int = 8 << 20) -> str:
    """Hash one stable O_NOFOLLOW file using bounded memory."""

    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DiagnosticBF4BindingError(f"hash input is not a regular file: {path}")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise DiagnosticBF4BindingError(
                f"artifact byte count differs for {path}: {before.st_size} != {expected_bytes}"
            )
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, chunk_size)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        live = path.lstat()
        signature = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        if signature(before) != signature(after) or signature(after) != signature(live):
            raise DiagnosticBF4BindingError(f"file identity changed during streaming verification: {path}")
        return digest.hexdigest()
    except OSError as error:
        raise DiagnosticBF4BindingError(f"failed to stream artifact: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    metadata = _canonical_file(path, label=label)
    if metadata.st_size > 8 << 20:
        raise DiagnosticBF4BindingError(f"{label} exceeds the bounded JSON limit: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DiagnosticBF4BindingError(f"{label} is not valid JSON: {path}") from error
    if type(document) is not dict:
        raise DiagnosticBF4BindingError(f"{label} must contain a JSON object: {path}")
    return document


def _validate_tree(root: Path, expected_files: set[Path]) -> None:
    expected_dirs = {root}
    for path in expected_files:
        parent = path.parent
        while parent != root:
            expected_dirs.add(parent)
            parent = parent.parent
    live_files: set[Path] = set()
    live_dirs: set[Path] = {root}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in dirnames:
            path = parent / name
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise DiagnosticBF4BindingError(f"unexpected non-directory corpus entry: {path}")
            live_dirs.add(path)
        for name in filenames:
            path = parent / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise DiagnosticBF4BindingError(f"unexpected non-regular corpus entry: {path}")
            live_files.add(path)
    if live_files != expected_files or live_dirs != expected_dirs:
        missing = sorted(str(path) for path in expected_files - live_files)
        extra = sorted(str(path) for path in live_files - expected_files)
        extra_dirs = sorted(str(path) for path in live_dirs - expected_dirs)
        raise DiagnosticBF4BindingError(
            f"corpus tree differs: missing={missing} extra={extra} unexpected_dirs={extra_dirs}"
        )


def _disk_bytes(artifact_root: Path, contract: _BindingContract) -> int:
    """The bytes the contract's artifacts occupy on disk (canonical regular files; a missing one refuses here). The
    verification record's ``corpus.bytes`` is this sum as the producer saw it; the contract's own total (the compact
    pins) is checked per artifact, not against the record."""
    total = 0
    for slot in contract.slots:
        namespace, layer_index = slot
        for name, spec in contract.artifact_specs.items():
            path = artifact_root / namespace / f"layer-{layer_index:02d}" / spec.filename
            total += _canonical_file(path, label=f"{slot} artifact {name}").st_size
    return total


def _source_results(verification: dict[str, Any], contract: _BindingContract, disk_bytes: int) -> list[dict[str, Any]]:
    _require_equal(verification.get("mode"), VERIFICATION_MODE, "verification mode")
    for key, expected in (
        ("status", "pass"),
        ("production_qualification", False),
        ("device_opened", False),
        ("device_locks_acquired", False),
        ("artifact_root", str(contract.artifact_root)),
        (
            "corpus",
            {
                "slots": len(contract.slots),
                "backbone_layers": list(range(48)),
                "mtp_layers": [0],
                "tensorbins": 2 * len(contract.slots),
                "bytes": disk_bytes,
            },
        ),
    ):
        _require_equal(verification.get(key), expected, f"verification {key}")
    expected_identity = {
        "source_head": contract.source_head,
        "runtime": dict(contract.runtime),
        "checkpoint": dict(contract.checkpoint),
    }
    _require_equal(verification.get("staging_identity"), expected_identity, "verification staging identity")
    summaries = verification.get("source_summaries")
    if type(summaries) is not list or len(summaries) != 2:
        raise DiagnosticBF4BindingError("verification must bind exactly two staging summaries")
    results: list[dict[str, Any]] = []
    for index, summary_ref in enumerate(summaries):
        if type(summary_ref) is not dict or set(summary_ref) != {"path", "sha256"}:
            raise DiagnosticBF4BindingError(f"verification summary reference {index} schema differs")
        summary_path = Path(summary_ref["path"])
        digest = _stream_sha256(summary_path)
        if digest != summary_ref["sha256"]:
            raise DiagnosticBF4BindingError(f"staging summary digest differs: {summary_path}")
        summary = _load_json(summary_path, label=f"staging summary {index}")
        for key, expected in (
            ("mode", STAGING_MODE),
            ("status", "pass"),
            ("production_qualification", False),
            ("device_opened", False),
            ("device_locks_acquired", False),
            ("source_head", contract.source_head),
            ("runtime", dict(contract.runtime)),
            ("checkpoint", dict(contract.checkpoint)),
        ):
            _require_equal(summary.get(key), expected, f"staging summary {index} {key}")
        if type(summary.get("results")) is not list:
            raise DiagnosticBF4BindingError(f"staging summary {index} results schema differs")
        results.extend(summary["results"])
    return results


def _bind(
    artifact_root: Path,
    verification_result: Path,
    *,
    contract: _BindingContract,
    hash_file: Callable[..., str] = _stream_sha256,
) -> DiagnosticBF4Corpus:
    if artifact_root != contract.artifact_root or verification_result != contract.verification_result:
        raise DiagnosticBF4BindingError("diagnostic binder paths differ from its contract")
    if artifact_root.resolve(strict=True) != artifact_root or not artifact_root.is_dir():
        raise DiagnosticBF4BindingError(f"artifact root must be a canonical directory: {artifact_root}")
    _canonical_file(verification_result, label="corpus verification result")
    verification = _load_json(verification_result, label="corpus verification result")
    disk_bytes = _disk_bytes(artifact_root, contract)
    results = _source_results(verification, contract, disk_bytes)

    result_by_slot: dict[tuple[str, int], dict[str, Any]] = {}
    for result in results:
        if type(result) is not dict or result.get("status") != "staged":
            raise DiagnosticBF4BindingError("staging result schema/status differs")
        raw_slot = result.get("slot")
        if (
            type(raw_slot) is not list
            or len(raw_slot) != 2
            or type(raw_slot[0]) is not str
            or type(raw_slot[1]) is not int
        ):
            raise DiagnosticBF4BindingError(f"staging result slot schema differs: {raw_slot!r}")
        slot = (raw_slot[0], raw_slot[1])
        if slot in result_by_slot:
            raise DiagnosticBF4BindingError(f"duplicate staging slot: {slot}")
        result_by_slot[slot] = result
    if set(result_by_slot) != set(contract.slots):
        raise DiagnosticBF4BindingError("staging slot coverage differs from exact backbone-0:47 plus mtp-0 contract")

    expected_files: set[Path] = set()
    records: dict[tuple[str, int], DiagnosticBF4Record] = {}
    total_bytes = 0
    physical_ids = contract.physical_ids
    for slot in contract.slots:
        namespace, layer_index = slot
        result = result_by_slot[slot]
        evidence_path = Path(result.get("evidence", ""))
        evidence = _load_json(evidence_path, label=f"{namespace}:{layer_index} staging evidence")
        if physical_ids is None:
            claimed_ids = (evidence.get("ownership") or {}).get("physical_ids")
            if (
                type(claimed_ids) is not list
                or len(claimed_ids) != 4
                or any(type(value) is not int or value < 0 for value in claimed_ids)
                or len(set(claimed_ids)) != 4
            ):
                raise DiagnosticBF4BindingError(
                    f"{slot} evidence physical ids must be four distinct ints: {claimed_ids!r}"
                )
            physical_ids = tuple(claimed_ids)
        for key, expected in (
            ("mode", STAGING_MODE),
            ("production_qualification", False),
            ("device_opened", False),
            ("device_locks_acquired", False),
            ("source_head", contract.source_head),
            ("runtime", dict(contract.runtime)),
            ("checkpoint", dict(contract.checkpoint)),
            ("slot", [namespace, layer_index]),
            (
                "ownership",
                {
                    "expert_ranges": [list(pair) for pair in contract.expert_ranges],
                    "mesh_coordinates": [list(pair) for pair in contract.mesh_coordinates],
                    "mesh_shape": list(contract.mesh_shape),
                    "physical_ids": list(physical_ids),
                    "ring_size": contract.ring_size,
                },
            ),
        ):
            _require_equal(evidence.get(key), expected, f"{namespace}:{layer_index} evidence {key}")
        _require_equal(evidence.get("artifacts"), result.get("artifacts"), f"{namespace}:{layer_index} artifacts")

        device_shards = evidence.get("device_shards")
        if type(device_shards) is not list or len(device_shards) != 4:
            raise DiagnosticBF4BindingError(f"{slot} device-shard count differs")
        for device_index, shard in enumerate(device_shards):
            if type(shard) is not dict:
                raise DiagnosticBF4BindingError(f"{slot} device shard {device_index} schema differs")
            for key, expected in (
                ("device_index", device_index),
                ("mesh_coordinate", list(contract.mesh_coordinates[device_index])),
                ("physical_id", physical_ids[device_index]),
                ("expert_range", list(contract.expert_ranges[device_index])),
            ):
                _require_equal(shard.get(key), expected, f"{slot} device shard {device_index} {key}")

        artifacts_document = result.get("artifacts")
        if type(artifacts_document) is not dict or set(artifacts_document) != set(contract.artifact_specs):
            raise DiagnosticBF4BindingError(f"{slot} artifact names differ")
        artifacts: dict[str, DiagnosticBF4Artifact] = {}
        for name, spec in contract.artifact_specs.items():
            raw = artifacts_document[name]
            if type(raw) is not dict:
                raise DiagnosticBF4BindingError(f"{slot} artifact {name} schema differs")
            path = artifact_root / namespace / f"layer-{layer_index:02d}" / spec.filename
            # the on-disk size against the pin first: a corpus of another layout is named as such, before its staging
            # evidence (which describes that other layout) is compared field by field
            on_disk = _canonical_file(path, label=f"{slot} artifact {name}").st_size
            if on_disk != spec.bytes:
                raise DiagnosticBF4BindingError(
                    f"{slot} artifact {name} is {on_disk} bytes on disk against the contract's {spec.bytes}: the retained "
                    "corpus is not the compact layout the contract pins (the Aug-27 corpus predates it, 2026-09-24) -- "
                    "pack a compact corpus or run corpus-free"
                )
            expected = {
                "bytes": spec.bytes,
                "dtype": "BFLOAT4_B",
                "global_shape": list(spec.global_shape),
                "layout": "TILE",
                "local_shape": list(spec.local_shape),
                "path": str(path),
            }
            for key, value in expected.items():
                _require_equal(raw.get(key), value, f"{slot} artifact {name} {key}")
            digest = raw.get("sha256")
            if type(digest) is not str or SHA256_PATTERN.fullmatch(digest) is None:
                raise DiagnosticBF4BindingError(f"{slot} artifact {name} digest schema differs")
            _canonical_file(path, label=f"{slot} artifact {name}")
            actual_digest = hash_file(path, expected_bytes=spec.bytes)
            if actual_digest != digest:
                raise DiagnosticBF4BindingError(f"{slot} artifact {name} SHA-256 differs: {actual_digest} != {digest}")
            expected_files.add(path)
            total_bytes += spec.bytes
            artifacts[name] = DiagnosticBF4Artifact(
                name=name,
                path=path,
                relative_path=str(path.relative_to(artifact_root)),
                sha256=digest,
                bytes=spec.bytes,
                dtype="BFLOAT4_B",
                layout="TILE",
                global_shape=spec.global_shape,
                local_shape=spec.local_shape,
            )
        records[slot] = DiagnosticBF4Record(
            namespace=namespace,
            layer_index=layer_index,
            evidence_path=evidence_path,
            expert_ranges=contract.expert_ranges,
            mesh_shape=contract.mesh_shape,
            physical_ids=physical_ids,
            mesh_coordinates=contract.mesh_coordinates,
            ring_size=contract.ring_size,
            w0_w1=artifacts["w0_w1"],
            w2=artifacts["w2"],
        )

    _validate_tree(artifact_root, expected_files)
    if total_bytes != contract.total_bytes:
        raise DiagnosticBF4BindingError(f"corpus bytes differ: {total_bytes} != {contract.total_bytes}")
    identity = DiagnosticBF4Identity(
        identity_key=artifact_root.name,
        namespace_key=artifact_root.parent.name,
        source_head=contract.source_head,
        checkpoint=MappingProxyType(dict(contract.checkpoint)),
        runtime=MappingProxyType(dict(contract.runtime)),
        mesh_shape=contract.mesh_shape,
        physical_ids=physical_ids,
        mesh_coordinates=contract.mesh_coordinates,
        expert_ranges=contract.expert_ranges,
        ring_size=contract.ring_size,
    )
    return DiagnosticBF4Corpus(
        root=artifact_root,
        verification_result=verification_result,
        identity=identity,
        records=MappingProxyType(records),
        total_bytes=total_bytes,
        disk_bytes=disk_bytes,
    )


def bind_diagnostic_bf4_corpus(
    artifact_root: str | Path,
    verification_result: str | Path,
    *,
    producer: Mapping[str, Any] | None = None,
) -> DiagnosticBF4Corpus:
    """Bind a completed corpus without producing a manifest or tensors; ``producer`` pins its staging identity."""

    root = Path(artifact_root)
    verification = Path(verification_result)
    if not root.is_absolute() or not verification.is_absolute():
        raise DiagnosticBF4BindingError("diagnostic corpus and verification paths must be absolute")
    return _bind(root, verification, contract=binding_contract(root, verification, producer=producer))


__all__ = [
    "DiagnosticBF4Artifact",
    "DiagnosticBF4BindingError",
    "DiagnosticBF4Corpus",
    "DiagnosticBF4Identity",
    "DiagnosticBF4Record",
    "EXPECTED_SLOTS",
    "PRODUCER_IDENTITY_KEYS",
    "bind_diagnostic_bf4_corpus",
    "binding_contract",
    "producer_identity",
]
