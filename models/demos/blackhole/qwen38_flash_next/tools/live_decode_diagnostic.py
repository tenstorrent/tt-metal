# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Live construction of the ordinary Qwen3.8 decode on an already-open mesh.

The CPU preparation binds the checkpoint, the cache roots and the runtime identity (the tt-metal commit of the
checkout that built the loaded ``ttnn`` and the extension's digest) into the builder provenance; the live
construction creates the exact :class:`Qwen38TTNNBuilder` on the caller's ``(1, 4)`` mesh and requires all 48
backbone BF4 records plus MTP layer 0.  Without a corpus the builder's production ``Qwen38BF4Cache`` converts the
routed experts from the checkpoint on the first start and reads its own cache afterwards; with a CPU-staged corpus
(``bf4_corpus_root`` + its verification record, ``diagnostic_bf4``) a read-only adapter serves the corpus's records
instead.  This module never discovers, opens, closes or resets a device and never configures fabric.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    CHECKPOINT_FILE_MANIFEST_SHA256,
    INDEX_SHA256,
    Qwen38Checkpoint,
)
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256, Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.diagnostic_bf4 import (
    EXPECTED_SLOTS,
    DiagnosticBF4Artifact,
    DiagnosticBF4Corpus,
    bind_diagnostic_bf4_corpus,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import (
    BF4Artifact,
    BF4LayerRecord,
    Qwen38BF4Cache,
    Qwen38BF4ResidentSet,
    qualify_live_bf4_ring,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    BACKBONE_LAYERS,
    RESIDENT_MAX_QSA_CACHE_CAPACITY,
    RESIDENT_QSA_CACHE_CAPACITIES,
    Qwen38BuildProvenance,
    Qwen38CacheRoots,
    Qwen38TTNNBuilder,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import Qwen38MeshContract, TensorPlacement
from models.demos.blackhole.qwen38_flash_next.ttnn.decode import (
    Qwen38OrdinaryDecodeSession,
    Qwen38OrdinarySessionStatus,
    Qwen38OrdinaryStopReason,
    Qwen38OrdinaryToken,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.diagnostic_decode import Qwen38NonqualifyingDecodedToken
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    PINNED_CHECKPOINT_REVISION,
    PINNED_TENSOR_MANIFEST_SHA256,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.model import LayerObserver

MODE = "diagnostic_non_promoting"
EXPECTED_MESH_SHAPE = (1, 4)
EXPECTED_RING_SIZE = 8
EXPECTED_RECORD_COUNT = 49
EXPECTED_TENSORBIN_COUNT = 98
BF4_CONSUMER_COMPATIBILITY_SCHEMA = "qwen38-bf4-consumer-compatibility-v3"
# The retained two-step CPU oracle (the greedy tokens of prompt "2" through the torch reference, 48 layers,
# with the layer-0/1 hidden digests and expert selections): shipped in tree, the first live token is checked against it.
DEFAULT_CPU_ORACLE = Path(__file__).with_name("acceptance") / "cpu-oracle-two-step.jsonl"
DIAGNOSTIC_PROMPT_TEXT = "2"
DIAGNOSTIC_INPUT_TOKEN_ID = 17
DIAGNOSTIC_EXPECTED_TOKEN_ID = 15
DIAGNOSTIC_ORACLE_INPUT_TOKEN_IDS = (DIAGNOSTIC_INPUT_TOKEN_ID, DIAGNOSTIC_EXPECTED_TOKEN_ID)
DIAGNOSTIC_ORACLE_OUTPUT_TOKEN_IDS = (DIAGNOSTIC_EXPECTED_TOKEN_ID, 16)
DIAGNOSTIC_LAYER0_HIDDEN_SHA256 = "33016923963b2b88c83b39368ecfa23bb32de83cb51e2ff25096173935da1aaf"
DIAGNOSTIC_LAYER0_SELECTED_EXPERTS = (215, 143, 436, 151, 449, 453, 287, 90, 84, 20)
DIAGNOSTIC_LAYER1_HIDDEN_SHA256 = "c4c10a8cebb37b2694ec78a8929d48aa4b15f218759a2f698b0fec77f7e8bc4e"
DIAGNOSTIC_LAYER1_SELECTED_EXPERTS = (19, 157, 320, 225, 165, 205, 60, 131, 104, 86)
DIAGNOSTIC_EOS_TOKEN_IDS = (248_046, 248_044)
MAX_RETAINED_CPU_ORACLE_BYTES = 1 << 20

Marker = Callable[[str], None]
FileSignature = tuple[int, int, int, int, int]


def _canonical_physical_ids(physical_ids: Any) -> tuple[int, int, int, int]:
    if (
        not isinstance(physical_ids, tuple)
        or len(physical_ids) != 4
        or any(isinstance(value, bool) or type(value) is not int or value < 0 for value in physical_ids)
        or len(set(physical_ids)) != 4
    ):
        raise ValueError(f"physical_ids must be four distinct non-negative ints, got {physical_ids!r}")
    return physical_ids


def _require_hex(value: Any, length: int, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase {length}-hex")
    return value


class LiveDecodeDiagnosticError(RuntimeError):
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


def _canonical_file_signature(path: Path, *, expected_bytes: int) -> FileSignature:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise LiveDecodeDiagnosticError(f"diagnostic BF4 artifact is unavailable: {path}") from error
    if resolved != path or not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_bytes:
        raise LiveDecodeDiagnosticError(
            f"diagnostic BF4 artifact identity/size differs: {path} bytes={metadata.st_size}/{expected_bytes}"
        )
    return _signature(metadata)


def _stream_sha256(path: Path, *, expected_bytes: int | None = None) -> str:
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (expected_bytes is not None and before.st_size != expected_bytes):
            raise LiveDecodeDiagnosticError(f"hash input identity/size differs: {path}")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 8 << 20):
            digest.update(block)
        after = os.fstat(descriptor)
        if _signature(before) != _signature(after) or _signature(after) != _signature(path.lstat()):
            raise LiveDecodeDiagnosticError(f"hash input changed during verification: {path}")
        return digest.hexdigest()
    except OSError as error:
        raise LiveDecodeDiagnosticError(f"cannot hash pinned file: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True)
class Qwen38RetainedCPUTokenOracle:
    """Exact token contract extracted from the retained two-step CPU oracle."""

    source: Path
    sha256: str
    input_token_ids: tuple[int, int]
    output_token_ids: tuple[int, int]
    layer0_hidden_sha256: str
    layer0_selected_experts: tuple[int, ...]
    layer1_hidden_sha256: str
    layer1_selected_experts: tuple[int, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "sha256": self.sha256,
            "raw_text": DIAGNOSTIC_PROMPT_TEXT,
            "input_token_ids": list(self.input_token_ids),
            "output_token_ids": list(self.output_token_ids),
            "first_step": {
                "input_token_id": self.input_token_ids[0],
                "expected_token_id": self.output_token_ids[0],
            },
            "continuation_step": {
                "input_token_id": self.input_token_ids[1],
                "expected_token_id": self.output_token_ids[1],
            },
            "first_step_layer0": {
                "hidden_sha256": self.layer0_hidden_sha256,
                "selected_experts": list(self.layer0_selected_experts),
            },
            "first_step_layer1": {
                "hidden_sha256": self.layer1_hidden_sha256,
                "selected_experts": list(self.layer1_selected_experts),
            },
        }


def load_retained_cpu_token_oracle(path: Path | None = None) -> Qwen38RetainedCPUTokenOracle:
    """Read and validate the two-step CPU oracle log (its lifecycle, layer coverage, token chain and the layer-0/1
    anchors) without invoking its producer; the digest is recorded, not pinned.  ``None``: the shipped oracle."""

    path = DEFAULT_CPU_ORACLE if path is None else Path(path)
    try:
        metadata = path.lstat()
        source = path.resolve(strict=True)
    except OSError as error:
        raise LiveDecodeDiagnosticError(f"retained CPU oracle is unavailable: {path}") from error
    if not stat.S_ISREG(source.lstat().st_mode):
        raise LiveDecodeDiagnosticError(f"retained CPU oracle is not a regular file: {path}")
    if metadata.st_size <= 0 or metadata.st_size > MAX_RETAINED_CPU_ORACLE_BYTES:
        raise LiveDecodeDiagnosticError(f"retained CPU oracle size is invalid: {metadata.st_size}")
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise LiveDecodeDiagnosticError("retained CPU oracle cannot be read") from error
    digest = hashlib.sha256(payload).hexdigest()
    try:
        events = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LiveDecodeDiagnosticError("retained CPU oracle is not complete JSONL") from error
    if not events or any(not isinstance(event, dict) for event in events):
        raise LiveDecodeDiagnosticError("retained CPU oracle contains a non-object event")
    decodes = sorted((event for event in events if event.get("event") == "decode"), key=lambda event: event.get("step"))
    layer_events = [event for event in events if event.get("event") == "layer"]
    layer_keys = [(event.get("step"), event.get("layer")) for event in layer_events]
    expected_layer_keys = {(step, layer) for step in range(2) for layer in range(48)}
    starts = [event for event in events if event.get("event") == "start"]
    completes = [event for event in events if event.get("event") == "complete"]
    if (
        len(starts) != 1
        or starts[0].get("layers") != 48
        or starts[0].get("execution") != "CPU integration oracle; no TT device opened"
        or len(decodes) != 2
        or len(completes) != 1
        or completes[0].get("steps") != 2
        or completes[0].get("final_position") != 2
        or len(layer_events) != 96
        or len(set(layer_keys)) != 96
        or set(layer_keys) != expected_layer_keys
    ):
        raise LiveDecodeDiagnosticError("retained CPU oracle lifecycle/layer coverage differs")
    inputs = tuple(event.get("input_token") for event in decodes)
    outputs = tuple(event.get("output_token") for event in decodes)
    if inputs != DIAGNOSTIC_ORACLE_INPUT_TOKEN_IDS or outputs != DIAGNOSTIC_ORACLE_OUTPUT_TOKEN_IDS:
        raise LiveDecodeDiagnosticError(f"retained CPU oracle token chain differs: {inputs} -> {outputs}")
    layer0_events = [event for event in layer_events if event.get("step") == 0 and event.get("layer") == 0]
    if len(layer0_events) != 1:
        raise LiveDecodeDiagnosticError("retained CPU oracle lacks one unique first-step layer-0 anchor")
    layer0 = layer0_events[0]
    if (
        layer0.get("hidden_sha256") != DIAGNOSTIC_LAYER0_HIDDEN_SHA256
        or tuple(layer0.get("selected_experts", ())) != DIAGNOSTIC_LAYER0_SELECTED_EXPERTS
    ):
        raise LiveDecodeDiagnosticError("retained CPU oracle first-step layer-0 anchor differs")
    layer1_events = [event for event in layer_events if event.get("step") == 0 and event.get("layer") == 1]
    if len(layer1_events) != 1:
        raise LiveDecodeDiagnosticError("retained CPU oracle lacks one unique first-step layer-1 anchor")
    layer1 = layer1_events[0]
    if (
        layer1.get("hidden_sha256") != DIAGNOSTIC_LAYER1_HIDDEN_SHA256
        or tuple(layer1.get("selected_experts", ())) != DIAGNOSTIC_LAYER1_SELECTED_EXPERTS
    ):
        raise LiveDecodeDiagnosticError("retained CPU oracle first-step layer-1 anchor differs")
    return Qwen38RetainedCPUTokenOracle(
        source=source,
        sha256=digest,
        input_token_ids=DIAGNOSTIC_ORACLE_INPUT_TOKEN_IDS,
        output_token_ids=DIAGNOSTIC_ORACLE_OUTPUT_TOKEN_IDS,
        layer0_hidden_sha256=DIAGNOSTIC_LAYER0_HIDDEN_SHA256,
        layer0_selected_experts=DIAGNOSTIC_LAYER0_SELECTED_EXPERTS,
        layer1_hidden_sha256=DIAGNOSTIC_LAYER1_HIDDEN_SHA256,
        layer1_selected_experts=DIAGNOSTIC_LAYER1_SELECTED_EXPERTS,
    )


def _require_retained_oracle_first_step(
    oracle: Qwen38RetainedCPUTokenOracle,
    *,
    input_token_id: int,
    output_token_id: int,
) -> None:
    if type(oracle) is not Qwen38RetainedCPUTokenOracle:
        raise TypeError("oracle must be Qwen38RetainedCPUTokenOracle")
    expected_input = oracle.input_token_ids[0]
    expected_output = oracle.output_token_ids[0]
    if input_token_id != expected_input or output_token_id != expected_output:
        raise LiveDecodeDiagnosticError(
            f"ordinary first token differs from retained CPU oracle: "
            f"{input_token_id}->{output_token_id} != {expected_input}->{expected_output}"
        )


def _snapshot_corpus(corpus: DiagnosticBF4Corpus) -> Mapping[Path, FileSignature]:
    if tuple(corpus.records) != EXPECTED_SLOTS or len(corpus.records) != EXPECTED_RECORD_COUNT:
        raise LiveDecodeDiagnosticError("diagnostic BF4 corpus is not exact backbone-0:47 plus mtp-0")
    if corpus.identity.mesh_shape != EXPECTED_MESH_SHAPE or corpus.identity.ring_size != EXPECTED_RING_SIZE:
        raise LiveDecodeDiagnosticError(
            f"diagnostic BF4 corpus topology differs: mesh {corpus.identity.mesh_shape} ring {corpus.identity.ring_size} "
            f"vs {EXPECTED_MESH_SHAPE} / {EXPECTED_RING_SIZE}"
        )
    signatures: dict[Path, FileSignature] = {}
    inodes: set[tuple[int, int]] = set()
    for slot in EXPECTED_SLOTS:
        record = corpus.get(*slot)
        if (record.namespace, record.layer_index) != slot:
            raise LiveDecodeDiagnosticError(f"diagnostic BF4 record identity differs at slot {slot}")
        for artifact in (record.w0_w1, record.w2):
            signature = _canonical_file_signature(artifact.path, expected_bytes=artifact.bytes)
            inode = signature[:2]
            if artifact.path in signatures or inode in inodes:
                raise LiveDecodeDiagnosticError(f"duplicate diagnostic BF4 artifact identity: {artifact.path}")
            signatures[artifact.path] = signature
            inodes.add(inode)
    if len(signatures) != EXPECTED_TENSORBIN_COUNT:
        raise LiveDecodeDiagnosticError(
            f"diagnostic BF4 corpus exposed {len(signatures)} tensorbins, expected {EXPECTED_TENSORBIN_COUNT}"
        )
    return MappingProxyType(signatures)


@dataclass(frozen=True)
class Qwen38BF4ConsumerCompatibility:
    """A CPU-staged corpus consumed by this checkout's runtime: the producer identity the corpus records, the
    consumer identity as prepared, and the topology both sides must share (mesh shape, ring size, expert ranges).
    The corpus's bytes are position-keyed by mesh coordinate, so the consumer's physical order may differ from
    the producer's (the QuietBox opens 0, 2, 1, 3; a four-chip host 1, 0, 2, 3)."""

    producer_source_head: str
    producer_runtime: Mapping[str, Any]
    producer_checkpoint: Mapping[str, Any]
    producer_physical_ids: tuple[int, int, int, int]
    consumer_provenance_key: str
    consumer_tt_metal_sha: str
    consumer_runtime_extension: Path
    consumer_runtime_sha256: str
    consumer_physical_ids: tuple[int, int, int, int]
    expert_residency: str
    qsa_cache_capacity: int
    schema: str = BF4_CONSUMER_COMPATIBILITY_SCHEMA
    mesh_shape: tuple[int, int] = EXPECTED_MESH_SHAPE
    ring_size: int = EXPECTED_RING_SIZE

    def __post_init__(self) -> None:
        _require_hex(self.consumer_provenance_key, 64, "BF4 compatibility consumer provenance key")
        _require_hex(self.consumer_tt_metal_sha, 40, "BF4 compatibility consumer tt-metal sha")
        _require_hex(self.consumer_runtime_sha256, 64, "BF4 compatibility consumer runtime digest")
        _canonical_physical_ids(self.producer_physical_ids)
        _canonical_physical_ids(self.consumer_physical_ids)
        if self.expert_residency != "resident" or self.qsa_cache_capacity not in RESIDENT_QSA_CACHE_CAPACITIES:
            raise LiveDecodeDiagnosticError(
                f"BF4 compatibility admits the resident builds {RESIDENT_QSA_CACHE_CAPACITIES}, got "
                f"{self.expert_residency}/C{self.qsa_cache_capacity}"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "producer_source_head": self.producer_source_head,
            "producer_runtime": dict(self.producer_runtime),
            "producer_checkpoint": dict(self.producer_checkpoint),
            "producer_physical_ids": list(self.producer_physical_ids),
            "consumer_provenance_key": self.consumer_provenance_key,
            "consumer_tt_metal_sha": self.consumer_tt_metal_sha,
            "consumer_runtime_extension": str(self.consumer_runtime_extension),
            "consumer_runtime_sha256": self.consumer_runtime_sha256,
            "consumer_physical_ids": list(self.consumer_physical_ids),
            "expert_residency": self.expert_residency,
            "qsa_cache_capacity": self.qsa_cache_capacity,
            "mesh_shape": list(self.mesh_shape),
            "ring_size": self.ring_size,
        }

    @property
    def key(self) -> str:
        payload = json.dumps(self._payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def validate_corpus(self, corpus: DiagnosticBF4Corpus) -> None:
        """The corpus is the producer side: identity, topology, and the same checkpoint (by digest) as this build."""

        identity = corpus.identity
        if (
            identity.source_head != self.producer_source_head
            or dict(identity.runtime) != dict(self.producer_runtime)
            or dict(identity.checkpoint) != dict(self.producer_checkpoint)
            or identity.physical_ids != self.producer_physical_ids
        ):
            raise LiveDecodeDiagnosticError("BF4 corpus differs from the producer side of compatibility")
        if (
            identity.mesh_shape != self.mesh_shape
            or identity.ring_size != self.ring_size
            or identity.expert_ranges != ((0, 128), (128, 256), (256, 384), (384, 512))
        ):
            raise LiveDecodeDiagnosticError(f"BF4 corpus topology differs: {identity.as_dict()}")
        expected_checkpoint = {
            "revision": PINNED_CHECKPOINT_REVISION,
            "config_sha256": CONFIG_SHA256,
            "index_sha256": INDEX_SHA256,
            "file_manifest_sha256": CHECKPOINT_FILE_MANIFEST_SHA256,
            "tensor_manifest_sha256": PINNED_TENSOR_MANIFEST_SHA256,
        }
        differences = {
            name: (identity.checkpoint.get(name), value)
            for name, value in expected_checkpoint.items()
            if identity.checkpoint.get(name) != value
        }
        if differences:
            raise LiveDecodeDiagnosticError(f"BF4 corpus was staged from another checkpoint: {differences}")

    def validate_live(self, *, builder: Any, production_cache: Qwen38BF4Cache) -> dict[str, str]:
        """The live builder is the consumer side; returns the live identity keys for the report."""

        differences = [
            f"{name} actual={actual} expected={expected}"
            for name, actual, expected in (
                ("provenance key", builder.provenance.key, self.consumer_provenance_key),
                ("physical order", production_cache.identity.physical_ids, self.consumer_physical_ids),
                ("ring size", production_cache.identity.ring_size, self.ring_size),
                ("mesh shape", tuple(production_cache.identity.mesh_shape), self.mesh_shape),
                ("expert residency", builder.identity.expert_residency, self.expert_residency),
                ("allocated context", builder.identity.qsa_cache_capacity, self.qsa_cache_capacity),
            )
            if actual != expected
        ]
        if differences:
            raise LiveDecodeDiagnosticError(
                "live builder differs from the consumer side of compatibility: " + "; ".join(differences)
            )
        return {
            "builder_identity_key": builder.identity.key,
            "bf4_identity_key": production_cache.identity.key,
            "dram_bank_ring_order": list(production_cache.identity.dram_bank_ring_order),
        }

    def summary(self) -> dict[str, Any]:
        return {"key": self.key, **self._payload()}


@dataclass(frozen=True)
class Qwen38PreparedLiveDecodeDiagnostic:
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
    # the CPU-staged corpus, when one is bound; otherwise the production cache converts on the first start
    corpus: DiagnosticBF4Corpus | None = None
    artifact_signatures: Mapping[Path, FileSignature] | None = None
    bf4_consumer_compatibility: Qwen38BF4ConsumerCompatibility | None = None

    @property
    def runtime_profile(self) -> str:
        return f"resident-c{self.qsa_cache_capacity}"

    def summary(self) -> dict[str, Any]:
        return {
            "mode": MODE,
            "production_qualification": False,
            "performance_qualification": False,
            "hardware_opened_by_module": False,
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
            "bf4_source": "production_cache" if self.corpus is None else "cpu_staged_corpus",
            "bound_records": 0 if self.corpus is None else len(self.corpus.records),
            "bound_tensorbins": 0 if self.artifact_signatures is None else len(self.artifact_signatures),
            "corpus_identity_key": None if self.corpus is None else self.corpus.identity.identity_key,
            "bf4_consumer_compatibility": (
                None if self.bf4_consumer_compatibility is None else self.bf4_consumer_compatibility.summary()
            ),
        }


def prepare_live_decode_diagnostic(
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
    bf4_corpus_root: Path | None = None,
    bf4_corpus_verification: Path | None = None,
    bf4_producer: Mapping[str, Any] | None = None,
    marker: Marker | None = None,
) -> Qwen38PreparedLiveDecodeDiagnostic:
    """Bind every software input before the caller opens its mesh.

    ``tt_metal_sha`` / ``runtime_extension`` / ``runtime_sha256`` are the admitted runtime identity (the checkout's
    head and its extension); the extension must be the one this interpreter loaded.  ``physical_ids`` is the order
    the host's mesh opens in (the hardware profile's route).  ``bf4_corpus_root`` with its verification record binds
    a CPU-staged corpus; without it the production cache under ``routed_bf4_scratch_root`` is used and filled on
    the first start.
    """

    _require_hex(tt_metal_sha, 40, "tt_metal_sha")
    _require_hex(runtime_sha256, 64, "runtime_sha256")
    physical_ids = _canonical_physical_ids(physical_ids)
    if allocated_context not in RESIDENT_QSA_CACHE_CAPACITIES:
        raise ValueError(f"allocated_context must be one of {RESIDENT_QSA_CACHE_CAPACITIES}, got {allocated_context!r}")
    if (bf4_corpus_root is None) != (bf4_corpus_verification is None):
        raise ValueError("a BF4 corpus needs both its root and its verification record")

    runtime_path = Path(runtime_extension).resolve(strict=True)
    loaded_extension = Path(ttnn._ttnn.__file__).resolve(strict=True)
    if runtime_path != loaded_extension:
        raise LiveDecodeDiagnosticError(f"runtime extension {runtime_path} is not the loaded one {loaded_extension}")
    _mark(marker, "before-runtime-sha256")
    actual_runtime_sha = _stream_sha256(runtime_path)
    _mark(marker, "after-runtime-sha256")
    if actual_runtime_sha != runtime_sha256:
        raise LiveDecodeDiagnosticError(f"runtime digest differs: {actual_runtime_sha} != {runtime_sha256}")

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
    corpus = signatures = compatibility = None
    if bf4_corpus_root is not None:
        _mark(marker, "before-diagnostic-bf4-bind")
        corpus = bind_diagnostic_bf4_corpus(bf4_corpus_root, bf4_corpus_verification, producer=bf4_producer)
        signatures = _snapshot_corpus(corpus)
        _mark(marker, "after-diagnostic-bf4-bind")
        compatibility = Qwen38BF4ConsumerCompatibility(
            producer_source_head=corpus.identity.source_head,
            producer_runtime=MappingProxyType(dict(corpus.identity.runtime)),
            producer_checkpoint=MappingProxyType(dict(corpus.identity.checkpoint)),
            producer_physical_ids=corpus.identity.physical_ids,
            consumer_provenance_key=provenance.key,
            consumer_tt_metal_sha=tt_metal_sha,
            consumer_runtime_extension=runtime_path,
            consumer_runtime_sha256=runtime_sha256,
            consumer_physical_ids=mesh_contract.physical_ids,
            expert_residency="resident",
            qsa_cache_capacity=allocated_context,
        )
        compatibility.validate_corpus(corpus)
    return Qwen38PreparedLiveDecodeDiagnostic(
        checkpoint=checkpoint,
        placement=placement,
        mesh_contract=mesh_contract,
        provenance=provenance,
        cache_roots=cache_roots,
        runtime_extension=runtime_path,
        runtime_sha256=runtime_sha256,
        tt_metal_sha=tt_metal_sha,
        qsa_cache_capacity=allocated_context,
        corpus=corpus,
        artifact_signatures=signatures,
        bf4_consumer_compatibility=compatibility,
    )


class Qwen38DiagnosticBF4Cache:
    """Read-only per-process cache adapter for the already-bound corpus."""

    def __init__(
        self,
        *,
        production_cache: Qwen38BF4Cache,
        corpus: DiagnosticBF4Corpus,
        artifact_signatures: Mapping[Path, FileSignature],
        compatibility: Qwen38BF4ConsumerCompatibility,
        mesh_device,
    ) -> None:
        if type(compatibility) is not Qwen38BF4ConsumerCompatibility:
            raise TypeError("diagnostic BF4 cache requires the exact consumer compatibility record")
        compatibility.validate_corpus(corpus)
        production_cache.mesh_contract.validate_mesh(mesh_device)
        # The corpus keeps its producer's physical order; the shards are keyed by mesh coordinate, so the live
        # builder must be in the consumer platform's order with the corpus's mesh shape and ring size.
        if production_cache.identity.physical_ids != compatibility.consumer_physical_ids:
            raise LiveDecodeDiagnosticError(
                f"live builder physical order {production_cache.identity.physical_ids} differs from the "
                f"consumer platform's {compatibility.consumer_physical_ids} "
                f"(corpus producer order {corpus.identity.physical_ids})"
            )
        if production_cache.identity.mesh_shape != corpus.identity.mesh_shape:
            raise LiveDecodeDiagnosticError(
                f"live builder mesh shape {production_cache.identity.mesh_shape} differs from corpus "
                f"{corpus.identity.mesh_shape}"
            )
        if production_cache.identity.ring_size != corpus.identity.ring_size:
            raise LiveDecodeDiagnosticError(
                f"live builder ring size {production_cache.identity.ring_size} differs from corpus "
                f"{corpus.identity.ring_size}"
            )
        self.root = corpus.root
        self.identity = production_cache.identity
        self.mesh_contract = production_cache.mesh_contract
        self.mesh_device = mesh_device
        self.corpus = corpus
        self.diagnostic_corpus = corpus
        self.compatibility = compatibility
        self.production_manifest_used = False
        self.artifact_signatures = artifact_signatures

    @staticmethod
    def _artifact(artifact: DiagnosticBF4Artifact) -> BF4Artifact:
        return BF4Artifact(
            name=artifact.name,
            relative_path=artifact.relative_path,
            sha256=artifact.sha256,
            bytes=artifact.bytes,
            logical_shape=artifact.global_shape,
            dtype=artifact.dtype,
            layout=artifact.layout,
        )

    def verify_layer(self, namespace: str, layer_index: int) -> BF4LayerRecord | None:
        try:
            record = self.corpus.get(namespace, layer_index)
        except RuntimeError:
            return None
        return BF4LayerRecord(
            namespace=record.namespace,
            layer_index=record.layer_index,
            expert_ranges=record.expert_ranges,
            ring_size=record.ring_size,
            w0_w1=self._artifact(record.w0_w1),
            w2=self._artifact(record.w2),
        )

    def _load_artifact(self, artifact: DiagnosticBF4Artifact, expected_memory: Any):
        expected_signature = self.artifact_signatures.get(artifact.path)
        if expected_signature is None:
            raise LiveDecodeDiagnosticError(f"unbound diagnostic artifact requested: {artifact.path}")
        descriptor = None
        tensor = None
        try:
            descriptor = os.open(artifact.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            before = _signature(os.fstat(descriptor))
            if before != expected_signature or _signature(artifact.path.lstat()) != expected_signature:
                raise LiveDecodeDiagnosticError(f"diagnostic BF4 artifact changed before load: {artifact.path}")
            tensor = ttnn.load_tensor(Path(f"/proc/self/fd/{descriptor}"), device=self.mesh_device)
            if (
                _signature(os.fstat(descriptor)) != expected_signature
                or _signature(artifact.path.lstat()) != expected_signature
            ):
                raise LiveDecodeDiagnosticError(f"diagnostic BF4 artifact changed during load: {artifact.path}")
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.EXPERT_SHARDED, shard_dim=2)
            if tuple(int(value) for value in tensor.shape) != artifact.local_shape:
                raise LiveDecodeDiagnosticError(
                    f"diagnostic BF4 local shape differs for {artifact.path}: "
                    f"{tuple(tensor.shape)} != {artifact.local_shape}"
                )
            if tensor.dtype != ttnn.bfloat4_b or tensor.layout != ttnn.TILE_LAYOUT:
                raise LiveDecodeDiagnosticError(f"diagnostic BF4 dtype/layout differs for {artifact.path}")
            if tensor.memory_config() != expected_memory:
                raise LiveDecodeDiagnosticError(f"diagnostic BF4 memory config differs for {artifact.path}")
            locals_in_order = tuple(ttnn.get_device_tensors(tensor))
            coordinates = tuple(
                tuple(int(value) for value in coordinate) for coordinate in tensor.tensor_topology().mesh_coords()
            )
            physical_ids = tuple(
                int(self.mesh_device.get_device_id(ttnn.MeshCoordinate(*coordinate))) for coordinate in coordinates
            )
            # Shards land by mesh coordinate (the corpus's); the device at each coordinate is the
            # consumer platform's (the producer's order).
            expected_physical_ids = self.compatibility.consumer_physical_ids
            if coordinates != self.corpus.identity.mesh_coordinates or physical_ids != expected_physical_ids:
                raise LiveDecodeDiagnosticError(
                    f"diagnostic BF4 coordinate ownership differs for {artifact.path}: coordinates "
                    f"actual={coordinates} expected={self.corpus.identity.mesh_coordinates}; physical ids "
                    f"actual={physical_ids} expected={expected_physical_ids} "
                    f"(corpus producer {self.corpus.identity.physical_ids})"
                )
            if len(locals_in_order) != 4 or any(
                tuple(int(value) for value in local.shape) != artifact.local_shape
                or local.dtype != ttnn.bfloat4_b
                or local.layout != ttnn.TILE_LAYOUT
                for local in locals_in_order
            ):
                raise LiveDecodeDiagnosticError(f"diagnostic BF4 local tensor metadata differs for {artifact.path}")
            return tensor
        except BaseException:
            if tensor is not None:
                ttnn.deallocate(tensor)
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def load_layer(self, mesh_device, *, layer_index: int, namespace: str = "backbone") -> tuple[Any, Any]:
        if mesh_device is not self.mesh_device:
            raise LiveDecodeDiagnosticError("diagnostic BF4 load requested on a different mesh object")
        self.mesh_contract.validate_mesh(mesh_device)
        if qualify_live_bf4_ring(mesh_device) != self.identity.dram_bank_ring_order:
            raise LiveDecodeDiagnosticError("live DRAM bank ring order changed before diagnostic BF4 load")
        record = self.corpus.get(namespace, layer_index)
        memory_configs = ttnn.experimental.get_weight_mem_configs(
            mesh_device,
            num_layers=1,
            experts_per_device=128,
            hidden_size=2560,
            intermediate_size=640,
            has_bias=False,
        )
        w0_w1 = self._load_artifact(record.w0_w1, memory_configs.w0_w1)
        try:
            w2 = self._load_artifact(record.w2, memory_configs.w2)
        except BaseException:
            ttnn.deallocate(w0_w1)
            raise
        return w0_w1, w2


@dataclass(frozen=True)
class Qwen38LiveDecodeConstruction:
    """A real live builder stopped before target/state/inference construction."""

    prepared: Qwen38PreparedLiveDecodeDiagnostic
    builder: Qwen38TTNNBuilder
    production_cache: Qwen38BF4Cache
    diagnostic_cache: Qwen38DiagnosticBF4Cache | None
    expert_streamer: Any
    staged_bf4_layers: tuple[tuple[str, int], ...] = ()
    bf4_admission: dict[str, Any] | None = None  # the one-expert byte check of the production cache, if it held a layer

    def summary(self) -> dict[str, Any]:
        return {
            "mode": MODE,
            "result": "live_builder_with_49_bf4_records_constructed",
            "production_qualification": False,
            "performance_qualification": False,
            "bf4_source": "production_cache" if self.diagnostic_cache is None else "cpu_staged_corpus",
            "builder_identity_key": self.builder.identity.key,
            "expert_residency": self.builder.identity.expert_residency,
            "qsa_cache_capacity": self.builder.identity.qsa_cache_capacity,
            "bf4_identity_key": self.production_cache.identity.key,
            "bound_records": EXPECTED_RECORD_COUNT,
            "bound_tensorbins": EXPECTED_TENSORBIN_COUNT,
            "staged_bf4_layers": [list(slot) for slot in self.staged_bf4_layers],
            "target_built": False,
            "session_constructed": False,
            "token_decoded": False,
            "next_live_boundary": "Qwen38TTNNBuilder.build_target",
        }


def missing_bf4_layers(builder: Qwen38TTNNBuilder) -> tuple[tuple[str, int], ...]:
    """The slots the builder's BF4 cache does not hold yet (each ``verify_layer`` hashes a present record)."""

    return tuple(slot for slot in EXPECTED_SLOTS if builder.bf4_cache.verify_layer(*slot) is None)


def construct_live_decode_diagnostic(
    prepared: Qwen38PreparedLiveDecodeDiagnostic,
    *,
    mesh_device,
    collective_topology,
    marker: Marker | None = None,
    stage_missing_bf4: bool = True,
    bf4_stage_limit: int | None = None,
    require_complete: bool = True,
    decode_dram_workers_per_bank: int | None = None,
) -> Qwen38LiveDecodeConstruction:
    """Construct and attach the real live builder, then stop before target build.

    With a corpus the read-only adapter replaces the builder's cache.  Without one the production cache is used:
    missing layers are converted from the checkpoint (``stage_missing_bf4``; at most ``bf4_stage_limit`` of them
    this run, for hosts that bound a job's wall time) and the construction refuses while any is still missing
    (``require_complete``; a prepare-only run returns the builder as is).
    """

    if type(prepared) is not Qwen38PreparedLiveDecodeDiagnostic:
        raise TypeError("prepared must be Qwen38PreparedLiveDecodeDiagnostic")
    if os.environ.get("QWEN38_HARDWARE_MODE") != MODE:
        raise LiveDecodeDiagnosticError(f"QWEN38_HARDWARE_MODE must be {MODE}")
    prepared.mesh_contract.validate_mesh(mesh_device)
    loaded_extension = Path(ttnn._ttnn.__file__).resolve(strict=True)
    if loaded_extension != prepared.runtime_extension or _stream_sha256(loaded_extension) != prepared.runtime_sha256:
        raise LiveDecodeDiagnosticError("loaded TTNN extension differs from CPU-prepared runtime identity")

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
        decode_dram_workers_per_bank=decode_dram_workers_per_bank,
    )
    _mark(marker, "after-live-builder-construction")
    production_cache = builder.bf4_cache
    diagnostic_cache = None
    staged: list[tuple[str, int]] = []
    admission = None
    if prepared.corpus is not None:
        prepared.bf4_consumer_compatibility.validate_live(builder=builder, production_cache=production_cache)
        diagnostic_cache = Qwen38DiagnosticBF4Cache(
            production_cache=production_cache,
            corpus=prepared.corpus,
            artifact_signatures=prepared.artifact_signatures,
            compatibility=prepared.bf4_consumer_compatibility,
            mesh_device=mesh_device,
        )
        builder.bf4_cache = diagnostic_cache
        builder.expert_streamer = Qwen38BF4ResidentSet(diagnostic_cache, mesh_device)
    else:
        _mark(marker, "before-bf4-cache-inventory")
        missing = missing_bf4_layers(builder)
        _mark(marker, f"after-bf4-cache-inventory-missing-{len(missing)}")
        # The cache pins the converter's sources, not the runtime: prove the cached bytes are this converter's output
        # on one expert of the first cached layer before anything is loaded or converted next to them.
        cached_slot = next((slot for slot in EXPECTED_SLOTS if slot not in missing), None)
        if cached_slot is not None:
            _mark(marker, f"before-bf4-cache-admission-{cached_slot[0]}-{cached_slot[1]:02d}")
            admission = production_cache.admit_converted_bytes(
                prepared.checkpoint, prepared.placement, namespace=cached_slot[0], layer_index=cached_slot[1]
            )
            _mark(marker, f"after-bf4-cache-admission-expert-{admission['expert']}-{admission['seconds']}s")
        if missing and stage_missing_bf4:
            for namespace, layer_index in missing[: len(missing) if bf4_stage_limit is None else bf4_stage_limit]:
                _mark(marker, f"before-bf4-stage-{namespace}-{layer_index:02d}")
                builder.stage_bf4_layer(namespace=namespace, layer_index=layer_index)
                ttnn.synchronize_device(mesh_device)
                staged.append((namespace, layer_index))
                _mark(marker, f"after-bf4-stage-{namespace}-{layer_index:02d}")
            missing = missing_bf4_layers(builder)
        if missing and require_complete:
            raise LiveDecodeDiagnosticError(
                f"the BF4 cache under {production_cache.root} lacks {len(missing)} of {EXPECTED_RECORD_COUNT} layers "
                f"({', '.join(f'{namespace}:{index}' for namespace, index in missing[:8])}{'...' if len(missing) > 8 else ''}); "
                f"{len(staged)} staged this run"
            )

    if require_complete:
        _mark(marker, "before-diagnostic-record-attachment")
        backbone = builder.require_complete_backbone_bf4()
        mtp = builder.require_bf4_layers(namespace="mtp", layer_indices=(0,))
        if len(backbone) != BACKBONE_LAYERS or len(mtp) != 1:
            raise LiveDecodeDiagnosticError(
                f"real builder attached {len(backbone)} backbone and {len(mtp)} MTP records, expected 48/1"
            )
        _mark(marker, "after-diagnostic-record-attachment")
    return Qwen38LiveDecodeConstruction(
        prepared=prepared,
        builder=builder,
        production_cache=production_cache,
        diagnostic_cache=diagnostic_cache,
        expert_streamer=builder.expert_streamer,
        staged_bf4_layers=tuple(staged),
        bf4_admission=admission,
    )


@dataclass(frozen=True)
class Qwen38LiveOrdinarySession:
    """Persistent owner of one real built target and ordinary session."""

    construction: Qwen38LiveDecodeConstruction
    built_target: Any
    builder: Qwen38TTNNBuilder
    session: Qwen38OrdinaryDecodeSession

    def __post_init__(self) -> None:
        if self.builder is not self.construction.builder:
            raise ValueError("live ordinary owner must retain the construction's exact builder")
        if type(self.session) is not Qwen38OrdinaryDecodeSession:
            raise TypeError("live ordinary owner must retain the exact ordinary decode session")

    def summary(self) -> dict[str, Any]:
        return {
            "mode": MODE,
            "result": "live_ordinary_session_constructed",
            "production_qualification": False,
            "performance_qualification": False,
            "builder_identity_key": self.builder.identity.key,
            "target_built": True,
            "session_constructed": True,
            "session_status": self.session.status.value,
            "token_decoded": False,
        }


@dataclass(frozen=True)
class Qwen38LiveOneTokenDecode:
    """One real ordinary token emitted from the live diagnostic construction."""

    construction: Qwen38LiveDecodeConstruction
    decoded: Qwen38NonqualifyingDecodedToken
    oracle: Qwen38RetainedCPUTokenOracle
    stop_reason: str
    timing_ns: Mapping[str, int | str]
    completed_layers: tuple[int, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "mode": MODE,
            "result": "ordinary_one_token_decoded",
            "production_qualification": False,
            "performance_qualification": False,
            "qualification": self.decoded.qualification,
            "production_manifest_read": False,
            "production_manifest_created": False,
            "bound_records": EXPECTED_RECORD_COUNT,
            "bound_tensorbins": EXPECTED_TENSORBIN_COUNT,
            "builder_identity_key": self.construction.builder.identity.key,
            "expert_residency": self.construction.builder.identity.expert_residency,
            "qsa_cache_capacity": self.construction.builder.identity.qsa_cache_capacity,
            "input_token_id": self.decoded.input_token_id,
            "token_id": self.decoded.token_id,
            "expected_token_id": self.oracle.output_token_ids[0],
            "matches_retained_cpu_oracle": self.decoded.token_id == self.oracle.output_token_ids[0],
            "retained_cpu_oracle_sha256": self.oracle.sha256,
            "cache_position": self.decoded.cache_position,
            "stop_reason": self.stop_reason,
            "timing_ns": dict(self.timing_ns),
            "layers_executed": self.decoded.layers_executed,
            "completed_layers": list(self.completed_layers),
            "terminal_mechanic": self.decoded.terminal_mechanic,
            "sampling_mechanic": self.decoded.sampling_mechanic,
            "target_built": True,
            "session_constructed": True,
            "session_closed": True,
            "resident_owner_closed": True,
            "token_decoded": True,
        }


def decode_live_one_token(
    construction: Qwen38LiveDecodeConstruction,
    *,
    oracle: Qwen38RetainedCPUTokenOracle,
    input_token_id: int = DIAGNOSTIC_INPUT_TOKEN_ID,
    eos_token_ids: tuple[int, ...] = DIAGNOSTIC_EOS_TOKEN_IDS,
    marker: Marker | None = None,
) -> Qwen38LiveOneTokenDecode:
    """Construct, use, and close one persistent owner for one greedy token."""

    if type(construction) is not Qwen38LiveDecodeConstruction:
        raise TypeError("construction must be Qwen38LiveDecodeConstruction")
    if (
        type(oracle) is not Qwen38RetainedCPUTokenOracle
        or oracle.input_token_ids != DIAGNOSTIC_ORACLE_INPUT_TOKEN_IDS
        or oracle.output_token_ids != DIAGNOSTIC_ORACLE_OUTPUT_TOKEN_IDS
    ):
        raise LiveDecodeDiagnosticError("one-token diagnostic requires the retained two-step CPU oracle")
    if input_token_id != DIAGNOSTIC_INPUT_TOKEN_ID:
        raise ValueError(f"diagnostic input token must remain pinned to {DIAGNOSTIC_INPUT_TOKEN_ID}")
    if tuple(eos_token_ids) != DIAGNOSTIC_EOS_TOKEN_IDS:
        raise ValueError(f"diagnostic EOS tokens must remain pinned to {DIAGNOSTIC_EOS_TOKEN_IDS}")
    owner = None
    completed_layers: list[int] = []

    def observe_layer(layer_index: int, residual: Any, state: Any, aux: Any) -> None:
        del residual, state, aux
        expected = len(completed_layers)
        if layer_index != expected:
            raise LiveDecodeDiagnosticError(
                f"ordinary layer observer advanced out of order: {layer_index} != {expected}"
            )
        _mark(marker, f"before-layer-{layer_index}-synchronize")
        ttnn.synchronize_device(construction.builder.mesh_device)
        completed_layers.append(layer_index)
        _mark(marker, f"after-layer-{layer_index}-device-complete")

    try:
        owner = construct_live_ordinary_session(
            construction,
            eos_token_ids=eos_token_ids,
            layer_observer=observe_layer,
            marker=marker,
        )
        session = owner.session
        _mark(marker, "before-token-decode")
        event = session.begin((input_token_id,), max_new_tokens=1)
        _mark(marker, "after-token-decode")
        if type(event) is not Qwen38OrdinaryToken:
            raise TypeError("ordinary decode returned a non-Qwen38 token record")
        if event.token_index != 0 or event.cache_position != 1:
            raise RuntimeError(
                f"first ordinary emission has index/position {(event.token_index, event.cache_position)}, "
                "expected (0,1)"
            )
        if event.stop_reason not in (Qwen38OrdinaryStopReason.EOS, Qwen38OrdinaryStopReason.MAX_NEW_TOKENS):
            raise RuntimeError(f"one-token diagnostic did not stop after its first emission: {event.stop_reason!r}")
        if session.status is not Qwen38OrdinarySessionStatus.FINISHED:
            raise RuntimeError(f"one-token diagnostic ended with session status {session.status.value!r}")
        if tuple(completed_layers) != tuple(range(48)):
            raise RuntimeError(f"one-token diagnostic completed layers {tuple(completed_layers)}, expected 0:47")
        _require_retained_oracle_first_step(
            oracle,
            input_token_id=input_token_id,
            output_token_id=event.token_id,
        )
        decoded = Qwen38NonqualifyingDecodedToken(
            input_token_id=input_token_id,
            token_id=event.token_id,
            cache_position=event.cache_position,
        )
        stop_reason = event.stop_reason.value
        timing_ns = MappingProxyType(
            {
                "phase": event.timing.phase.value,
                "input_token_count": event.timing.input_token_count,
                "model_call": event.timing.model_call_ns,
                "explicit_sync": event.timing.explicit_sync_ns,
                "host_overhead": event.timing.host_overhead_ns,
                "end_to_end": event.timing.end_to_end_ns,
            }
        )
    finally:
        if owner is not None and owner.session.status is not Qwen38OrdinarySessionStatus.POISONED:
            _mark(marker, "before-session-close")
            close_live_ordinary_owner(owner, marker=marker)
            if owner.session.status is not Qwen38OrdinarySessionStatus.CLOSED:
                raise RuntimeError(f"ordinary session close ended with status {owner.session.status.value!r}")
            _mark(marker, "after-session-close")
    return Qwen38LiveOneTokenDecode(
        construction=construction,
        decoded=decoded,
        oracle=oracle,
        stop_reason=stop_reason,
        timing_ns=timing_ns,
        completed_layers=tuple(completed_layers),
    )


def construct_live_ordinary_session(
    construction: Qwen38LiveDecodeConstruction,
    *,
    eos_token_ids: tuple[int, ...] = DIAGNOSTIC_EOS_TOKEN_IDS,
    stop_on_eos: bool = True,
    layer_observer: LayerObserver | None = None,
    phase_observer: Marker | None = None,
    marker: Marker | None = None,
) -> Qwen38LiveOrdinarySession:
    """Build and retain the exact builder, target, and ordinary session."""

    if type(construction) is not Qwen38LiveDecodeConstruction:
        raise TypeError("construction must be Qwen38LiveDecodeConstruction")
    if tuple(eos_token_ids) != DIAGNOSTIC_EOS_TOKEN_IDS:
        raise ValueError(f"diagnostic EOS tokens must remain pinned to {DIAGNOSTIC_EOS_TOKEN_IDS}")
    if type(stop_on_eos) is not bool:
        raise TypeError("stop_on_eos must be an explicit bool")
    expected_cache = (
        construction.production_cache if construction.diagnostic_cache is None else construction.diagnostic_cache
    )
    if construction.builder.bf4_cache is not expected_cache:
        raise LiveDecodeDiagnosticError("builder no longer owns the bound BF4 cache")
    if construction.builder.expert_streamer is not construction.expert_streamer:
        raise LiveDecodeDiagnosticError("builder no longer owns the bound diagnostic BF4 streamer")
    if (
        construction.builder.identity.expert_residency != "resident"
        or construction.builder.identity.qsa_cache_capacity not in RESIDENT_QSA_CACHE_CAPACITIES
        or not isinstance(construction.expert_streamer, Qwen38BF4ResidentSet)
    ):
        raise LiveDecodeDiagnosticError(
            "full ordinary inference requires a resident diagnostic profile (C in "
            f"{RESIDENT_QSA_CACHE_CAPACITIES}), got "
            f"{construction.builder.identity.expert_residency}/C{construction.builder.identity.qsa_cache_capacity}"
        )

    session = None
    built_target = None
    try:
        _mark(marker, "before-target-build")
        built_target = construction.builder.build_target()
        _mark(marker, "after-target-build")
        _mark(marker, "before-target-build-synchronize")
        ttnn.synchronize_device(construction.builder.mesh_device)
        _mark(marker, "after-target-build-synchronize")
        _mark(marker, "before-session-state-allocation")
        if phase_observer is None:
            session = Qwen38OrdinaryDecodeSession(
                built_target,
                expected_provenance=construction.prepared.provenance,
                expected_physical_ids=construction.prepared.mesh_contract.physical_ids,
                expected_identity_key=construction.builder.identity.key,
                eos_token_ids=eos_token_ids,
                stop_on_eos=stop_on_eos,
                layer_observer=layer_observer,
            )
        else:
            session = Qwen38OrdinaryDecodeSession(
                built_target,
                expected_provenance=construction.prepared.provenance,
                expected_physical_ids=construction.prepared.mesh_contract.physical_ids,
                expected_identity_key=construction.builder.identity.key,
                eos_token_ids=eos_token_ids,
                stop_on_eos=stop_on_eos,
                layer_observer=layer_observer,
                phase_observer=phase_observer,
            )
        if session.allocated_context != construction.builder.identity.qsa_cache_capacity:
            raise LiveDecodeDiagnosticError(
                f"ordinary session allocated a C{session.allocated_context} cache, the builder identity admits "
                f"C{construction.builder.identity.qsa_cache_capacity}"
            )
        _mark(marker, "after-session-state-allocation")
        shared_tt_ccl = construction.builder.tt_ccl
        if shared_tt_ccl._manager is not None:
            raise LiveDecodeDiagnosticError("builder TT-CCL manager resolved before diagnostic session preflight")
        _mark(marker, "before-tt-ccl-resolution")
        resolved_tt_ccl = shared_tt_ccl._resolve()
        _mark(marker, "after-tt-ccl-resolution")
        _mark(marker, "before-tt-ccl-resolution-synchronize")
        ttnn.synchronize_device(construction.builder.mesh_device)
        _mark(marker, "after-tt-ccl-resolution-synchronize")
        if shared_tt_ccl._manager is not resolved_tt_ccl:
            raise LiveDecodeDiagnosticError("builder TT-CCL proxy lost its resolved manager")
        return Qwen38LiveOrdinarySession(
            construction=construction,
            built_target=built_target,
            builder=construction.builder,
            session=session,
        )
    except BaseException:
        if session is not None and session.status is not Qwen38OrdinarySessionStatus.POISONED:
            _mark(marker, "before-session-close")
            session.close()
            _mark(marker, "after-session-close")
        if (
            built_target is not None
            and session is not None
            and session.status is Qwen38OrdinarySessionStatus.CLOSED
            and not construction.expert_streamer.closed
        ):
            built_target.components.close_resident_experts()
            ttnn.synchronize_device(construction.builder.mesh_device)
        raise


def close_live_ordinary_owner(
    owner: Qwen38LiveOrdinarySession,
    *,
    marker: Marker | None = None,
) -> None:
    """Close one ordinary state owner and its resident graph exactly once."""

    if type(owner) is not Qwen38LiveOrdinarySession:
        raise TypeError("owner must be the exact Qwen38LiveOrdinarySession")
    session = owner.session
    resident_owner = owner.built_target.components.expert_streamer
    if not isinstance(resident_owner, Qwen38BF4ResidentSet):
        raise LiveDecodeDiagnosticError("full ordinary owner lacks its resident BF4 graph")
    if resident_owner.closed:
        raise LiveDecodeDiagnosticError("resident ordinary graph was already closed")
    if session.status is Qwen38OrdinarySessionStatus.POISONED:
        raise LiveDecodeDiagnosticError("cannot release resident weights after an uncertain ordinary transaction")
    if session.status is not Qwen38OrdinarySessionStatus.CLOSED:
        session.close()
    _mark(marker, "before-resident-expert-close")
    owner.built_target.components.close_resident_experts()
    ttnn.synchronize_device(owner.builder.mesh_device)
    _mark(marker, "after-resident-expert-close")
    if (
        resident_owner.closed is not True
        or resident_owner.poisoned is not False
        or resident_owner.live_tensor_handle_count != 0
        or resident_owner.resident_layers
    ):
        raise LiveDecodeDiagnosticError("resident ordinary graph did not close cleanly")


__all__ = [
    "DEFAULT_CPU_ORACLE",
    "DIAGNOSTIC_EOS_TOKEN_IDS",
    "DIAGNOSTIC_EXPECTED_TOKEN_ID",
    "DIAGNOSTIC_INPUT_TOKEN_ID",
    "DIAGNOSTIC_ORACLE_INPUT_TOKEN_IDS",
    "DIAGNOSTIC_ORACLE_OUTPUT_TOKEN_IDS",
    "DIAGNOSTIC_PROMPT_TEXT",
    "LiveDecodeDiagnosticError",
    "Qwen38BF4ConsumerCompatibility",
    "Qwen38DiagnosticBF4Cache",
    "Qwen38LiveDecodeConstruction",
    "Qwen38LiveOrdinarySession",
    "Qwen38LiveOneTokenDecode",
    "Qwen38PreparedLiveDecodeDiagnostic",
    "Qwen38RetainedCPUTokenOracle",
    "construct_live_decode_diagnostic",
    "construct_live_ordinary_session",
    "close_live_ordinary_owner",
    "decode_live_one_token",
    "load_retained_cpu_token_oracle",
    "missing_bf4_layers",
    "prepare_live_decode_diagnostic",
]
