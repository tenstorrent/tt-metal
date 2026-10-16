"""The CPU-staged BF4 corpus binder on a synthetic 49-slot corpus: the producer identity is read from the verification
record (and pinned when the caller passes one), the physical order is adopted from the evidence, every tensorbin is
hashed, and a missing file, a digest, a shape or a slot that differs refuses."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from models.demos.blackhole.qwen38_flash_next import diagnostic_bf4

PHYSICAL_IDS = (1, 0, 2, 3)
PRODUCER = {
    "source_head": "a" * 40,
    "runtime": {"extension": "/runtime/site-packages/ttnn/extension.so", "sha256": "d" * 64},
    "checkpoint": {"config_sha256": "c" * 64, "revision": "r"},
}


def _write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    artifact_root = tmp_path / "namespace-key" / "identity-key"
    artifact_root.mkdir(parents=True)
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    verification_result = tmp_path / "verification.json"
    specs = MappingProxyType(
        {
            "w0_w1": diagnostic_bf4._ArtifactSpec(
                "w0_w1_dtype_BFLOAT4_B_layout_TILE.tensorbin", 11, (8, 1, 128, 2), (8, 1, 512, 2)
            ),
            "w2": diagnostic_bf4._ArtifactSpec(
                "w2_dtype_BFLOAT4_B_layout_TILE.tensorbin", 13, (8, 1, 128, 3), (8, 1, 512, 3)
            ),
        }
    )
    contract = diagnostic_bf4._BindingContract(
        artifact_root=artifact_root,
        verification_result=verification_result,
        slots=diagnostic_bf4.EXPECTED_SLOTS,
        artifact_specs=specs,
        checkpoint=MappingProxyType(PRODUCER["checkpoint"]),
        runtime=MappingProxyType(PRODUCER["runtime"]),
        source_head=PRODUCER["source_head"],
        mesh_shape=(1, 4),
        physical_ids=None,
        mesh_coordinates=((0, 0), (0, 1), (0, 2), (0, 3)),
        expert_ranges=((0, 128), (128, 256), (256, 384), (384, 512)),
        ring_size=8,
        total_bytes=49 * 24,
    )
    results = []
    for ordinal, slot in enumerate(contract.slots):
        namespace, layer_index = slot
        slot_dir = artifact_root / namespace / f"layer-{layer_index:02d}"
        slot_dir.mkdir(parents=True)
        artifacts = {}
        for artifact_index, (name, spec) in enumerate(specs.items()):
            path = slot_dir / spec.filename
            path.write_bytes(bytes([(ordinal + artifact_index) % 251]) * spec.bytes)
            artifacts[name] = {
                "bytes": spec.bytes,
                "dtype": "BFLOAT4_B",
                "global_shape": list(spec.global_shape),
                "layout": "TILE",
                "local_shape": list(spec.local_shape),
                "path": str(path),
                "sha256": _digest(path),
            }
        evidence_path = evidence_root / f"{namespace}-layer-{layer_index:02d}.json"
        _write_json(
            evidence_path,
            {
                "mode": diagnostic_bf4.STAGING_MODE,
                "production_qualification": False,
                "device_opened": False,
                "device_locks_acquired": False,
                **PRODUCER,
                "slot": list(slot),
                "ownership": {
                    "expert_ranges": [list(pair) for pair in contract.expert_ranges],
                    "mesh_coordinates": [list(pair) for pair in contract.mesh_coordinates],
                    "mesh_shape": list(contract.mesh_shape),
                    "physical_ids": list(PHYSICAL_IDS),
                    "ring_size": contract.ring_size,
                },
                "device_shards": [
                    {
                        "device_index": index,
                        "mesh_coordinate": list(contract.mesh_coordinates[index]),
                        "physical_id": PHYSICAL_IDS[index],
                        "expert_range": list(contract.expert_ranges[index]),
                    }
                    for index in range(4)
                ],
                "artifacts": artifacts,
            },
        )
        results.append({"status": "staged", "slot": list(slot), "artifacts": artifacts, "evidence": str(evidence_path)})
    summary_paths = (evidence_root / "layer0-summary.json", evidence_root / "remaining-summary.json")
    _write_summaries(results, summary_paths, verification_result, artifact_root, contract.total_bytes, create=True)
    return contract, results, summary_paths


def _write_summaries(
    results, summary_paths, verification_result, artifact_root=None, total_bytes=None, *, create=False
):
    for path, summary_results in zip(summary_paths, (results[:1], results[1:])):
        _write_json(
            path,
            {
                "mode": diagnostic_bf4.STAGING_MODE,
                "status": "pass",
                "production_qualification": False,
                "device_opened": False,
                "device_locks_acquired": False,
                **PRODUCER,
                "results": summary_results,
            },
        )
    if create:
        verification = {
            "mode": diagnostic_bf4.VERIFICATION_MODE,
            "status": "pass",
            "production_qualification": False,
            "device_opened": False,
            "device_locks_acquired": False,
            "artifact_root": str(artifact_root),
            "corpus": {
                "slots": 49,
                "backbone_layers": list(range(48)),
                "mtp_layers": [0],
                "tensorbins": 98,
                "bytes": total_bytes,
            },
            "staging_identity": PRODUCER,
        }
    else:
        verification = json.loads(verification_result.read_text())
    verification["source_summaries"] = [{"path": str(path), "sha256": _digest(path)} for path in summary_paths]
    _write_json(verification_result, verification)


def test_binds_exact_49_slot_corpus_and_adopts_the_producer_order(tmp_path):
    contract, _, _ = _fixture(tmp_path)
    corpus = diagnostic_bf4._bind(contract.artifact_root, contract.verification_result, contract=contract)
    assert len(corpus.records) == 49
    assert corpus.get("backbone", 47).w0_w1.local_shape == (8, 1, 128, 2)
    assert corpus.get("mtp", 0).physical_ids == PHYSICAL_IDS
    assert corpus.identity.physical_ids == PHYSICAL_IDS
    assert corpus.identity.source_head == PRODUCER["source_head"]
    assert dict(corpus.identity.runtime) == PRODUCER["runtime"]
    assert corpus.summary()["tensorbins"] == 98
    assert corpus.summary()["all_payload_sha256_verified"] is True


def test_public_binder_reads_the_producer_identity_from_the_record_and_pins_a_given_one(expect_error, tmp_path):
    contract, _, _ = _fixture(tmp_path)
    assert diagnostic_bf4.producer_identity(contract.verification_result) == PRODUCER
    built = diagnostic_bf4.binding_contract(contract.artifact_root, contract.verification_result, producer=PRODUCER)
    assert built.source_head == PRODUCER["source_head"] and built.physical_ids is None
    assert built.total_bytes == diagnostic_bf4.EXPECTED_TOTAL_BYTES  # the public contract is the real corpus's
    other = {**PRODUCER, "source_head": "b" * 40}
    with expect_error(diagnostic_bf4.DiagnosticBF4BindingError, match="producer identity differs"):
        diagnostic_bf4.binding_contract(contract.artifact_root, contract.verification_result, producer=other)
    # the synthetic corpus is not the real one: its record matches its disk, so the public binder refuses it at the
    # first artifact whose size is not the compact pin
    with expect_error(diagnostic_bf4.DiagnosticBF4BindingError, match="not the compact layout"):
        diagnostic_bf4.bind_diagnostic_bf4_corpus(contract.artifact_root, contract.verification_result)
    with expect_error(diagnostic_bf4.DiagnosticBF4BindingError, match="absolute"):
        diagnostic_bf4.bind_diagnostic_bf4_corpus("relative/corpus", contract.verification_result)


def test_rejects_missing_artifact(expect_error, tmp_path):
    contract, _, _ = _fixture(tmp_path)
    (contract.artifact_root / "backbone/layer-12/w2_dtype_BFLOAT4_B_layout_TILE.tensorbin").unlink()
    with expect_error(diagnostic_bf4.DiagnosticBF4BindingError, match="unavailable"):
        diagnostic_bf4._bind(contract.artifact_root, contract.verification_result, contract=contract)


def test_rejects_artifact_hash_mismatch(expect_error, tmp_path):
    contract, results, summary_paths = _fixture(tmp_path)
    evidence_path = Path(results[4]["evidence"])
    evidence = json.loads(evidence_path.read_text())
    evidence["artifacts"]["w0_w1"]["sha256"] = "f" * 64
    results[4]["artifacts"]["w0_w1"]["sha256"] = "f" * 64
    _write_json(evidence_path, evidence)
    _write_summaries(results, summary_paths, contract.verification_result)
    with expect_error(diagnostic_bf4.DiagnosticBF4BindingError, match="SHA-256 differs"):
        diagnostic_bf4._bind(contract.artifact_root, contract.verification_result, contract=contract)


def test_rejects_artifact_shape_mismatch(expect_error, tmp_path):
    contract, results, summary_paths = _fixture(tmp_path)
    evidence_path = Path(results[9]["evidence"])
    evidence = json.loads(evidence_path.read_text())
    evidence["artifacts"]["w2"]["local_shape"] = [8, 1, 127, 3]
    results[9]["artifacts"]["w2"]["local_shape"] = [8, 1, 127, 3]
    _write_json(evidence_path, evidence)
    _write_summaries(results, summary_paths, contract.verification_result)
    with expect_error(diagnostic_bf4.DiagnosticBF4BindingError, match="local_shape differs"):
        diagnostic_bf4._bind(contract.artifact_root, contract.verification_result, contract=contract)


def test_rejects_slot_mismatch_and_a_second_physical_order(expect_error, tmp_path):
    contract, results, summary_paths = _fixture(tmp_path)
    results[-1]["slot"] = ["mtp", 1]
    _write_summaries(results, summary_paths, contract.verification_result)
    with expect_error(diagnostic_bf4.DiagnosticBF4BindingError, match="slot coverage differs"):
        diagnostic_bf4._bind(contract.artifact_root, contract.verification_result, contract=contract)
    contract, results, summary_paths = _fixture(tmp_path / "second")
    evidence_path = Path(results[3]["evidence"])
    evidence = json.loads(evidence_path.read_text())
    evidence["ownership"]["physical_ids"] = [0, 2, 1, 3]
    _write_json(evidence_path, evidence)
    with expect_error(diagnostic_bf4.DiagnosticBF4BindingError, match="physical_ids"):
        diagnostic_bf4._bind(contract.artifact_root, contract.verification_result, contract=contract)


def test_summary_reports_disk_bytes_and_the_contract_total_apart(tmp_path):
    contract, _, _ = _fixture(tmp_path)
    corpus = diagnostic_bf4._bind(contract.artifact_root, contract.verification_result, contract=contract)
    assert corpus.summary()["bytes"] == 49 * 24 == corpus.disk_bytes
    assert corpus.summary()["contract_bytes"] == contract.total_bytes == 49 * 24


def test_record_is_compared_to_the_disk_and_a_pre_compact_corpus_is_named(expect_error, tmp_path):
    """2026-09-26: the retained Aug-27 corpus (its record's bytes = its on-disk sum, 106,819,608,000) against the compact
    pins (ddd4e99e2f, 49 x 1,415,580,096) read "verification corpus differs: {... 'bytes': 106819608000} != {... 'bytes':
    69363424704}" -- the record was compared to the contract total instead of the disk. The record now compares to the
    disk and the first artifact names the layout; a record that disagrees with the disk still differs, before any
    artifact is read."""
    contract, results, summary_paths = _fixture(tmp_path)
    compact = dataclasses.replace(
        contract,
        artifact_specs=MappingProxyType(
            {name: dataclasses.replace(spec, bytes=spec.bytes - 2) for name, spec in contract.artifact_specs.items()}
        ),
        total_bytes=49 * 20,
    )
    with pytest.raises(  # allow-pytest.raises: inspect the captured exception object
        diagnostic_bf4.DiagnosticBF4BindingError, match="not the compact layout"
    ) as caught:  # allow-pytest.raises: inspect the captured exception object
        diagnostic_bf4._bind(compact.artifact_root, compact.verification_result, contract=compact)
    assert "w0_w1 is 11 bytes on disk against the contract's 9" in str(caught.value)
    _write_summaries(results, summary_paths, contract.verification_result, contract.artifact_root, 49 * 20, create=True)
    with expect_error(diagnostic_bf4.DiagnosticBF4BindingError, match="verification corpus differs"):
        diagnostic_bf4._bind(contract.artifact_root, contract.verification_result, contract=contract)
