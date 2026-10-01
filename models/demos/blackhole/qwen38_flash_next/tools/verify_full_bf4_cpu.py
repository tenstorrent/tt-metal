# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Boundedly verify the already-staged Qwen3.8 BF4 corpus without devices.

This diagnostic, non-promoting consumer checks every recorded artifact path,
regular-file identity, byte count, shape, dtype, and layer evidence record.  It
hashes four deterministic boundary samples rather than rereading the complete
69 GB corpus (49 slots in moe_compute's compact expert layout).  It never imports Torch or TTNN and never writes beneath the
artifact root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGING_MODE = "diagnostic_non_promoting_cpu_bf4_staging"
VERIFY_MODE = "diagnostic_non_promoting_cpu_bf4_corpus_verification"
EXPECTED_SLOTS = tuple(("backbone", index) for index in range(48)) + (("mtp", 0),)
ARTIFACT_SPECS = {
    "w0_w1": {
        "filename": "w0_w1_dtype_BFLOAT4_B_layout_TILE.tensorbin",
        "bytes": 943_719_648,
        "local_shape": [8, 1, 128, 10, 320, 128],
        "global_shape": [8, 1, 512, 10, 320, 128],
    },
    "w2": {
        "filename": "w2_dtype_BFLOAT4_B_layout_TILE.tensorbin",
        "bytes": 471_860_448,
        "local_shape": [8, 1, 128, 5, 320, 128],
        "global_shape": [8, 1, 512, 5, 320, 128],
    },
}
HASH_SAMPLE = (
    ("backbone", 0, "w0_w1"),
    ("backbone", 23, "w2"),
    ("backbone", 47, "w0_w1"),
    ("mtp", 0, "w2"),
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file():
        raise ValueError(f"{label} must be a canonical regular file: {path}")
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return document


def _validate_summary(document: dict[str, Any], *, label: str) -> None:
    expected = {
        "mode": STAGING_MODE,
        "status": "pass",
        "production_qualification": False,
        "device_opened": False,
        "device_locks_acquired": False,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise RuntimeError(f"{label} {key} differs: {document.get(key)!r} != {value!r}")
    if not isinstance(document.get("results"), list):
        raise RuntimeError(f"{label} results must be a list")


def _expected_path(artifact_root: Path, namespace: str, layer_index: int, name: str) -> Path:
    return artifact_root / namespace / f"layer-{layer_index:02d}" / ARTIFACT_SPECS[name]["filename"]


def _validate_regular_file(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.resolve(strict=True) != path:
        raise RuntimeError(f"artifact must be a canonical regular file, not an alias: {path}")
    return metadata


def _validate_evidence(
    evidence_path: Path,
    *,
    slot: tuple[str, int],
    artifacts: dict[str, Any],
    identity: dict[str, Any],
) -> None:
    document = _load_json(evidence_path, label=f"{slot[0]}:{slot[1]} evidence")
    if document.get("mode") != STAGING_MODE or document.get("production_qualification") is not False:
        raise RuntimeError(f"staging evidence mode differs: {evidence_path}")
    if document.get("device_opened") is not False or document.get("device_locks_acquired") is not False:
        raise RuntimeError(f"staging evidence claims device activity: {evidence_path}")
    if document.get("slot") != list(slot) or document.get("artifacts") != artifacts:
        raise RuntimeError(f"staging evidence content differs from its summary: {evidence_path}")
    for key, expected in identity.items():
        if document.get(key) != expected:
            raise RuntimeError(f"staging evidence {key} differs from its summary: {evidence_path}")


def verify_corpus(
    *,
    artifact_root: Path,
    layer0_summary_path: Path,
    remaining_summary_path: Path,
    verifier_source_head: str,
) -> dict[str, Any]:
    if (
        SHA256_PATTERN.fullmatch(verifier_source_head) is None
        and re.fullmatch(r"[0-9a-f]{40}", verifier_source_head) is None
    ):
        raise ValueError("--verifier-source-head must be an exact lowercase Git object ID")
    if not artifact_root.is_absolute():
        raise ValueError(f"artifact root must be absolute: {artifact_root}")
    artifact_root = artifact_root.resolve(strict=True)
    if not artifact_root.is_dir():
        raise ValueError(f"artifact root must be a directory: {artifact_root}")

    layer0 = _load_json(layer0_summary_path, label="layer-0 summary")
    remaining = _load_json(remaining_summary_path, label="remaining-layers summary")
    _validate_summary(layer0, label="layer-0 summary")
    _validate_summary(remaining, label="remaining-layers summary")
    if len(layer0["results"]) != 1 or len(remaining["results"]) != 48:
        raise RuntimeError(
            f"summary result counts differ: layer0={len(layer0['results'])}/1 remaining={len(remaining['results'])}/48"
        )
    identity = {
        "source_head": layer0.get("source_head"),
        "runtime": layer0.get("runtime"),
        "checkpoint": layer0.get("checkpoint"),
    }
    for key, expected in identity.items():
        if remaining.get(key) != expected:
            raise RuntimeError(f"staging summaries have different {key}")

    records: dict[tuple[str, int, str], dict[str, Any]] = {}
    expected_paths: set[Path] = set()
    total_bytes = 0
    results = layer0["results"] + remaining["results"]
    for result in results:
        if result.get("status") != "staged":
            raise RuntimeError(f"staging result status differs: {result.get('status')!r}")
        raw_slot = result.get("slot")
        if not isinstance(raw_slot, list) or len(raw_slot) != 2:
            raise RuntimeError(f"invalid staging result slot: {raw_slot!r}")
        slot = (raw_slot[0], raw_slot[1])
        if slot not in EXPECTED_SLOTS:
            raise RuntimeError(f"unexpected staging result slot: {slot}")
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_SPECS):
            raise RuntimeError(f"artifact names differ for {slot}: {sorted(artifacts or {})}")
        evidence = result.get("evidence")
        if not isinstance(evidence, str):
            raise RuntimeError(f"missing evidence path for {slot}")
        _validate_evidence(Path(evidence), slot=slot, artifacts=artifacts, identity=identity)

        for name, spec in ARTIFACT_SPECS.items():
            record = artifacts[name]
            key = (*slot, name)
            if key in records:
                raise RuntimeError(f"duplicate artifact record: {key}")
            expected_path = _expected_path(artifact_root, *slot, name)
            if record.get("path") != str(expected_path):
                raise RuntimeError(f"artifact path differs for {key}: {record.get('path')!r}")
            metadata = _validate_regular_file(expected_path)
            if record.get("bytes") != spec["bytes"] or metadata.st_size != spec["bytes"]:
                raise RuntimeError(
                    f"artifact byte count differs for {key}: summary={record.get('bytes')} "
                    f"file={metadata.st_size} expected={spec['bytes']}"
                )
            for field in ("local_shape", "global_shape"):
                if record.get(field) != spec[field]:
                    raise RuntimeError(f"artifact {field} differs for {key}: {record.get(field)!r}")
            if record.get("dtype") != "BFLOAT4_B" or record.get("layout") != "TILE":
                raise RuntimeError(f"artifact format differs for {key}")
            if SHA256_PATTERN.fullmatch(str(record.get("sha256"))) is None:
                raise RuntimeError(f"artifact digest is not exact lowercase SHA-256 for {key}")
            records[key] = record
            expected_paths.add(expected_path)
            total_bytes += metadata.st_size

    expected_keys = {(*slot, name) for slot in EXPECTED_SLOTS for name in ARTIFACT_SPECS}
    if set(records) != expected_keys:
        missing = sorted(expected_keys - set(records))
        extra = sorted(set(records) - expected_keys)
        raise RuntimeError(f"staging corpus slots differ: missing={missing} extra={extra}")
    live_paths = set(artifact_root.rglob("*.tensorbin"))
    if live_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - live_paths)
        extra = sorted(str(path) for path in live_paths - expected_paths)
        raise RuntimeError(f"staging corpus paths differ: missing={missing} extra={extra}")

    expected_total = len(EXPECTED_SLOTS) * sum(spec["bytes"] for spec in ARTIFACT_SPECS.values())
    expected_files = len(EXPECTED_SLOTS) * len(ARTIFACT_SPECS)
    if len(records) != expected_files or total_bytes != expected_total:
        raise RuntimeError(
            f"staging corpus aggregate differs: files={len(records)}/{expected_files} bytes={total_bytes}/{expected_total}"
        )

    selective_hashes = []
    for key in HASH_SAMPLE:
        path = Path(records[key]["path"])
        actual = _sha256(path)
        expected = records[key]["sha256"]
        if actual != expected:
            raise RuntimeError(f"sampled artifact digest differs for {key}: {actual} != {expected}")
        selective_hashes.append({"slot": list(key[:2]), "artifact": key[2], "path": str(path), "sha256": actual})

    return {
        "mode": VERIFY_MODE,
        "status": "pass",
        "production_qualification": False,
        "device_opened": False,
        "device_locks_acquired": False,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "completed_utc": _utc_now(),
        "verifier_source_head": verifier_source_head,
        "artifact_root": str(artifact_root),
        "source_summaries": [
            {"path": str(layer0_summary_path), "sha256": _sha256(layer0_summary_path)},
            {"path": str(remaining_summary_path), "sha256": _sha256(remaining_summary_path)},
        ],
        "staging_identity": identity,
        "corpus": {
            "slots": 49,
            "backbone_layers": list(range(48)),
            "mtp_layers": [0],
            "tensorbins": len(records),
            "bytes": total_bytes,
        },
        "selective_hashes": selective_hashes,
        "unhashed_tensorbins": len(records) - len(selective_hashes),
    }


def _write_json_exclusive(path: Path, document: dict[str, Any]) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError(f"output parent must be an existing absolute directory: {path}")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--layer0-summary", type=Path, required=True)
    parser.add_argument("--remaining-summary", type=Path, required=True)
    parser.add_argument("--verifier-source-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_corpus(
        artifact_root=args.artifact_root,
        layer0_summary_path=args.layer0_summary,
        remaining_summary_path=args.remaining_summary,
        verifier_source_head=args.verifier_source_head,
    )
    _write_json_exclusive(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
