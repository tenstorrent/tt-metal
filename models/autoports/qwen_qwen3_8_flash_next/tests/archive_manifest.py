# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lossless archive index loading/writing with hash-bound JSON entry parts.

Schema1 inline manifests remain readable. Schema2 deliberately omits ``entries``
from the root so legacy direct readers fail instead of silently seeing no history.
Part order, each part digest/size/count, combined digest/count and source uniqueness
are verified before any archive entry is returned. This is packaging, not acceptance.
"""

import argparse
import hashlib
import json
from pathlib import Path


def encoded(value):
    return (json.dumps(value, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def unique_sources(entries):
    require(isinstance(entries, list), "Archive entries must be a list")
    sources = [entry["source"] for entry in entries]
    require(len(set(sources)) == len(sources), "Duplicate archive source identities")


def load_archive_manifest(root, path):
    root = root.resolve()
    document = json.loads(path.read_text())
    version = document.get("schema_version")
    if version == 1:
        require("entry_parts" not in document, "Ambiguous inline/part manifest")
        unique_sources(document["entries"])
        return document
    require(version == 2, "Unsupported archive manifest version")
    require("entries" not in document, "Schema2 cannot mix inline and part entries")
    records = document["entry_parts"]
    require(isinstance(records, list) and records, "Missing entry manifest parts")
    names = [record["path"] for record in records]
    require(len(set(names)) == len(names), "Duplicate entry manifest part")
    entries = []
    for record in records:
        relative = Path(record["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe manifest part path")
        part_path = (root / relative).resolve()
        require(part_path.is_relative_to(root), "Manifest part escapes model root")
        payload = part_path.read_bytes()
        require(
            len(payload) == record["bytes"] and sha256(payload) == record["sha256"], "Manifest part identity changed"
        )
        part = json.loads(payload)
        require(part.get("schema_version") == 1 and part.get("kind") == "archive_entries", "Wrong manifest part schema")
        require(len(part["entries"]) == record["entry_count"], "Manifest part count differs")
        entries.extend(part["entries"])
    require(len(entries) == document["entry_count"], "Combined archive count differs")
    require(sha256(encoded(entries)) == document["entries_sha256"], "Combined archive identity/order differs")
    unique_sources(entries)
    return {**document, "entries": entries}


def write_archive_manifest(root, path, document, max_part_bytes=300000):
    root = root.resolve()
    path = path.resolve()
    require(path.is_relative_to(root), "Manifest root is outside model")
    require(1000 <= max_part_bytes <= 350000, "Part byte limit must be1000..350000")
    entries = document["entries"]
    unique_sources(entries)
    require(entries, "Cannot partition an empty archive history")
    groups = []
    current = []
    for entry in entries:
        trial = {"schema_version": 1, "kind": "archive_entries", "entries": current + [entry]}
        if len(encoded(trial)) > max_part_bytes:
            require(current, "One archive entry exceeds part byte limit")
            groups.append(current)
            current = []
        current.append(entry)
        require(
            len(encoded({"schema_version": 1, "kind": "archive_entries", "entries": current})) <= max_part_bytes,
            "One archive entry exceeds part byte limit",
        )
    groups.append(current)
    parts = []
    for index, group in enumerate(groups):
        part_path = path.parent / "evidence_archive_parts" / f"entries_{index:03d}.json"
        payload = encoded({"schema_version": 1, "kind": "archive_entries", "entries": group})
        parts.append(
            (
                part_path,
                payload,
                {
                    "path": str(part_path.relative_to(root)),
                    "bytes": len(payload),
                    "sha256": sha256(payload),
                    "entry_count": len(group),
                },
            )
        )
    metadata = {
        key: value
        for key, value in document.items()
        if key not in ("entries", "entry_parts", "entry_count", "entries_sha256")
    }
    metadata.update(
        schema_version=2,
        entry_parts=[record for _, _, record in parts],
        entry_count=len(entries),
        entries_sha256=sha256(encoded(entries)),
    )
    payload = encoded(metadata)
    require(len(payload) <= 350000, "Root metadata exceeds350KB; retain history in separate referenced metadata")
    for part_path, part_bytes, _ in parts:
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_bytes(part_bytes)
    path.write_bytes(payload)
    loaded = load_archive_manifest(root, path)
    require(loaded["entries"] == entries, "Written manifest lost/reordered entries")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("migrate", "verify"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    path = args.manifest or root / "doc/functional_decoder/evidence_archives.json"
    document = load_archive_manifest(root, path)
    if args.action == "migrate":
        write_archive_manifest(root, path, document)
        document = load_archive_manifest(root, path)
    from .summarize_functional_evidence import verify_archive

    results = [verify_archive(root, entry) for entry in document["entries"]]
    failures = [
        row for row in results if row["integrity"] != "verified" or row["original_status"] == "differs_from_archive"
    ]
    report = {
        "manifest": str(path),
        "manifest_sha256": sha256(path.read_bytes()),
        "manifest_parts": document.get("entry_parts", []),
        "archive_entries": len(results),
        "failures": failures,
        "all_verified": not failures,
        "stage_acceptance": "not_evaluated",
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    require(not failures, "Archive verification failed")


if __name__ == "__main__":
    main()
