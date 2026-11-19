# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Restore exact archived evidence bytes without changing recorded provenance.

Requires an explicit model root. Existing identical files are retained; conflicting
files are never replaced. Verification does not imply a stage-review verdict.
"""

import argparse
import hashlib
import json
import lzma
import os
import shutil
import tempfile
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests.archive_manifest import load_archive_manifest

CHUNK = 1024 * 1024


def safe_path(root, name):
    relative = Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError(f"Unsafe relative path: {name!r}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes model root: {name!r}")
    # Reject aliases, even an internal symlink: restoring must name its own file.
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
        raise ValueError(f"Symlink in artifact path: {name!r}")
    return path


def file_identity(path):
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            sha.update(block)
            size += len(block)
    return sha.hexdigest(), size


def restore_entry(root, entry, verify_only=False):
    root = root.resolve()
    target = safe_path(root, entry["source"])
    part_records = entry.get("archive_parts") or [entry["archive"]]
    if not isinstance(part_records, list) or not part_records:
        raise ValueError("Missing archive payload")
    names = [part["path"] if isinstance(part, dict) else part for part in part_records]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate archive parts")
    parts = [safe_path(root, name) for name in names]
    expected = (entry["source_sha256"], entry["source_bytes"])
    existed = target.exists()
    if existed and (not target.is_file() or file_identity(target) != expected):
        raise ValueError(f"Conflicting existing original: {entry['source']}")
    output_path = None
    try:
        # Disk spool keeps memory bounded even for large multipart raw logs.
        with tempfile.TemporaryFile() as packed:
            combined_sha = hashlib.sha256()
            combined_size = 0
            for record, path in zip(part_records, parts):
                sha = hashlib.sha256()
                size = 0
                with path.open("rb") as stream:
                    while block := stream.read(CHUNK):
                        packed.write(block)
                        sha.update(block)
                        size += len(block)
                        combined_sha.update(block)
                        combined_size += len(block)
                if isinstance(record, dict) and (sha.hexdigest() != record["sha256"] or size != record["bytes"]):
                    raise ValueError(f"Archive part identity mismatch: {path}")
            if (combined_sha.hexdigest(), combined_size) != (
                entry["archive_sha256"],
                entry["archive_bytes"],
            ):
                raise ValueError("Combined compressed identity mismatch")
            packed.seek(0)
            with tempfile.TemporaryFile() as restored, lzma.LZMAFile(packed) as source:
                sha = hashlib.sha256()
                size = 0
                while block := source.read(CHUNK):
                    restored.write(block)
                    sha.update(block)
                    size += len(block)
                    if size > entry["source_bytes"]:
                        raise ValueError("Decompressed size exceeds manifest")
                if (sha.hexdigest(), size) != expected:
                    raise ValueError("Decompressed source identity mismatch")
                if not verify_only and not existed:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    safe_path(root, entry["source"])
                    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".restore-", delete=False) as output:
                        output_path = Path(output.name)
                        restored.seek(0)
                        shutil.copyfileobj(restored, output, length=CHUNK)
                        output.flush()
                        os.fsync(output.fileno())
                    # Atomic no-replacement publication; a racing existing file fails.
                    os.link(output_path, target)
        return {
            "source": entry["source"],
            "status": ("verified_only" if verify_only else "existing_identical" if existed else "restored"),
            "sha256": expected[0],
            "bytes": expected[1],
        }
    finally:
        if output_path is not None:
            output_path.unlink(missing_ok=True)


def restore_manifest(root, manifest, sources=(), verify_only=False):
    root = root.resolve()
    document = load_archive_manifest(root, safe_path(root, manifest))
    entries = {entry["source"]: entry for entry in document["entries"]}
    requested = list(sources) if sources else list(entries)
    if len(requested) != len(set(requested)) or any(name not in entries for name in requested):
        raise ValueError("Requested sources must be unique manifest entries")
    # Validate every selected path before writing any selected original.
    for name in requested:
        safe_path(root, name)
        record = entries[name]
        for part in record.get("archive_parts") or [record["archive"]]:
            safe_path(root, part["path"] if isinstance(part, dict) else part)
    return [restore_entry(root, entries[name], verify_only) for name in requested]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", default="doc/functional_decoder/evidence_archives.json")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Model-relative archived original; repeat to select. Default: all.",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    rows = restore_manifest(args.root, args.manifest, args.source, args.verify_only)
    print(json.dumps({"stage_acceptance": "not_evaluated", "records": rows}))


if __name__ == "__main__":
    main()
