# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Restore or verify losslessly archived fused-stage text evidence without TTNN.

Run this after a fresh checkout before the evidence checker. Existing matching
files are retained; conflicting files fail instead of overwriting later runs.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def contained(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"Invalid archive path: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name or name == ".":
        raise ValueError(f"Unsafe/noncanonical archive path: {name}")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Archive path escapes evidence root: {name}")
    return path


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in archive manifest: {key}")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=unique)


def read_archive_manifest(root):
    """Validate all index parts before restoring any evidence file."""
    root = root.resolve()
    manifest = strict_json((root / "archive_manifest.json").read_bytes())
    if "files" in manifest:
        if "manifest_parts" in manifest:
            raise ValueError("Ambiguous legacy and partitioned manifest")
        records = manifest["files"]
    else:
        if manifest.get("manifest_format") != "partitioned-files-v1":
            raise ValueError("Unknown archive manifest format")
        parts = manifest.get("manifest_parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError("Empty/invalid manifest parts")
        records = {}
        for number, part in enumerate(parts):
            expected = f"archive_manifest.part{number:03d}.json"
            if part.get("path") != expected:
                raise ValueError(f"Missing, duplicate, unsafe or unordered manifest part: {part.get('path')}")
            path = contained(root, expected)
            try:
                payload = path.read_bytes()
            except OSError as error:
                raise ValueError(f"Missing/unreadable manifest part: {expected}") from error
            if len(payload) != part["bytes"] or hashlib.sha256(payload).hexdigest() != part["sha256"]:
                raise ValueError(f"Manifest part length or SHA256 mismatch: {expected}")
            body = strict_json(payload)
            if set(body) != {"files"} or not isinstance(body["files"], dict):
                raise ValueError(f"Invalid manifest part body: {expected}")
            if (
                type(part["file_count"]) is not int
                or part["file_count"] <= 0
                or len(body["files"]) != part["file_count"]
            ):
                raise ValueError(f"Manifest part file count mismatch: {expected}")
            duplicates = records.keys() & body["files"].keys()
            if duplicates:
                raise ValueError(f"Duplicate evidence paths across manifest parts: {sorted(duplicates)}")
            records.update(body["files"])
        if type(manifest["file_count"]) is not int or len(records) != manifest["file_count"]:
            raise ValueError("Manifest total file count mismatch")
    if not isinstance(records, dict):
        raise ValueError("Invalid manifest files map")
    for name, entry in records.items():
        contained(root, name)
        if ("archive" in entry) == ("archive_parts" in entry):
            raise ValueError(f"Ambiguous/missing archive payload: {name}")
        parts = entry.get("archive_parts") or [entry.get("archive")]
        if not isinstance(parts, list) or not parts or not all(isinstance(part, str) for part in parts):
            raise ValueError(f"Invalid archive payload parts: {name}")
        if len(parts) != len(set(parts)):
            raise ValueError(f"Duplicate archive payload parts: {name}")
        for part in parts:
            contained(root, part)
    return records


def restore(root, *, verify_only=False):
    root = root.resolve()
    records = read_archive_manifest(root)
    verified = restored = 0
    for name, entry in records.items():
        parts = entry.get("archive_parts") or [entry["archive"]]
        raw = gzip.decompress(b"".join(contained(root, part).read_bytes() for part in parts))
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Archive length or SHA256 mismatch: {name}")
        destination = contained(root, name)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Existing evidence differs from archive; retained unchanged: {name}")
        elif not verify_only:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also protects against a writer starting during restoration.
            with destination.open("xb") as output:
                output.write(raw)
            restored += 1
        verified += 1
    return dict(verified_files=verified, restored_files=restored, verify_only=verify_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=Path(__file__).parents[1] / "doc/fused_decoder")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(restore(args.doc, verify_only=args.verify_only), indent=2))


if __name__ == "__main__":
    main()
