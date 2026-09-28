#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Preserve raw serving evidence before formatting; verify it afterward.

Run snapshot only after writers stop. No staging, ignore-rule changes, report
updates, or native-file edits are performed. Publish every listed gzip and the
manifest; publish browsable originals only where publish_original is true.
"""

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

MANIFEST = "publication_manifest.json"
LIMIT = 500 * 1024
EXCLUDED = {"runtime", "vllm_cache", "inspector", "cache", "__pycache__"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def discover(root):
    files = []
    for directory, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if name not in EXCLUDED and not name.endswith("_runtime") and not (Path(directory) / name).is_symlink()
        )
        for name in sorted(names):
            path = Path(directory) / name
            if name == MANIFEST or path.suffix not in {".json", ".log"}:
                continue
            if path.is_symlink():
                raise ValueError(f"Refusing evidence symlink: {path}")
            files.append(path.relative_to(root).as_posix())
    return sorted(files)


def json_value(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def equivalent_json(left, right):
    # Type-sensitive: bool/int and int/float substitutions are not formatting.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equivalent_json(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(equivalent_json(a, b) for a, b in zip(left, right))
    return left == right


def log_normalized(data):
    # Only whitespace cleanup and EOF/line-ending normalization are allowed.
    return b"\n".join(line.rstrip(b" \t") for line in data.splitlines()).rstrip(b"\n")


def create_or_verify(path, data):
    if path.is_symlink():
        raise ValueError(f"Refusing output symlink: {path}")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing differing overwrite: {path}")
        return
    with path.open("xb") as output:
        output.write(data)


def snapshot(root):
    manifest_path = root / MANIFEST
    if manifest_path.exists():
        # Never rebase original hashes on already-formatted content.
        verify(root, finalize=False)
        return
    names = discover(root)
    if not names:
        raise ValueError("No raw JSON/log evidence found")
    entries = []
    for name in names:
        path = root / name
        original = path.read_bytes()
        if path.suffix == ".json":
            json_value(original)
        compressed = gzip.compress(original, compresslevel=9, mtime=0)
        if len(compressed) > LIMIT:
            raise ValueError(f"Compressed artifact exceeds publication limit: {name}")
        create_or_verify(root / (name + ".gz"), compressed)
        if path.read_bytes() != original:
            raise ValueError(f"Evidence changed during snapshot: {name}")
        entries.append(
            {
                "path": name,
                "original_sha256": digest(original),
                "original_size": len(original),
                "gzip_path": name + ".gz",
                "gzip_sha256": digest(compressed),
                "gzip_size": len(compressed),
                "publish_original": len(original) <= LIMIT,
                "normalized_sha256": None,
                "normalized_size": None,
            }
        )
    if discover(root) != names:
        raise ValueError("Evidence file set changed during snapshot")
    manifest = {"schema": 1, "status": "snapshotted", "files": entries}
    data = (json.dumps(manifest, indent=2) + "\n").encode()
    if len(data) > LIMIT:
        raise ValueError("Manifest exceeds publication limit")
    create_or_verify(manifest_path, data)
    verify(root, finalize=False)


def verify(root, finalize, published=False):
    if published and finalize:
        raise ValueError("Published-checkout verification cannot finalize")
    manifest_path = root / MANIFEST
    if manifest_path.is_symlink():
        raise ValueError("Manifest cannot be a symlink")
    before = manifest_path.read_bytes()
    manifest = json_value(before)
    if manifest["schema"] != 1:
        raise ValueError("Unsupported manifest schema")
    if manifest["status"] not in {"snapshotted", "finalized"}:
        raise ValueError("Unsupported manifest status")
    names = [entry["path"] for entry in manifest["files"]]
    present = set(discover(root))
    required = {entry["path"] for entry in manifest["files"] if entry["publish_original"]}
    if len(names) != len(set(names)):
        raise ValueError("Duplicate manifest paths")
    if published:
        if not required <= present or not present <= set(names):
            raise ValueError("Published evidence file set differs from manifest")
    elif set(names) != present:
        raise ValueError("Evidence file set differs from snapshot")
    for entry in manifest["files"]:
        name = entry["path"]
        if Path(name).is_absolute() or ".." in Path(name).parts or entry["gzip_path"] != name + ".gz":
            raise ValueError("Unsafe manifest path")
        archive = root / entry["gzip_path"]
        if archive.is_symlink():
            raise ValueError(f"Archive cannot be a symlink: {name}")
        compressed = archive.read_bytes()
        if digest(compressed) != entry["gzip_sha256"] or len(compressed) != entry["gzip_size"]:
            raise ValueError(f"Compressed evidence changed: {name}")
        original = gzip.decompress(compressed)
        if digest(original) != entry["original_sha256"] or len(original) != entry["original_size"]:
            raise ValueError(f"Original evidence hash mismatch: {name}")
        current = (root / name).read_bytes() if name in present else original
        equivalent = (
            equivalent_json(json_value(original), json_value(current))
            if name.endswith(".json")
            else log_normalized(original) == log_normalized(current)
        )
        if not equivalent:
            raise ValueError(f"Non-formatting evidence change: {name}")
        if not entry["publish_original"] and current != original:
            raise ValueError(f"Gzip-only local original must remain byte-exact: {name}")
        if entry["publish_original"] and len(current) > LIMIT:
            raise ValueError(f"Normalized artifact exceeds publication limit: {name}")
        if manifest["status"] == "finalized" and digest(current) != entry["normalized_sha256"]:
            raise ValueError(f"Finalized evidence changed: {name}")
        if finalize:
            entry["normalized_sha256"] = digest(current)
            entry["normalized_size"] = len(current)
    if finalize and manifest["status"] != "finalized":
        manifest["status"] = "finalized"
        if manifest_path.read_bytes() != before:
            raise ValueError("Manifest changed during verification")
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        if len(data) > LIMIT:
            raise ValueError("Finalized manifest exceeds publication limit")
        # Only this utility-owned manifest is updated, after all checks pass.
        with tempfile.NamedTemporaryFile(dir=root, prefix=".publication-", delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        try:
            os.replace(temporary, manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Verified {len(names)} artifacts; status={manifest['status']}")
    for entry in manifest["files"]:
        if not entry["publish_original"]:
            print(f"Gzip-only publication; retain original locally and ignore exact path: {entry['path']}")


def self_test():
    with tempfile.TemporaryDirectory(prefix="ttft-publication-test-") as directory:
        root = Path(directory)
        (root / "raw.json").write_bytes(b'{"value": [1, true]}')
        (root / "raw.log").write_bytes(b"line  \n")
        (root / "large.json").write_text(json.dumps({"text": "x" * (LIMIT + 1)}))
        (root / "runtime").mkdir()
        (root / "runtime" / "excluded.json").write_text("invalid")
        snapshot(root)
        (root / "raw.json").write_text('{\n  "value": [1, true]\n}\n')
        (root / "raw.log").write_bytes(b"line\n")
        verify(root, finalize=True)
        snapshot(root)
        large_original = (root / "large.json").read_bytes()
        (root / "large.json").unlink()
        verify(root, finalize=False, published=True)
        try:
            verify(root, finalize=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Local finalize accepted missing original")
        (root / "unknown.json").write_text("{}")
        try:
            verify(root, finalize=False, published=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown published artifact accepted")
        (root / "unknown.json").unlink()
        browsable = (root / "raw.json").read_bytes()
        (root / "raw.json").unlink()
        try:
            verify(root, finalize=False, published=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing browsable artifact accepted")
        (root / "raw.json").write_bytes(browsable)
        (root / "large.json").write_bytes(large_original)
        (root / "raw.json").write_text('{"value": [2, true]}')
        try:
            verify(root, finalize=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Semantic mutation accepted")
        try:
            create_or_verify(root / "raw.log.gz", b"different")
        except ValueError:
            pass
        else:
            raise AssertionError("Differing archive overwrite accepted")
        (root / "raw.json").write_text('{\n  "value": [1, true]\n}\n')
        (root / "raw.log.gz").write_bytes(b"corrupt")
        try:
            verify(root, finalize=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt archive accepted")
        assert not equivalent_json(True, 1)
    print("Self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["snapshot", "finalize", "verify", "self-test"])
    parser.add_argument("--root", type=Path, help="Stage-owned readiness_vllm/ttft_optimization directory")
    args = parser.parse_args()
    if args.mode == "self-test":
        self_test()
        return
    if args.root is None or not args.root.is_dir() or args.root.is_symlink():
        parser.error("--root must be an existing, non-symlink stage evidence directory")
    root = args.root.resolve()
    if root.name != "ttft_optimization" or root.parent.name != "readiness_vllm":
        parser.error("--root must identify readiness_vllm/ttft_optimization")
    if args.mode == "snapshot":
        snapshot(root)
    else:
        verify(root, finalize=args.mode == "finalize", published=args.mode == "verify")


if __name__ == "__main__":
    main()
