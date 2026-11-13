#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Retain byte-exact raw evidence in Git-sized archive parts; restore on checkout."""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "doc/benchmark"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    archive = BASE / "raw_archive"
    manifest_path = archive / "manifest.json"
    if args.restore:
        manifest = json.loads(manifest_path.read_text())
        chunks = []
        for part in manifest["parts"]:
            data = (archive / part["file"]).read_bytes()
            assert sha(data) == part["sha256"], part["file"]
            chunks.append(data)
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(b"".join(chunks))), mode="r:") as bundle:
            for member in bundle:
                target = BASE / member.name
                assert target.resolve().is_relative_to(BASE.resolve()) and member.isfile()
                data = bundle.extractfile(member).read()
                assert sha(data) == manifest["files"][member.name]["sha256"], member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        print(f"Restored {len(manifest['files'])} byte-exact evidence files")
        return
    archive.mkdir(exist_ok=True)
    output = io.BytesIO()
    manifest = {"format": "gzip tar split into <=400000-byte parts", "files": {}, "parts": []}
    with tarfile.open(fileobj=output, mode="w:") as bundle:
        paths = list((BASE / "run").rglob("*")) + list((BASE / "attempts").rglob("*"))
        paths += list((BASE / "setup").glob("*.log"))
        for path in sorted(paths):
            if path.parent == BASE / "run" and path.name in ("REPORT.md", "runner-REPORT.md", "final_timing.json"):
                continue
            if not path.is_file():
                continue
            data = path.read_bytes()
            name = str(path.relative_to(BASE))
            manifest["files"][name] = {"bytes": len(data), "sha256": sha(data)}
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    compressed = gzip.compress(output.getvalue(), mtime=0)
    for i, start in enumerate(range(0, len(compressed), 400000)):
        name = f"evidence.tar.gz.part{i:04d}"
        data = compressed[start : start + 400000]
        (archive / name).write_bytes(data)
        manifest["parts"].append({"file": name, "bytes": len(data), "sha256": sha(data)})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Archived {len(manifest['files'])} files in {len(manifest['parts'])} parts; originals retained")


if __name__ == "__main__":
    main()
