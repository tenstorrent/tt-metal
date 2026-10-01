# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify every pinned release file without network access or checkpoint writes."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RELEASE_REVISION = "2741eec155d03a8ce151b993ccce1a7b1e398d6b"
# The immutable listing used to provision and qualify the 145-file release.
MANIFEST_SHA256 = "2618cf53db8e04d539d96bb03d3f5c6e08f8ad3dddd0a54252d5d1b739d84ded"


def _verify_file(checkpoint: Path, entry: dict) -> dict:
    path = checkpoint / entry["Path"]
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            size += len(block)
            digest.update(block)
    actual = digest.hexdigest()
    if size != entry["Size"] or actual != entry["Sha256"]:
        raise ValueError(f"Release file differs from its pinned size or SHA256: {entry['Path']}")
    return {"path": entry["Path"], "bytes": size, "sha256": actual}


def verify(checkpoint: Path, manifest: Path, workers: int = 2) -> dict:
    if not 1 <= workers <= 4:
        raise ValueError("workers must be in [1, 4]")
    payload = manifest.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Release manifest differs from the pinned SHA256")
    entries = [entry for entry in json.loads(payload)["Data"]["Files"] if entry["Type"] == "blob"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(lambda entry: _verify_file(checkpoint, entry), entries))
    return {
        "revision": RELEASE_REVISION,
        "manifest_sha256": MANIFEST_SHA256,
        "verified_files": len(files),
        "verified_bytes": sum(file["bytes"] for file in files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.checkpoint.resolve()):
        raise ValueError("Write the verification report outside the read-only checkpoint directory")
    result = verify(args.checkpoint, args.manifest, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, sort_keys=True))


if __name__ == "__main__":
    main()
