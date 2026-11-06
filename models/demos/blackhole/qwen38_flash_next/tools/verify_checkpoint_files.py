# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hash the complete exact HF snapshot and corroborate common ModelScope files."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify(checkpoint: Path, modelscope_tree: Path, workers: int = 2) -> dict:
    source = json.loads(modelscope_tree.read_text())
    release_files = {
        entry["Path"]: entry
        for entry in source["Data"]["Files"]
        if entry.get("Type") == "blob" and "/" not in entry["Path"]
    }
    local_paths = sorted(path for path in checkpoint.iterdir() if path.is_file())
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        digests = dict(zip(local_paths, pool.map(_sha256, local_paths)))

    files = []
    mismatches = []
    for path in local_paths:
        entry = release_files.get(path.name)
        item = {"path": path.name, "size": path.stat().st_size, "sha256": digests[path]}
        if entry is not None:
            item["modelscope"] = {
                "revision": entry["Revision"],
                "size": entry["Size"],
                "sha256": entry["Sha256"],
                "size_match": path.stat().st_size == entry["Size"],
                "sha256_match": digests[path] == entry["Sha256"],
            }
            if path.suffix == ".safetensors" and not (
                item["modelscope"]["size_match"] and item["modelscope"]["sha256_match"]
            ):
                mismatches.append(path.name)
        files.append(item)

    weight_files = [item for item in files if item["path"].endswith(".safetensors")]
    if len(weight_files) != 131:
        raise ValueError(f"expected 131 weight shards, found {len(weight_files)}")
    if sum(item["size"] for item in weight_files) != 360_000_192_888:
        raise ValueError("weight-shard file bytes differ from the pinned HF snapshot")
    if mismatches:
        raise ValueError(f"HF/ModelScope weight mismatch: {mismatches}")

    return {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": str(checkpoint),
        "modelscope_tree": str(modelscope_tree),
        "modelscope_revision": "2741eec155d03a8ce151b993ccce1a7b1e398d6b",
        "file_count": len(files),
        "weight_shard_count": len(weight_files),
        "weight_file_bytes": sum(item["size"] for item in weight_files),
        "modelscope_weight_hashes_all_match": True,
        "modelscope_only_files": sorted(set(release_files) - {path.name for path in local_paths}),
        "provider_metadata_differences": sorted(
            item["path"]
            for item in files
            if "modelscope" in item
            and not (item["modelscope"]["size_match"] and item["modelscope"]["sha256_match"])
            and not item["path"].endswith(".safetensors")
        ),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--modelscope-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("--workers must be in [1, 4]")
    result = verify(args.checkpoint.resolve(), args.modelscope_tree.resolve(), args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, sort_keys=True))


if __name__ == "__main__":
    main()
