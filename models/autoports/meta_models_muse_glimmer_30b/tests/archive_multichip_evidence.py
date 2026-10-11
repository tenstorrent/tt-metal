# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Preserve exact log bytes and split large gzip streams for the repo size gate."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "doc/multichip_decoder")
ROOT = parser.parse_args().root
manifest = []
for path in sorted(ROOT.rglob("*.log")):
    if any(part in ("raw", "generated") or part.startswith("kernel_cache") for part in path.relative_to(ROOT).parts):
        continue
    blob = path.read_bytes()
    archive = path.with_suffix(path.suffix + ".gz")
    archive.write_bytes(gzip.compress(blob, mtime=0))
    manifest.append(
        dict(
            source=str(path.relative_to(ROOT)),
            archive=str(archive.relative_to(ROOT)),
            bytes=len(blob),
            sha256=hashlib.sha256(blob).hexdigest(),
        )
    )
# Retain the actual watcher samples and their kernel maps, not only stdout.
# The remaining generated/raw tree stays on persistent local storage.
for directory in sorted(ROOT.glob("watcher*/generated/watcher")):
    for path in sorted(directory.iterdir()):
        if path.name not in ("watcher.log", "kernel_names.txt", "kernel_elf_paths.txt"):
            continue
        blob = path.read_bytes()
        archive = ROOT / "watcher_artifacts" / directory.parents[1].name / (path.name + ".gz")
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(gzip.compress(blob, mtime=0))
        manifest.append(
            dict(
                source=str(path.relative_to(ROOT)),
                archive=str(archive.relative_to(ROOT)),
                bytes=len(blob),
                sha256=hashlib.sha256(blob).hexdigest(),
            )
        )
ignored = []
structured = []
for path in sorted(ROOT.rglob("*")):
    if path.suffix not in (".json", ".csv") or not path.is_file() or path.stat().st_size <= 480000:
        continue
    if any(part in ("raw", "generated") or part.startswith("kernel_cache") for part in path.relative_to(ROOT).parts):
        continue
    blob = path.read_bytes()
    archive = path.with_suffix(path.suffix + ".gz")
    archive.write_bytes(gzip.compress(blob, mtime=0))
    structured.append(
        dict(
            source=str(path.relative_to(ROOT)),
            archive=str(archive.relative_to(ROOT)),
            bytes=len(blob),
            sha256=hashlib.sha256(blob).hexdigest(),
        )
    )
    ignored.append("/" + str(path.relative_to(ROOT)))
(ROOT / "structured_archive_manifest.json").write_text(json.dumps(structured, indent=2) + "\n")
for path in sorted(ROOT.rglob("*.gz")):
    if "raw" in path.relative_to(ROOT).parts or path.stat().st_size <= 480000:
        continue
    blob = path.read_bytes()
    parts = []
    for index, offset in enumerate(range(0, len(blob), 480000)):
        part = path.with_name(path.name + f".part{index:03}")
        data = blob[offset : offset + 480000]
        part.write_bytes(data)
        parts.append(dict(path=part.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    split = dict(
        archive=path.name,
        bytes=len(blob),
        sha256=hashlib.sha256(blob).hexdigest(),
        uncompressed_sha256=hashlib.sha256(gzip.decompress(blob)).hexdigest(),
        parts=parts,
        restore="Concatenate parts in the listed order to reconstruct the original gzip stream; verify SHA256 before decompressing.",
    )
    path.with_name(path.name + ".manifest.json").write_text(json.dumps(split, indent=2) + "\n")
    ignored.append("/" + str(path.relative_to(ROOT)))
ignore_path = ROOT / ".gitignore"
existing = ignore_path.read_text().splitlines()
for entry in ignored:
    if entry not in existing:
        existing.append(entry)
ignore_path.write_text("\n".join(existing) + "\n")
(ROOT / "log_archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Archived {len(manifest)} exact logs; split {len(ignored)} large gzip streams.")
