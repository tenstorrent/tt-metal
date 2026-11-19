# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Losslessly archive completed evidence within the repository's 500KiB/file limit.

Raw PT references and Tracy capture databases stay local. Human tables, rendered
CSV, JSON provenance, and compressed raw profiler rows/logs are retained.
--unpack reconstructs every archived original from the committed manifest.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unpack", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1] / "doc/multichip_decoder"
    manifest = root / "evidence_archive.json"
    if a.unpack:
        for item in json.loads(manifest.read_text())["files"]:
            blob = b"".join((root / name).read_bytes() for name in item["stored"])
            assert sha(blob) == item["stored_sha256"]
            data = gzip.decompress(blob) if item["compression"] == "gzip" else blob
            assert sha(data) == item["original_sha256"]
            (root / item["original"]).write_bytes(data)
        print("Reconstructed lossless evidence")
        return
    entries = []
    ignored = []
    # Only finished stage artifacts, not multi-gigabyte Tracy captures or PTs.
    for source in sorted(root.rglob("*")):
        if not source.is_file() or "tracy" in source.relative_to(root).parts:
            continue
        rel = str(source.relative_to(root))
        if source.name in ("launch.log", "autodebug_launch.log"):
            continue  # Private agent transcripts remain local; reports are retained.
        if source.suffix == ".log" or (
            source.suffix in (".json", ".txt", ".jsonl") and source.stat().st_size > 500 * 1024
        ):
            data = source.read_bytes()
            blob = gzip.compress(data, mtime=0)
            compression = "gzip"
            target = source.with_name(source.name + ".gz")
        elif source.name == "ops.csv.gz":
            data = source.read_bytes()
            blob = data
            compression = "none"
            target = source
        else:
            continue
        parts = []
        if len(blob) <= 450 * 1024:
            target.write_bytes(blob)
            parts = [str(target.relative_to(root))]
        else:
            ignored.append(str(target.relative_to(root)))
            for offset in range(0, len(blob), 450 * 1024):
                part = target.with_name(target.name + f".part{offset//(450*1024):03d}")
                part.write_bytes(blob[offset : offset + 450 * 1024])
                parts.append(str(part.relative_to(root)))
        if compression == "gzip" or len(blob) > 450 * 1024:
            ignored.append(rel)
        entries.append(
            dict(
                original=rel,
                original_bytes=len(data),
                original_sha256=sha(data),
                compression=compression,
                stored=parts,
                stored_sha256=sha(blob),
            )
        )
    manifest.write_text(
        json.dumps(
            {
                "format": 1,
                "reconstruct": "python -m models.autoports.qwen_qwen3_6_27b.tests.pack_multichip_evidence --unpack",
                "files": entries,
            },
            indent=2,
        )
        + "\n"
    )
    base = [
        "*.pt",
        "*.log",
        "tracy/",
        "**/ops.csv",
        "**/__pycache__/",
        "!*.log.gz",
        "!perf/**/prefill_perf_report.csv",
        "!perf/**/decode_perf_report.csv",
        "!/*.csv",
        "**/launch.log.gz*",
        "**/autodebug_launch.log.gz*",
    ]
    (root / ".gitignore").write_text(
        "\n".join(base + sorted(set("/" + v for v in ignored if not v.endswith(".log")))) + "\n"
    )
    print(f"Archived {len(entries)} completed artifacts losslessly")


if __name__ == "__main__":
    main()
