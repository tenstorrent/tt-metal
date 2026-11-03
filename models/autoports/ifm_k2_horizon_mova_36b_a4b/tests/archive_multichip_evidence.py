# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Losslessly archive large logs and hook-sensitive immutable source snapshots."""

import gzip
import hashlib
import json
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"


def main():
    manifest = DOC / "artifact_archives.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else {}
    paths = [*DOC.glob("logs/*.log"), *DOC.glob("watcher/**/*.log"), *DOC.glob("candidates/*.py.txt")]
    for path in paths:
        content = path.read_bytes()
        if content.startswith(b"Lossless archive:"):
            continue
        text = content.decode()
        sensitive = text.endswith("\n\n") or not text.endswith("\n") or any(x.rstrip() != x for x in text.splitlines())
        if len(content) <= 500000 and not sensitive:
            continue
        archive = path.with_name(path.name + ".gz")
        archive.write_bytes(gzip.compress(content, mtime=0))
        assert gzip.decompress(archive.read_bytes()) == content
        key = str(path.relative_to(DOC))
        records[key] = dict(
            archive=str(archive.relative_to(DOC)),
            bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        )
        if path.name.endswith(".py.txt"):
            path.unlink()
        else:
            path.write_text(f"Lossless archive: {archive.name}\nOriginal SHA256: {records[key]['sha256']}\n")
    manifest.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print("Archived artifacts", len(records))


if __name__ == "__main__":
    main()
