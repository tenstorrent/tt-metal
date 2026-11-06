#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Retain exact large/raw log bytes in deterministic gzip archives for local Git."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    index = args.directory / "raw_archives.json"
    if args.restore:
        for row in json.loads(index.read_text()):
            path = args.directory / row["file"]
            data = gzip.decompress((args.directory / row["archive"]).read_bytes())
            assert sha(data) == row["sha256"]
            if path.exists():
                assert sha(path.read_bytes()) == row["sha256"]
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        return
    rows = []
    for path in sorted(args.directory.rglob("*")):
        if not path.is_file() or path.suffix == ".gz" or path == index or path.name.endswith(".sock"):
            continue
        data = path.read_bytes()
        if len(data) < 450000 and path.suffix != ".log":
            continue
        archive = path.with_name(path.name + ".gz")
        if archive.exists() and gzip.decompress(archive.read_bytes()) != data:
            archive = path.with_name(path.name + ".exact.gz")
        archive.write_bytes(gzip.compress(data, mtime=0))
        rows.append(
            {
                "file": str(path.relative_to(args.directory)),
                "archive": str(archive.relative_to(args.directory)),
                "bytes": len(data),
                "sha256": sha(data),
                "archive_sha256": sha(archive.read_bytes()),
            }
        )
    index.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"archived_files": len(rows), "originals_retained_locally": True}))


if __name__ == "__main__":
    main()
