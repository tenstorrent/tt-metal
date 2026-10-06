# SPDX-License-Identifier: Apache-2.0
"""Preserve large textual evidence as exact gzip; restore it for report regeneration."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "doc/multichip_decoder"
MANIFEST = ROOT / "artifact_manifest.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main(restore=False):
    if restore:
        for row in json.loads(MANIFEST.read_text()):
            parts = row.get("archive_parts", [row.get("archive")])
            data = gzip.decompress(b"".join((ROOT / part).read_bytes() for part in parts))
            assert len(data) == row["bytes"] and digest(data) == row["sha256"]
            target = ROOT / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return
    rows = []
    ignored = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.stat().st_size <= 450000 and path.name != "watcher.log" and not path.name.endswith(".py.txt"):
            continue
        if path.suffix in (".pt", ".tracy", ".gz") or path.name == "profile_log_device.csv":
            continue
        if any(part in (".logs", ".tracy") for part in path.parts):
            continue
        data = path.read_bytes()
        rel = str(path.relative_to(ROOT))
        archive = ROOT / "archives" / (rel.replace("/", "__") + ".gz")
        archive.parent.mkdir(exist_ok=True)
        compressed = gzip.compress(data, compresslevel=9, mtime=0)
        row = dict(path=rel, bytes=len(data), sha256=digest(data))
        if len(compressed) <= 450000:
            archive.write_bytes(compressed)
            row["archive"] = str(archive.relative_to(ROOT))
        else:
            row["archive_parts"] = []
            for index, start in enumerate(range(0, len(compressed), 450000)):
                part = archive.with_name(archive.name + f".part{index:03d}")
                part.write_bytes(compressed[start : start + 450000])
                row["archive_parts"].append(str(part.relative_to(ROOT)))
            archive.unlink(missing_ok=True)
        rows.append(row)
        ignored.append("/" + rel)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n")
    marker = "# Generated exact originals (restore with tests/multichip_evidence_archive.py --restore)"
    ignore_path = ROOT / ".gitignore"
    prefix = ignore_path.read_text().split(marker)[0].rstrip()
    ignore_path.write_text(prefix + "\n" + marker + "\n" + "\n".join(ignored) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    main(parser.parse_args().restore)
