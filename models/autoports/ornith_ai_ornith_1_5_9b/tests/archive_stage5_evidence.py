# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Archive completed stage-five evidence without deleting local raw captures.

Run only after device/profiler queues have stopped. Large binary profiler and
failure captures remain local with hashes; postprocessed CSVs and logs get
lossless, deterministic gzip archives for the stage commit.
"""

import gzip
import hashlib
import json
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"
    records = []
    hashes = {}
    compressed_by_hash = {}
    ignored = ["!*.csv", "!*.gz", "__pycache__/", "*.pyc", "/STOP"]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".gz" or path.name in {"archive_manifest.json", ".gitignore", "STOP"}:
            continue
        relative = path.relative_to(root).as_posix()
        local_only = path.suffix in {".tracy", ".pt"} or path.name in {
            "profile_log_device.csv",
            "tracy_ops_times.csv",
            "tracy_ops_data.csv",
        }
        archive = path.suffix == ".log" or (
            path.stat().st_size > 512 * 1024 and path.suffix in {".csv", ".txt", ".json", ".yaml"}
        )
        if not local_only and not archive:
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity not in hashes:
            with path.open("rb") as source:
                hashes[identity] = hashlib.file_digest(source, "sha256").hexdigest()
        digest = hashes[identity]
        row = {"path": relative, "size": stat.st_size, "sha256": digest}
        if local_only:
            row["retention"] = "local raw capture; compact diagnosis and postprocessed CSV retained in commit"
        else:
            target = Path(str(path) + ".gz")
            if digest in compressed_by_hash:
                shutil.copy2(compressed_by_hash[digest], target)
            else:
                with (
                    path.open("rb") as source,
                    target.open("wb") as output,
                    gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed,
                ):
                    shutil.copyfileobj(source, compressed, length=1024 * 1024)
                compressed_by_hash[digest] = target
            row.update(archive=target.relative_to(root).as_posix(), archive_size=target.stat().st_size)
        records.append(row)
        ignored.append("/" + relative)
    (root / "archive_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    (root / ".gitignore").write_text(
        "# Raw evidence remains local; deterministic lossless archives are committed.\n" + "\n".join(ignored) + "\n"
    )
    print(f"ARCHIVE_PASS {len(records)} files")


if __name__ == "__main__":
    main()
