# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Archive immutable logs/large CSVs; retain raw local files and hash provenance."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "doc/fused_decoder"
records = []
ignored = ["!*.csv", "!generated/", "!*.gz", "__pycache__/", "*.pyc"]
for path in sorted(EVIDENCE.rglob("*")):
    if not path.is_file() or path.suffix == ".gz":
        continue
    relative = path.relative_to(EVIDENCE).as_posix()
    if path.suffix == ".tracy" or path.name in ("profile_log_device.csv", "tracy_ops_times.csv"):
        data = path.read_bytes()
        records.append(
            {
                "path": relative,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "retention": "local raw capture; postprocessed ops CSV archived",
            }
        )
        ignored.append("/" + relative)
    elif path.suffix == ".log" or (
        path.stat().st_size > 500 * 1024 and path.suffix in (".csv", ".txt", ".yaml", ".json")
    ):
        data = path.read_bytes()
        compressed = gzip.compress(data, mtime=0)
        target = Path(str(path) + ".gz")
        target.write_bytes(compressed)
        records.append(
            {
                "path": relative,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "archive": target.relative_to(EVIDENCE).as_posix(),
                "archive_size": len(compressed),
            }
        )
        ignored.append("/" + relative)
(EVIDENCE / "archive_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
(EVIDENCE / ".gitignore").write_text(
    "# Raw evidence remains local; lossless archives are committed.\n" + "\n".join(ignored) + "\n"
)
print(f"Archived {len(records)} raw evidence files")
