# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Preserve compact ignored logs/tables without raw profiler dumps or tensors."""

import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
archive = root / "preserved_logs"
files = list(root.glob("*.log")) + list(root.glob("*.csv")) + list(root.glob("*.patch"))
for folder in ("profile_final_sliding", "profile_final_full", "watcher_failure"):
    for pattern in ("*.log", "*.csv", "*_table.txt", "*.txt", "native_decode_policy_rows.json"):
        files.extend((root / folder).glob(pattern))
manifest = []
for source in sorted(set(files)):
    relative = source.relative_to(root)
    target = archive / (str(relative) + ".gz")
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    packed = gzip.compress(raw, mtime=0)
    target.write_bytes(packed)
    manifest.append(
        dict(
            source=str(relative),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            archive=str(target.relative_to(root)),
            archive_sha256=hashlib.sha256(packed).hexdigest(),
            source_bytes=len(raw),
            archive_bytes=len(packed),
        )
    )
(root / "preserved_evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Preserved {len(manifest)} compact evidence files; raw captures/tensors stay local.")
