# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep compact compressed CSV evidence in git and archive bulky raw captures."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path

root = Path("models/autoports/qwen_qwen3_8_27b/doc/full_model")
archive = Path("/home/mvasiljevic/qwen38-full-rerun/artifacts/full_model")
records = []
for directory in sorted((root / "tracy").iterdir()):
    if not directory.is_dir():
        continue
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        oversized_combined = (
            path.name.startswith("ops_perf_results") and path.name.endswith(".gz") and path.stat().st_size > 500 * 1024
        )
        if path.name.endswith(".gz") and not oversized_combined:
            continue
        raw = ".logs" in path.parts or "reports" in path.parts or path.name.endswith("_ops.csv")
        if not raw:
            continue
        if not path.name.endswith(".gz") and (
            path.name.startswith("ops_perf_results") or path.name.endswith("_ops.csv")
        ):
            with path.open("rb") as source, gzip.open(path.with_suffix(path.suffix + ".gz"), "wb") as out:
                shutil.copyfileobj(source, out)
        destination = archive / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        records.append(dict(original=str(path), archive=str(destination), bytes=path.stat().st_size, sha256=digest))
        shutil.move(path, destination)
manifest = root / "profile_archive.json"
previous = json.loads(manifest.read_text()) if manifest.exists() else []
manifest.write_text(json.dumps(previous + records, indent=2) + "\n")
