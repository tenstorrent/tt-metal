# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Archive large raw profiler files while preserving compact reproducible evidence."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--stage", choices=["multichip_decoder", "optimized_multichip_decoder"], default="multichip_decoder"
)
parser.add_argument("--subdir", type=Path, help="Archive only a completed subtree while other runs are active")
args = parser.parse_args()
DOC = Path(__file__).resolve().parents[1] / "doc" / args.stage
scope = (DOC / args.subdir).resolve() if args.subdir else DOC
assert scope.is_relative_to(DOC.resolve()) and scope.is_dir(), scope
archive_name = "multichip_raw_archive" if args.stage == "multichip_decoder" else "optimized_multichip_raw_archive"
ARCHIVE = Path("/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache") / archive_name
manifest = DOC / "raw_archive_manifest.json"
entries = json.loads(manifest.read_text()) if manifest.exists() else []
for path in sorted(scope.rglob("*")):
    if not path.is_file() or path.stat().st_size <= 500_000:
        continue
    relative = path.relative_to(DOC)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # Rerunning a named experiment must not overwrite an older manifest hash.
    target = ARCHIVE / digest / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    compressed = None
    if path.suffix in (".csv", ".log", ".json") and not any(
        x in str(relative) for x in ("/.logs/", "profile_log_device", "tracy_ops_times")
    ):
        compressed = path.with_suffix(path.suffix + ".gz")
        for entry in entries:
            if entry.get("compressed") == str(compressed.relative_to(DOC)):
                entry["compressed"] = None
        with path.open("rb") as src, gzip.open(compressed, "wb") as dst:
            shutil.copyfileobj(src, dst)
        if compressed.stat().st_size > 500_000:
            shutil.move(compressed, str(target) + ".gz")
            compressed = None
    entries.append(
        {
            "original": str(relative),
            "archive": str(target),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "compressed": str(compressed.relative_to(DOC)) if compressed else None,
        }
    )
    path.unlink()
manifest.write_text(json.dumps(entries, indent=2) + "\n")
