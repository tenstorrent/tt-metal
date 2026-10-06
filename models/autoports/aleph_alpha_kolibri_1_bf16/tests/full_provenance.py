# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path


def provenance():
    root = Path(__file__).resolve().parents[1]
    snapshots = Path(os.environ.get("FULL_ARTIFACT_DIR", root / "doc/full_model")) / "source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    sources = {}
    for path in sorted((root / "tt").glob("*.py")):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        target = snapshots / (digest + ".py.txt")
        if not target.exists():
            target.write_bytes(data)
        sources[str(path.relative_to(root))] = digest
    return dict(
        command=[sys.executable, *sys.argv],
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        time=time.time(),
        source_sha256=sources,
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("TT_METAL_TRACE_ALLOC_", "TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"))
        },
    )
