# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
from pathlib import Path

from .provenance import provenance as base_provenance


def provenance():
    result = base_provenance()
    root = Path(__file__).resolve().parents[1]
    snapshots = root / "doc/optimized_decoder/source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    for path in [root / "tt/optimized_decoder.py", *root.glob("tests/optimized*.py")]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result["source_sha256"][str(path.relative_to(root))] = digest
        dest = snapshots / (digest + ".py.txt")
        if not dest.exists():
            dest.write_bytes(path.read_bytes())
    result["environment"].update(
        {
            key: os.environ.get(key)
            for key in (
                "OPT_POLICY",
                "OPT_TAG",
                "OPT_REAL_INPUT",
                "OPT_PROJECTIONS",
                "OPT_LAYOUT",
                "OPT_PREFILL",
                "OPT_CACHE",
                "OPT_SPLIT",
                "OPT_RUNTIME",
                "OPT_WORKER_L1_SIZE",
                "OPT_PROFILE_DRAIN",
            )
        }
    )
    return result
