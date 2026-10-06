# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def provenance():
    root = Path(__file__).resolve().parents[1]
    names = [
        "tt/functional_decoder.py",
        "tests/reference.py",
        "tests/config.json",
        "tests/run_coverage.py",
        "tests/run_decoder.py",
        "tests/run_context.py",
        "tests/run_profile.py",
    ]
    return dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv,
        source_sha256={n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names},
        environment={
            n: os.environ.get(n)
            for n in [
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_WATCHER",
                "TT_METAL_LOGS_PATH",
                "TT_METAL_DEVICE_PROFILER",
                "HF_HOME",
            ]
        },
    )
