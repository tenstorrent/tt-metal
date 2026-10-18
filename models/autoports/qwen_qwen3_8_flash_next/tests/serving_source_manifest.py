# SPDX-License-Identifier: Apache-2.0
"""Record source provenance without importing TTNN or opening devices."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    model = Path(__file__).resolve().parents[1]
    plugin = root.parent / "vllm-tt-plugin"
    sources = list((model / "tt").rglob("*.py")) + list((model / "tt").rglob("*.cpp"))
    sources += list((plugin / "src").rglob("*.py"))
    package = Path(os.environ["TT_MODEL_BRINGUP_ROOT"])
    sources.append(package / "runtime/readiness_check/run_vllm_server.py")
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        interpreter=sys.executable,
        repos={
            str(repo): dict(
                commit=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                status=subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True),
            )
            for repo in (root, plugin)
        },
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)},
        environment={
            key: os.environ.get(key)
            for key in (
                "QWEN_FLASH_MAX_SEQS",
                "QWEN_FLASH_ALLOW_HOST_SAMPLING",
                "QWEN_FLASH_SERVING_LAYERS",
                "QWEN_FLASH_SYNC_CONTROL",
                "QWEN_FLASH_AUDIT_POLICY",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "QWEN_FLASH_TRACE_EVENTS",
                "TT_METAL_DEVICE_PROFILER",
            )
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
