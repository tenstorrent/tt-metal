"""Retained watcher checks, runtime fallback rejection and post-close observation."""

import argparse
import json
import os
import re
import runpy
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("models/autoports/ifm_k2_horizon_7b/doc/full_model"))
    args = parser.parse_args()
    assert os.environ.get("TT_METAL_WATCHER")
    assert not any(k.startswith("TT_METAL_WATCHER_DISABLE") for k in os.environ)
    assert not os.environ.get("TT_METAL_DEVICE_PROFILER")
    import ttnn

    ttnn.CONFIG.throw_exception_on_fallback = True
    doc = args.output_dir
    doc.mkdir(parents=True, exist_ok=True)
    sys.argv = ["probe_full_trace", "--layers", "2", "--output", str(doc / "watcher_trace.json")]
    runpy.run_module("models.autoports.ifm_k2_horizon_7b.tests.probe_full_trace", run_name="__main__")
    log = Path("generated/watcher/watcher.log")

    def completed():
        return len(re.findall(r"Dump #\d+ completed", log.read_text()))

    before = completed()
    deadline = time.monotonic() + 45
    while completed() < before + 2:
        if time.monotonic() >= deadline:
            raise TimeoutError("No two watcher polls after mesh close")
        time.sleep(0.1)
    result = {
        "layers": 2,
        "throw_exception_on_fallback": ttnn.CONFIG.throw_exception_on_fallback,
        "post_close_completed_polls": completed() - before,
        "environment": {k: v for k, v in os.environ.items() if k.startswith("TT_METAL_WATCHER")},
        "all_checks_retained": True,
    }
    (doc / "watcher.json").write_text(json.dumps(result, indent=2) + "\n")
    print("POST_CLOSE_WATCHER_POLLS_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
