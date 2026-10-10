# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recover a bounded full-model report from an existing complete Tracy capture.

The standard exporter emits every CPU zone before filtering in pandas. Export
TT_ zones directly instead. Device timing, capture metadata, signposts and all
messages are retained. Optional host child-function timing is intentionally
omitted; no hardware is opened and the original capture is not changed.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save
from models.demos.qwen38_27b_qb2.tests.full_trace_profile import collect


def run(args):
    args.output.mkdir()
    state = dict(state="exporting", physical_devices_accessed=False, started_at=time.time(), files={})
    status_path = args.output / "queue.json"
    save(status_path, state)
    try:
        original_logs = args.original / "tracy/.logs"
        logs = args.output / "tracy/.logs"
        logs.mkdir(parents=True)
        for name in ("profile.json", "hardware.xml"):
            shutil.copy2(args.original / name, args.output / name)
        for name in ("tracy_profile_log_host.tracy", "cpp_device_perf_report.csv"):
            source = original_logs / name
            state["files"][name] = dict(
                bytes=source.stat().st_size, sha256=hashlib.sha256(source.read_bytes()).hexdigest()
            )
            (logs / name).symlink_to(source)
        for name in ("zone_src_locations.log", "new_zone_src_locations.log"):
            if (original_logs / name).exists():
                shutil.copy2(original_logs / name, logs / name)
        save(status_path, state)
        trace = str(logs / "tracy_profile_log_host.tracy")
        for name, flags in (
            ("tracy_ops_times.csv", ["-u", "-f", "TT_", "-t", "TT_"]),
            ("tracy_ops_data.csv", ["-m", "-s", ";"]),
        ):
            with (logs / name).open("w") as output:
                subprocess.run([str(args.exporter), *flags, trace], stdout=output, check=True, timeout=900)
            if (logs / name).stat().st_size > 2 * 1024**3:
                raise ValueError("Filtered export unexpectedly exceeds 2 GiB")
            state["files"][name] = dict(bytes=(logs / name).stat().st_size)
            save(status_path, state)
        os.environ["TT_METAL_PROFILER_DIR"] = str(args.output / "tracy")
        from tracy.process_ops_logs import process_ops

        state["state"] = "processing"
        save(status_path, state)
        process_ops(args.output / "tracy", None, True)
        analysis = collect(args.output)
        original = json.loads((args.output / "profile.json").read_text())
        baseline = json.loads(args.baseline.read_text())
        if (
            original["output_hashes"] != baseline["output_hashes"]
            or original["token_hashes"] != baseline["token_hashes"]
        ):
            raise ValueError("Profiled/unprofiled outputs differ")
        state.update(
            state="completed",
            full_trace_reconciliation_passed=analysis["full_trace_reconciliation_passed"],
            comparisons=analysis["comparisons"],
            profiled_unprofiled_outputs_match=True,
            baseline_replays=baseline["replays"],
            profiled_replays=original["replays"],
            finished_at=time.time(),
        )
    except BaseException as error:
        state.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        save(status_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "output", "exporter", "baseline"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())
