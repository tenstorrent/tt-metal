# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Normal Tracy capture with a trace-only host-metadata export adapter.

The normal per-op device join is incompatible with aggregate replay counters.
Keep the raw device CSV and exact host markers; report_trace_performance uses
the official trace analysis and emits an explicitly labeled whole-trace row.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_CSVEXPORT_EXECUTABLE = Path(__file__).resolve().parents[4] / "build/tools/profiler/bin/tracy-csvexport"


def export_operation_times(logs, executable):
    """Re-export only operation zones; retain the original all-zone CSV intact.

    The standard child-call export includes generic-kernel source as unescaped
    multiline src_file fields. No child timings are needed for trace selection.
    The official message parser still receives ALL original operation messages.
    """
    from tracy.common import TRACY_FILE_NAME, TRACY_OPS_DATA_FILE_NAME, TRACY_OPS_TIMES_FILE_NAME

    target = logs / "operation_only_export"
    target.mkdir(exist_ok=True)
    command = [str(executable), "-u", "-f", "TT_", str(logs / TRACY_FILE_NAME)]
    with (target / TRACY_OPS_TIMES_FILE_NAME).open("w") as stream:
        subprocess.run(command, stdout=stream, check=True)
    data = target / TRACY_OPS_DATA_FILE_NAME
    if not data.exists():
        data.symlink_to((logs / TRACY_OPS_DATA_FILE_NAME).resolve())
    return target, command


def process_trace_only(output_folder, name_append, date, **options):
    from tracy.common import generate_logs_folder, generate_reports_folder
    from tracy.process_ops_logs import generate_reports, import_tracy_op_logs

    if not output_folder or options.get("device_only") or options.get("analyze_noc_traces"):
        raise ValueError("Use a dedicated output folder and ordinary host metadata capture")
    logs = generate_logs_folder(Path(output_folder))
    operation_logs, export_command = export_operation_times(logs, _CSVEXPORT_EXECUTABLE)
    ops, signposts, replays = import_tracy_op_logs(operation_logs)
    if not ops or not signposts or not replays:
        raise ValueError("Missing host operations, measurement markers, or trace replay metadata")
    metadata = {
        "schema_version": 1,
        "operation_times_export": {
            "command": export_command,
            "scope": "Operation zones only; original complete messages retained; child-call timings not used",
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in operation_logs.glob("tracy_ops_*.csv")
            },
        },
        "trace_replays": replays,
        "signposts": signposts,
        "ops": [
            {
                key: op.get(key)
                for key in ("global_call_count", "op_type", "op_code", "device_id", "metal_trace_id", "tracy_time")
            }
            for op in ops.values()
        ],
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in logs.glob("tracy_ops_*.csv")},
    }
    (logs / "trace_host_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    generate_reports(ops, {}, {}, signposts, logs, generate_reports_folder(Path(output_folder)), date, name_append)


def main():
    global _CSVEXPORT_EXECUTABLE
    args = sys.argv[1:]
    if "-m" not in args:
        raise ValueError("Use -m with profile_trace_decoder")
    outer = args[: args.index("-m")]
    required = ("-r", "-o", "--device-trace-profiler", "--no-runtime-analysis", "--disable-device-data-push-to-tracy")
    if not all(flag in outer for flag in required) or "--no-device" in outer:
        raise ValueError(f"Required outer flags: {required}; --no-device is incompatible")
    if os.environ.get("TT_METAL_STREAMING_PROFILER") not in (None, "", "0"):
        raise ValueError("Unset streaming profiler; no unsupported firmware feature is enabled")
    if "--tracy-tools-folder" in outer:
        from tracy.common import TRACY_CSVEXPROT_TOOL, resolve_tracy_tool_path

        folder = Path(outer[outer.index("--tracy-tools-folder") + 1])
        _CSVEXPORT_EXECUTABLE = resolve_tracy_tool_path(folder, TRACY_CSVEXPROT_TOOL)
        if _CSVEXPORT_EXECUTABLE is None:
            raise ValueError("Tracy CSV exporter not found")
    os.environ["TT_METAL_PROFILER_CPP_POST_PROCESS"] = "0"
    import tracy
    from tracy.__main__ import main as tracy_main

    tracy.process_ops = process_trace_only
    tracy_main()


if __name__ == "__main__":
    main()
