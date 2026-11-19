# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run normal Tracy capture/export, retaining host-only operation metadata.

Use the same arguments as ``python -m tracy``, including ``-r --no-device``.
The standard postprocessor unconditionally attempts a classic device-log join
even with --no-device. This test-only outer-process adapter instead calls its
existing host CSV generator; report_streaming_performance joins measured
streaming durations afterwards. No TT runtime implementation is changed.
"""

import os
import sys
from pathlib import Path


def process_host_only(output_folder, name_append, date, **options):
    from tracy.common import generate_logs_folder, generate_reports_folder
    from tracy.process_ops_logs import generate_reports, import_tracy_op_logs

    if not output_folder or options.get("device_only") or options.get("analyze_noc_traces"):
        raise ValueError("Streaming host export requires a dedicated output folder and host metadata only")
    logs = generate_logs_folder(Path(output_folder))
    reports = generate_reports_folder(Path(output_folder))
    ops, signposts, trace_replays = import_tracy_op_logs(logs)
    if not ops or not signposts:
        raise ValueError("Missing captured host operations/signposts")
    if trace_replays:
        raise ValueError("Unexpected standard device-profiler trace metadata in streaming capture")
    generate_reports(ops, {}, {}, signposts, logs, reports, date, name_append)


def main():
    # These are outer Tracy flags. The model's arguments follow its module name.
    arguments = sys.argv[1:]
    if "-m" not in arguments:
        raise ValueError("Use -m with the model-local profile_decoder module")
    outer = arguments[: arguments.index("-m")]
    if not all(flag in outer for flag in ("-r", "--no-device", "-o")):
        raise ValueError("Required outer capture flags: -r --no-device -o <dedicated-directory>")
    if os.environ.get("TT_METAL_STREAMING_PROFILER") != "1" or os.environ.get("TT_METAL_DEVICE_PROFILER"):
        raise ValueError("Set TT_METAL_STREAMING_PROFILER=1 and unset TT_METAL_DEVICE_PROFILER")
    import tracy
    from tracy.__main__ import main as tracy_main

    # generate_report resolves this global in tracy, after capture and CSV export.
    # The child still runs the unmodified `python -m tracy --no-capture-tool`.
    tracy.process_ops = process_host_only
    tracy_main()


if __name__ == "__main__":
    main()
