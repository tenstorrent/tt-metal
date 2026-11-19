# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-export valid operation zones, then perform the ordinary Tracy device join.

Generic kernel source contains unescaped multiline CSV fields in the all-zone
export. Preserve that original and re-export TT_ operation zones from the same
binary. All operation messages and device timings remain the original capture.
Use --capture followed by ordinary Tracy arguments, or --folder for offline
recovery of a completed capture. No trace/device timing is synthesized.
"""

import argparse
import json
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests.profile_trace_capture import (
    _CSVEXPORT_EXECUTABLE,
    export_operation_times,
)


def process(output_folder, name_append, date, **options):
    from tracy.common import generate_logs_folder, generate_reports_folder
    from tracy.process_ops_logs import append_device_data, generate_reports, import_tracy_op_logs

    if options.get("device_only") or options.get("analyze_noc_traces"):
        raise ValueError("This adapter requires ordinary host/device operation profiling")
    root = Path(output_folder)
    logs = generate_logs_folder(root)
    operation_logs, command = export_operation_times(logs, _CSVEXPORT_EXECUTABLE)
    ops, signposts, replays = import_tracy_op_logs(operation_logs)
    if not ops or not signposts or replays:
        raise ValueError("Expected eager operation records and signposts, without trace replay")
    device_ops, trace_ops = append_device_data(
        ops,
        replays,
        logs,
        False,
        options.get("device_analysis_types", ()),
        force_legacy_device_logs=options.get("force_legacy_device_logs", False),
    )
    generate_reports(ops, device_ops, trace_ops, signposts, logs, generate_reports_folder(root), date, name_append)
    (root / "operation_export_provenance.json").write_text(
        json.dumps(
            dict(
                command=command,
                source_capture=str(logs.resolve()),
                operation_records=len(ops),
                device_operation_records=len(device_ops),
                signposts=signposts,
                scope="Original host operation messages joined to original per-op device timings; child zones excluded",
                raw_all_zone_export_preserved=True,
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    if sys.argv[1:2] == ["--capture"]:
        import tracy
        from tracy.__main__ import main as capture

        del sys.argv[1]
        tracy.process_ops = process
        capture()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    process(args.folder, args.name, False)


if __name__ == "__main__":
    main()
