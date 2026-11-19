# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Account for serialized experiment time without calling host time kernel time."""

import datetime
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"


def seconds_union(rows):
    spans = sorted(
        (datetime.datetime.fromisoformat(r["start"]).timestamp(),
         datetime.datetime.fromisoformat(r["end"]).timestamp())
        for r in rows
    )
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def main():
    jobs = [json.loads(line) for line in (D / "device_jobs.jsonl").read_text().splitlines()]
    recoveries = [json.loads(line) for line in (D / "recovery_jobs.jsonl").read_text().splitlines()]
    # Some early sidecars describe the reserved mesh even for CPU calibration.
    host = [row for row in jobs if any("record_precision_inputs" in str(arg) for arg in row["command"])]
    host_labels = {row["label"] for row in host}
    hardware = [row for row in jobs if row.get("devices") and row["label"] not in host_labels]
    profiles = [row for row in hardware if any("tracy" in str(arg) for arg in row["command"])]
    first = datetime.datetime.fromisoformat(min(row["start"] for row in jobs))
    last = datetime.datetime.fromisoformat(max(row["end"] for row in jobs + recoveries))
    report = {
        "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "first_recorded_job": first.isoformat(),
        "last_completed_job_or_recovery": last.isoformat(),
        "elapsed_window_hours": (last - first).total_seconds() / 3600,
        "completed_jobs": len(jobs),
        "hardware_job_elapsed_hours": seconds_union(hardware) / 3600,
        "host_calibration_elapsed_hours": seconds_union(host) / 3600,
        "recovery_health_elapsed_hours": seconds_union(recoveries) / 3600,
        "hardware_and_recovery_union_hours": seconds_union(hardware + recoveries) / 3600,
        "profile_job_elapsed_hours_included_above": seconds_union(profiles) / 3600,
        "profile_jobs": [row["label"] for row in profiles],
        "notes": [
            "Elapsed intervals cover setup, compilation, execution, validation and teardown; these are not kernel execution hours.",
            "Profiler jobs also include capture serialization and host postprocessing; that subset is shown separately.",
            "Recovery includes serialized health checks and reset work. Interval union prevents overlap being counted twice.",
            "Unrecorded early recovery work and active unfinished jobs are excluded.",
            "The first candidate measurement was 2026-11-12 20:52:36 UTC; initial baseline jobs may precede it.",
        ],
    }
    candidate_start = datetime.datetime(2026, 11, 12, 20, 52, 36, tzinfo=datetime.timezone.utc)
    eligible = []
    for row in hardware + recoveries:
        if datetime.datetime.fromisoformat(row["end"]) > candidate_start:
            row = dict(row)
            row["start"] = max(datetime.datetime.fromisoformat(row["start"]), candidate_start).isoformat()
            eligible.append(row)
    report["from_first_candidate_hardware_and_recovery_union_hours"] = seconds_union(eligible) / 3600
    report["from_first_candidate_elapsed_window_hours"] = (last - candidate_start).total_seconds() / 3600
    (D / "device_time_accounting.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"notes", "profile_jobs"}}, indent=2))


if __name__ == "__main__":
    main()
