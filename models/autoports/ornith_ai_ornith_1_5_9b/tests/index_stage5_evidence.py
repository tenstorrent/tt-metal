# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact index of completed serialized experiments, metrics and source provenance."""

import csv
import datetime
import json
import re
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    stage = root / "doc/optimized_multichip_decoder"
    rows = []
    jobs = [json.loads(line) for line in (stage / "device_jobs.jsonl").read_text().splitlines() if line]
    for job in jobs:
        row = {
            "label": job["label"], "start": job["start"], "end": job.get("end"),
            "exit_code": job["exit_code"], "experiment_seconds": job["seconds"],
            "source_sha256": job.get("source_sha256", {}).get("tt/multichip_decoder.py", job.get("implementation_sha256")),
            "command_provenance": job["label"] + "_command.json", "log": job["label"] + ".log",
            "regime": job["regime"], "devices": job["devices"],
        }
        command = job["command"]
        # Earlier generic launcher sidecars described the reserved mesh rather
        # than the actual CPU/reference path. Keep immutable sidecars and make
        # this command-derived correction explicit in the derived index.
        host_calibration = any("record_precision_inputs" in str(part) for part in command)
        row["devices"] = [] if host_calibration else [0] if "--baseline" in command else job["devices"]
        row["execution_kind"] = "CPU calibration" if host_calibration else "single-chip reference" if "--baseline" in command else "TP4 experiment"
        if "--output" in command:
            output = Path(command[command.index("--output") + 1])
            if not output.is_absolute():
                output = Path(job.get("cwd") or Path.cwd()) / output
            if output.exists() and output.suffix == ".json":
                data = json.loads(output.read_text())
                row["result"] = str(output)
                if isinstance(data, dict):
                    records = data.get("records", [])
                    pccs = [value for r in records for key, value in r.items() if key.endswith("pcc") and isinstance(value, (float, int))]
                    row.update(
                        min_pcc=min(pccs) if pccs else None, pcc_checks=len(pccs),
                        batch=data.get("batch", 1), layer=data.get("layer"), stack=data.get("stack"),
                        implementation_class=data.get("implementation_class"),
                        io_contract=data.get("io_contract", "per-request tensor list"),
                        trace_replays=data.get("trace_replays", data.get("steps")),
                        tracking=data.get("tracking"),
                        decode_stack_ms=data.get("decode_stack_ms"),
                        decode_layer_ms=data.get("decode_ms"),
                        profile_decode_host_ms=data.get("profile_decode_host_ms"),
                        per_length_timing=[r for r in records if "prefill_ms" in r],
                    )
        if job["exit_code"]:
            log = (stage / (job["label"] + ".log")).read_text(errors="replace")
            failures = re.findall(r"^(?:AssertionError|RuntimeError|TypeError|ValueError|TT_FATAL).*", log, re.M)
            row["failure"] = failures[-1][:600] if failures else "See retained log and recovery evidence"
        rows.append(row)
    summary = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "completed_jobs": len(rows),
        "serialized_experiment_seconds": sum(r["experiment_seconds"] for r in rows),
        "hardware_experiment_seconds": sum(r["experiment_seconds"] for r in rows if r["execution_kind"] != "CPU calibration"),
        "time_note": "Experiment elapsed includes setup, compilation, validation and profiler postprocessing; not summed device-kernel time. Recovery intervals are recorded separately. Stop-rule decision is documented in work_log.md.",
        "jobs": rows,
    }
    (stage / "evidence_index.json").write_text(json.dumps(summary, indent=2) + "\n")
    columns = ["label", "exit_code", "experiment_seconds", "source_sha256", "batch", "layer", "min_pcc", "pcc_checks", "decode_stack_ms", "decode_layer_ms", "io_contract", "command_provenance", "failure"]
    with (stage / "evidence_index.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("EVIDENCE_INDEX_PASS", len(rows), round(summary["serialized_experiment_seconds"], 1))


if __name__ == "__main__":
    main()
