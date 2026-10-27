# SPDX-License-Identifier: Apache-2.0
"""CPU-only report extraction. Run in the environment containing tt-perf-report."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def within_window_report_rows(window, per_device):
    """Exclude only predecessor edges outside this window; preserve raw rows."""
    derived, excluded = [], []
    fields = ("OP TO OP LATENCY [ns]", "OP TO OP LATENCY BR/NRISC START [ns]")
    for row in window:
        copy = dict(row)
        device = row["DEVICE ID"]
        if int(row["DEVICE FW START CYCLE"]) == per_device[device]["first_cycle"]:
            excluded.append(
                dict(
                    device_id=device,
                    global_call_count=row["GLOBAL CALL COUNT"],
                    original_boundary_gaps={key: row[key] for key in fields if key in row},
                )
            )
            for key in fields:
                if key in copy:
                    copy[key] = ""
        derived.append(copy)
    assert len(excluded) == len(per_device), "Expected one first firmware op per ASIC"
    return derived, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--decode-only", action="store_true")
    args = parser.parse_args()
    root = args.directory
    source = sorted(root.glob("reports/**/ops_perf_results*.csv"))[-1]
    copied = root / "ops.csv"
    shutil.copyfile(source, copied)
    rows = list(csv.DictReader(copied.open()))
    with (root / ".logs/profile_log_device.csv").open() as f:
        header = next(f).strip()
    mhz = float(re.search(r"CHIP_FREQ\[MHz\]: (\d+)", header)[1])
    metrics = {
        "source": str(source),
        "device_header": header,
        "clock_mhz": mhz,
        "time_basis": "Maximum complete per-device whole-layer firmware envelope, all ops and gaps included. Per-chip cycles / MHz = us; never subtract unsynchronized clocks across ASICs.",
        "ops_csv_sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
        "phases": {},
    }
    for phase in ("DECODE",) if args.decode_only else ("PREFILL", "DECODE"):
        start = next(i for i, r in enumerate(rows) if r["OP TYPE"] == "signpost" and r["OP CODE"] == f"PERF_{phase}")
        end = next(
            i
            for i, r in enumerate(rows)
            if i > start and r["OP TYPE"] == "signpost" and r["OP CODE"] == f"PERF_{phase}_END"
        )
        window = [r for r in rows[start + 1 : end] if r["OP TYPE"] == "tt_dnn_device"]
        assert window and all(r["DEVICE FW START CYCLE"] and r["DEVICE FW END CYCLE"] for r in window)
        phase_csv = root / f"{phase.lower()}_ops.csv"
        with phase_csv.open("w") as compact:
            writer = csv.DictWriter(compact, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(window)
        per_device = {}
        for device in sorted(set(r["DEVICE ID"] for r in window)):
            local = [r for r in window if r["DEVICE ID"] == device]
            first_local = min(int(r["DEVICE FW START CYCLE"]) for r in local)
            last_local = max(int(r["DEVICE FW END CYCLE"]) for r in local)
            per_device[device] = {
                "first_cycle": first_local,
                "last_cycle": last_local,
                "device_us": (last_local - first_local) / mhz,
            }
        derived, excluded = within_window_report_rows(window, per_device)
        report_csv = root / f"{phase.lower()}_within_window_ops.csv"
        with report_csv.open("w") as compact:
            writer = csv.DictWriter(compact, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(derived)
        metrics["phases"][phase.lower()] = dict(
            device_us=max(v["device_us"] for v in per_device.values()),
            per_device=per_device,
            device_ops=len(window),
            summed_kernel_us=sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in window) / 1000,
            trace_ids=sorted(set(r["METAL TRACE ID"] for r in window)),
            trace_replay_sessions=sorted(set(r["METAL TRACE REPLAY SESSION ID"] for r in window)),
            window_ops_csv_sha256=hashlib.sha256(phase_csv.read_bytes()).hexdigest(),
            within_window_ops_csv_sha256=hashlib.sha256(report_csv.read_bytes()).hexdigest(),
            report_gap_scope="Within-window gap accounting only; first per-ASIC predecessor is outside the window. Raw timestamps/durations/rows unchanged; merged report is diagnostic, not max-chip wall time.",
            excluded_boundary_gaps=excluded,
        )
        executable = shutil.which("tt-perf-report") or str(Path(sys.executable).with_name("tt-perf-report"))
        common = [
            executable,
            str(report_csv),
            "--no-summary",
            "--no-color",
        ]
        with (root / f"{phase.lower()}_window_perf_report.console.log").open("w") as f:
            subprocess.run(
                common + ["--csv", str(root / f"{phase.lower()}_window_perf_report.csv")],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        with (root / f"{phase.lower()}_window_perf_report.txt").open("w") as f:
            subprocess.run(common, stdout=f, stderr=subprocess.STDOUT, check=True)
    (root / "whole_layer_timing.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)
    # Preserve raw evidence compressed after extraction; single-layer runs can
    # still produce GB-sized per-core event CSVs.
    for raw in root.rglob("*.csv"):
        if raw.stat().st_size > 64 * 1024**2:
            subprocess.run(["zstd", "-q", "-T2", "--rm", str(raw)], check=True)


if __name__ == "__main__":
    main()
