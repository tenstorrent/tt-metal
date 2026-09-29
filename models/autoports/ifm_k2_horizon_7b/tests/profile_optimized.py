"""Capture the exact layer workload and retain whole-window device evidence."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parents[1] / "doc/optimized_decoder"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--label", default="final")
    a = parser.parse_args()
    doc = DOC / ("baseline" if a.baseline else a.label)
    raw = doc / "tracy/raw"
    dense = doc / "tracy/dense"
    raw.mkdir(parents=True, exist_ok=True)
    dense.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_DPRINT_CORES"):
        env.pop(key, None)
    env["TT_METAL_PROFILER_DIR"] = str(raw)
    cmd = [
        sys.executable,
        "-m",
        "tracy",
        "-r",
        "-p",
        "-v",
        "--check-exit-code",
        "--op-support-count",
        "50000",
        "--web-app-port",
        "55183",
        "-o",
        str(raw),
        "-m",
        "models.autoports.ifm_k2_horizon_7b.tests.run_optimized",
        "--lengths",
        "4096",
        "--decode-steps",
        "128",
        "--profile",
        "--output",
        str(doc / "profile_accuracy.json"),
    ]
    if a.baseline:
        cmd.append("--baseline-fused")
    provenance = {"capture_command": cmd, "reports": []}
    (doc / "profile_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    candidates = sorted(raw.glob("reports/*/ops_perf_results_*.csv"), key=lambda p: p.stat().st_mtime)
    source = candidates[-1]
    ops = dense / "ops.csv"
    shutil.copyfile(source, ops)
    for phase, start, end in [
        ("prefill", "PERF_PREFILL", "PERF_PREFILL_END"),
        ("decode", "PERF_DECODE_000", "PERF_DECODE_000_END"),
    ]:
        base = ["tt-perf-report", str(ops), "--start-signpost", start, "--end-signpost", end]
        for suffix, extra in [
            ("txt", ["--no-summary"]),
            ("console.log", ["--csv", str(dense / (phase + "_perf_report.csv"))]),
        ]:
            report = dense / (phase + "_perf_report." + suffix)
            with report.open("w") as f:
                subprocess.run(base + extra, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
            if suffix == "txt":
                report.write_text("\n".join(line.rstrip() for line in report.read_text().splitlines()).rstrip() + "\n")
            else:
                csv_report = dense / (phase + "_perf_report.csv")
                csv_report.write_bytes(csv_report.read_bytes().replace(b"\r\n", b"\n"))
            provenance["reports"].append({"argv": base + extra, "output": str(report)})
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "models/autoports/ifm_k2_horizon_7b/tests/analyze_optimized_profile.py"),
            str(ops),
            "--device-log",
            str(raw / ".logs/profile_log_device.csv"),
            "--output",
            str(doc / "performance.json"),
            *(["--baseline"] if a.baseline else []),
        ],
        cwd=ROOT,
        check=True,
    )
    paths = [
        ops,
        raw / ".logs/profile_log_device.csv",
        doc / "performance.json",
        doc / "profile_accuracy.json",
        ROOT / "models/autoports/ifm_k2_horizon_7b/tt" / ("fused_decoder.py" if a.baseline else "optimized_decoder.py"),
    ] + list(dense.glob("*perf_report*"))
    provenance["sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with ops.open() as f:
        rows = list(csv.DictReader(f))
    provenance["signposts"] = sum(r.get("OP TYPE") == "signpost" for r in rows)
    provenance[
        "text_normalization"
    ] = "Text-table trailing whitespace removed; CSV CRLF normalized to LF. Measurements unchanged."
    (doc / "profile_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
