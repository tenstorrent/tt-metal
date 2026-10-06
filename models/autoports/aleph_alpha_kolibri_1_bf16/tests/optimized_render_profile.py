# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--repetitions", type=int, default=3)
    a = p.parse_args()
    source = max((a.directory / "raw/reports").glob("*/ops_perf_results_*.csv"), key=lambda x: x.parent.name)
    result = {}
    for mode in ("prefill", "decode"):
        target = a.directory / f"{mode}_ops.csv"
        shutil.copy2(source, target)
        args = [
            "tt-perf-report",
            str(target),
            "--start-signpost",
            f"PERF_{mode.upper()}",
            "--end-signpost",
            f"PERF_{mode.upper()}_END",
        ]
        if mode == "decode":
            args += ["--active-experts", "6"]
        for suffix, extra in [
            ("console.log", ["--csv", str(a.directory / f"{mode}_perf_report.csv")]),
            ("txt", ["--no-summary", "--no-color"]),
        ]:
            with (a.directory / f"{mode}_perf_report.{suffix}").open("w") as f:
                subprocess.run(args + extra, stdout=f, stderr=subprocess.STDOUT, check=True)
        rows = list(csv.DictReader((a.directory / f"{mode}_perf_report.csv").open()))
        result[mode] = dict(
            kernel_ms=sum(float(r["Device Time"] or 0) for r in rows) / a.repetitions / 1000,
            gap_ms=sum(float(r["Op-to-Op Gap"] or 0) for r in rows) / a.repetitions / 1000,
            ops=len(rows) / a.repetitions,
            commands=args,
        )
    (a.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
