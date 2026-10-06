# SPDX-License-Identifier: Apache-2.0
"""Reconcile isolated DRAM-reader device timings with stored tile traffic."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("geometry", type=Path)
    args = parser.parse_args()
    source = max((args.directory / "raw/reports").glob("*/ops_perf_results_*.csv"))
    geometry = json.loads(args.geometry.read_text())
    measurements = {}
    active = None
    # Raw CSV already places trace replay rows between their replay signposts.
    # HOST START TS on those rows is the original capture timestamp.
    for row in csv.DictReader(source.open()):
        name = row["OP CODE"]
        if row["OP TYPE"] == "signpost":
            active = name if name.startswith("DENSE_") and not name.endswith("_END") else None
        elif active and row["OP CODE"] == "MatmulDeviceOperation" and row["DEVICE ID"] == "3":
            measurements.setdefault(active, []).append(row)
    output = []
    for config in geometry["rows"]:
        if "error" in config:
            continue
        prefix = f"DENSE_{config['role']}_C{config['cores']}_K{config['kblock']}_R{config['readers']}_ROUND"
        rounds = []
        for repeat in range(3):
            rows = measurements[prefix + str(repeat)]
            assert len(rows) == geometry["repetitions"], (prefix, len(rows))
            assert all(r["INPUT_1_DATATYPE"] == "BFLOAT4_B" and r["MATH FIDELITY"] == "LoFi" for r in rows)

            def average(column):
                return statistics.mean(float(r[column] or 0) for r in rows)

            rounds.append(
                dict(
                    kernel_ns=average("DEVICE KERNEL DURATION [ns]"),
                    brisc_ns=average("DEVICE BRISC KERNEL DURATION [ns]"),
                    ncrisc_ns=average("DEVICE NCRISC KERNEL DURATION [ns]"),
                    math_ns=average("DEVICE TRISC1 KERNEL DURATION [ns]"),
                )
            )
        ns = statistics.median(r["kernel_ns"] for r in rounds)
        row = dict(
            config,
            device_rounds=rounds,
            device_median_ns=ns,
            physical_gb_s=config["stored_weight_bytes"] / ns,
            logical_gb_s=config["logical_weight_bytes"] / ns,
            declared_peak_gb_s=512,
            percent_peak=100 * config["stored_weight_bytes"] / ns / 512,
        )
        output.append(row)
    result = dict(
        source=str(source),
        geometry=str(args.geometry),
        device=3,
        mesh=[1, 4],
        note="Weight-only effective bandwidth includes BFP4 tile headers; no fictional extra DRAM channels.",
        rows=output,
    )
    (args.directory / "reader_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    columns = [
        "role",
        "cores",
        "kblock",
        "readers",
        "bank_n_tiles",
        "reader_row_bytes",
        "device_median_ns",
        "physical_gb_s",
        "logical_gb_s",
        "percent_peak",
        "same_dtype_control_pcc",
    ]
    with (args.directory / "reader_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
