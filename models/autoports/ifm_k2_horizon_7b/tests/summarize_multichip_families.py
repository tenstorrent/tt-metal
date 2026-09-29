"""Compact device-row and complete-MLP evidence from existing TP4 captures."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from .analyze_multichip_profile import tensor, window_rows


def summarize(directory):
    performance = json.loads((directory / "performance.json").read_text())
    accuracy = json.loads((directory / "accuracy.json").read_text())
    with Path(performance["source_csv"]).open() as stream:
        rows = list(csv.DictReader(stream))
    mhz = performance["clock_mhz"]
    mlp_times = []
    for step in range(accuracy["steps"]):
        mark = f"MULTICHIP_PERF_DECODE_{step:03d}"
        ops = window_rows(rows, mark, mark + "_END")
        spans = []
        for device in sorted({row["DEVICE ID"] for row in ops}):
            rank = [row for row in ops if row["DEVICE ID"] == device]
            norms = [i for i, row in enumerate(rank) if row["OP CODE"] == "LayerNormDeviceOperation"]
            assert len(norms) == 2
            selected = rank[norms[1] :]
            spans.append(
                (
                    max(int(row["DEVICE FW END CYCLE"]) for row in selected)
                    - min(int(row["DEVICE FW START CYCLE"]) for row in selected)
                )
                / mhz
            )
        mlp_times.append(max(spans))
    result = {
        "evidence": str(directory / "performance.json"),
        "command_args": accuracy["command_args"],
        "whole_layer_decode_device_us": performance["paths"]["multichip"]["decode_device_mean_us"],
        "whole_mlp_decode_device_us": statistics.mean(mlp_times),
        "mlp_scope": "Second RMS norm through final residual add; maximum per-rank FW span including gaps, averaged over all decode steps",
        "prefill_pcc": accuracy["prefill_pcc"],
        "decode_pcc_min": min(accuracy["decode_pcc"]),
        "phases": {},
    }
    for phase, mark in (("prefill", "MULTICHIP_PERF_PREFILL"), ("decode", "MULTICHIP_PERF_DECODE_000")):
        selected = [row for row in window_rows(rows, mark, mark + "_END") if row["DEVICE ID"] == "0"]
        compact = []
        previous = None
        for row in selected:
            start, end = (int(row[key]) for key in ("DEVICE FW START CYCLE", "DEVICE FW END CYCLE"))
            compact.append(
                {
                    "op": row["OP CODE"],
                    "firmware_us": (end - start) / mhz,
                    "gap_before_us": (start - previous) / mhz if previous is not None else None,
                    "attributes": row["ATTRIBUTES"],
                    "ports": {
                        name: spec
                        for kind, count in (("INPUT", 6), ("OUTPUT", 3))
                        for i in range(count)
                        if (spec := tensor(row, name := f"{kind}_{i}")) is not None
                    },
                }
            )
            previous = end
        result["phases"][phase] = compact
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps({path.name: summarize(path) for path in args.directories}, indent=2) + "\n")


if __name__ == "__main__":
    main()
