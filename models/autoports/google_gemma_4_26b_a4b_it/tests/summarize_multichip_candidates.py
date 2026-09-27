# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare recorded paired 4096/128 layer runs without launching hardware."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or (data.get("length"), data.get("steps")) != (4096, 128):
            continue
        timings = data.get("timings", {})
        if data.get("passed") is not True or set(timings) != {"1", "4"} or len(data.get("pcc", [])) != 129:
            continue
        command = data.get("command", [])
        if "--probe" in command and command[command.index("--probe") + 1] != "none":
            continue
        if any(len(timings[rank]["decode_host_us"]) != 128 for rank in ("1", "4")):
            raise ValueError(f"Incomplete decode timing: {path}")
        row = dict(
            artifact=path.name,
            harness=Path(command[0]).name if command else None,
            layer_type=data["layer_type"],
            runtime_sha256=data.get("runtime_sha256"),
            runner_sha256=data.get("runner_sha256"),
            min_output_pcc=min(data["pcc"]),
            prefill_pcc=data["pcc"][0],
            min_decode_pcc=min(data["pcc"][1:]),
            min_cache_pcc=min(data["cache_pcc"]) if data.get("cache_pcc") else None,
            trace=data.get("trace"),
        )
        for phase in ("prefill", "decode"):
            values = [statistics.median(timings[rank][f"{phase}_host_us"]) for rank in ("1", "4")]
            if not all(value > 0 for value in values):
                raise ValueError(f"Nonpositive timing: {path}")
            row[f"{phase}_tp1_host_us"], row[f"{phase}_tp4_host_us"] = values
            row[f"{phase}_host_speedup"] = values[0] / values[1]
            row[f"{phase}_four_asic_efficiency"] = values[0] / values[1] / 4
        rows.append(row)
    if not rows:
        raise ValueError("No complete paired target-workload records")
    result = dict(
        status="candidate comparisons; stage acceptance is separate",
        workload=dict(input_tokens=4096, output_tokens=128, batch=1, concurrency=1),
        basis="Paired warmed host medians, not device durations. Different source hashes remain distinct candidates.",
        limitation="An individually passing run does not clear failures in other runs, contexts, or validation gates.",
        rows=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summarized {len(rows)} recorded candidates; no hardware execution")


if __name__ == "__main__":
    main()
