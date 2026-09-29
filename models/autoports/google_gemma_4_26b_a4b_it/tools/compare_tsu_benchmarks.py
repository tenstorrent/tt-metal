# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare identical serving cohorts, retaining correctness and latency evidence."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def comparable_command(path):
    command = json.loads(path.read_text())
    command[command.index("--result-filename") + 1] = "<cohort-output>"
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    controls = json.loads((args.control / "summary.json").read_text())
    candidates = json.loads((args.candidate / "summary.json").read_text())
    assert len(controls) == len(candidates) and controls, "Incomplete cohort sets"
    report = {
        "scope": "Matched serving benchmark cohorts; TSU is 1000 / mean TPOT",
        "control": str(args.control),
        "candidate": str(args.candidate),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [],
        "pooled": [],
        "passed": False,
    }
    for control_row, candidate_row in zip(controls, candidates):
        assert (control_row["shape"], control_row["repeat"]) == (candidate_row["shape"], candidate_row["repeat"])
        paths = [root / row["source"] for root, row in ((args.control, control_row), (args.candidate, candidate_row))]
        control, candidate = (json.loads(path.read_text()) for path in paths)
        commands = [comparable_command(path.with_suffix(".command.json")) for path in paths]
        checks = {
            "identical_harness_command": commands[0] == commands[1],
            "complete": all(
                data["completed"] == control_row["shape"][3] and data["failed"] == 0 for data in (control, candidate)
            ),
        }
        for key in ("generated_texts", "input_lens", "output_lens", "model_id", "tokenizer_id"):
            checks[key + "_equal"] = control[key] == candidate[key]
        row = {"shape": control_row["shape"], "repeat": control_row["repeat"], "checks": checks}
        for label, data in (("control", control), ("candidate", candidate)):
            row[label] = {
                key: data[key]
                for key in ("mean_tpot_ms", "mean_ttft_ms", "mean_e2el_ms", "median_itl_ms", "output_throughput")
            }
            row[label]["tsu"] = 1000 / data["mean_tpot_ms"]
        report["rows"].append(row)
    for shape in dict.fromkeys(tuple(row["shape"]) for row in report["rows"]):
        rows = [row for row in report["rows"] if tuple(row["shape"]) == shape]
        pooled = {"shape": shape, "repeats": len(rows)}
        for label in ("control", "candidate"):
            pooled[label] = {
                key: mean(row[label][key] for row in rows)
                for key in ("mean_tpot_ms", "mean_ttft_ms", "mean_e2el_ms", "median_itl_ms")
            }
            pooled[label]["tsu"] = 1000 / pooled[label]["mean_tpot_ms"]
        pooled["tsu_gain_percent"] = 100 * (pooled["candidate"]["tsu"] / pooled["control"]["tsu"] - 1)
        report["pooled"].append(pooled)
    report["passed"] = all(all(row["checks"].values()) for row in report["rows"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["passed"], "Harness/correctness mismatch; inspect saved comparison"


if __name__ == "__main__":
    main()
