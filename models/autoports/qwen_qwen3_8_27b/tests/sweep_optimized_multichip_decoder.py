# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize a recorded candidate matrix; stop at the first failure for diagnosis."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    doc = root / "doc/optimized_multichip_decoder"
    for row in json.loads(args.matrix.read_text()):
        name = row["name"]
        policy = doc / (name + "_policy.json")
        policy.write_text(json.dumps(row.get("policy", {}), indent=2) + "\n")
        command = [
            "bash",
            str(
                root
                / "tests"
                / (
                    "profile_optimized_multichip_decoder.sh"
                    if args.profile
                    else "run_optimized_multichip_experiment.sh"
                )
            ),
            name,
            "--policy-file",
            str(policy),
            "--layer",
            str(row.get("layer", 0)),
            "--length",
            str(row.get("length", 128)),
            "--batch",
            str(row.get("batch", 1)),
            "--repeats",
            str(row.get("repeats", 30)),
            "--prefill-repeats",
            str(row.get("prefill_repeats", 10)),
            *row.get("args", []),
        ]
        print("RUN", name, flush=True)
        subprocess.run(command, check=True)
        report = json.loads((doc / (name + ".json")).read_text())
        print(
            "PASS",
            name,
            report.get("prefill_ms"),
            report.get("decode_ms"),
            min(report.get("pcc", {"missing": 0}).values()),
            flush=True,
        )


if __name__ == "__main__":
    main()
