# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize whole-layer real-weight precision/geometry controls."""

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--length", type=int, default=33)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--weights", nargs="+", default=["bfloat8_b", "bfloat4_b"])
    p.add_argument("--grids", nargs="+", default=["11x1", "11x2", "11x4"])
    p.add_argument("--blocks", type=int, nargs="+", default=[2, 11, 22])
    p.add_argument("--fidelities", nargs="+", default=["LoFi", "HiFi2"])
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    journal = []
    for weight, grid, block, fidelity in itertools.product(args.weights, args.grids, args.blocks, args.fidelities):
        name = f"layer{args.layer}_{weight}_{grid}_k{block}_{fidelity}"
        output = args.output_dir / (name + ".json")
        if output.exists():
            raise RuntimeError(f"Refusing to overwrite {output}")
        command = [
            sys.executable,
            "-m",
            "models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_decoder",
            "--layer",
            str(args.layer),
            "--length",
            str(args.length),
            "--steps",
            str(args.steps),
            "--real",
            "--decode",
            "--timing",
            "--expert-gate-dtype",
            weight,
            "--expert-down-dtype",
            "bfloat8_b",
            "--expert-grid",
            *grid.split("x"),
            "--expert-block-w",
            str(block),
            "--expert-fidelity",
            fidelity,
            "--output",
            str(output),
        ]
        entry = dict(
            command=command,
            source_sha256=hashlib.sha256(
                Path(__file__).parents[1].joinpath("tt/optimized_decoder.py").read_bytes()
            ).hexdigest(),
        )
        journal.append(entry)
        (args.output_dir / "commands.json").write_text(json.dumps(journal, indent=2) + "\n")
        with output.with_suffix(".log").open("w") as log:
            entry["exit_code"] = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        (args.output_dir / "commands.json").write_text(json.dumps(journal, indent=2) + "\n")
        print(name, entry["exit_code"], flush=True)
        if not output.exists():
            raise RuntimeError("Runtime failed before evidence; investigate before next hardware command")


if __name__ == "__main__":
    main()
