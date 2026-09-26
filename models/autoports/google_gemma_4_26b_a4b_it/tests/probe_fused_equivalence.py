# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Paired real-weight functional/fused outputs on the complete headline workload."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from models.common.utility_functions import comp_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commands, outputs = [], {}
    for decoder in ("functional", "fused"):
        stem = args.output.with_name(args.output.stem + "_" + decoder)
        tensors = stem.with_suffix(".pt")
        command = [
            sys.executable,
            "-m",
            "models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder",
            "--decoder",
            decoder,
            "--layer",
            str(args.layer),
            "--length",
            "4096",
            "--real",
            "--decode",
            "--steps",
            "128",
            "--verify-program-cache",
            "--timing",
            "--output",
            str(stem.with_suffix(".json")),
            "--save-output-tensors",
            str(tensors),
        ]
        with stem.with_suffix(".log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        commands.append(dict(command=command, exit_code=completed.returncode))
        assert completed.returncode == 0, commands[-1]
        outputs[decoder] = torch.load(tensors, weights_only=True)
    checks = []
    for phase in ("prefill", "decode"):
        a, b = outputs["functional"][phase], outputs["fused"][phase]
        if phase == "prefill":
            a, b = [a], [b]
        assert len(a) == len(b) == (1 if phase == "prefill" else 128)
        for index, (baseline, fused) in enumerate(zip(a, b)):
            passed, pcc = comp_pcc(baseline, fused, 0.995)
            checks.append(dict(phase=phase, sample=index, pcc=float(pcc), passed=bool(passed)))
    result = dict(
        layer=args.layer,
        real_weights=True,
        workload=dict(input_tokens=4096, output_tokens=128, batch=1, concurrency=1),
        commands=commands,
        checks=checks,
        passed=all(row["passed"] for row in checks),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["passed"], result


if __name__ == "__main__":
    main()
