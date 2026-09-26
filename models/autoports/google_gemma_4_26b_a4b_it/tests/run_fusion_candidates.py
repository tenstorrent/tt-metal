# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized real-weight A/B candidates on the exact headline workload."""

import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder")
    base = "broadcast_pack_gelu_router_tail"
    variants = ["attention", "norm", "rope"]
    results = []
    for variant in variants:
        name = f"isolated_{variant}_sliding"
        output = root / (name + ".json")
        command = [
            sys.executable,
            "-m",
            "models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder",
            "--decoder",
            "fused",
            "--fusion",
            base + "_" + variant,
            "--group-size",
            "16384",
            "--layer",
            "0",
            "--length",
            "4096",
            "--real",
            "--decode",
            "--steps",
            "128",
            "--timing",
            "--verify-program-cache",
            "--output",
            str(output),
        ]
        with output.with_suffix(".log").open("w") as log:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        results.append(dict(command=command, exit_code=rc, output=str(output)))
        (root / "isolated_candidate_commands.json").write_text(json.dumps(results, indent=2) + "\n")
        print(name, rc, flush=True)


if __name__ == "__main__":
    main()
