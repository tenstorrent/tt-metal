# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial advertised-context controls with full-model memory reservation."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for layer in (0, 3):
    for length in (262143, 262144):
        for baseline in (True, False):
            name = f"capacity_l{layer}_s{length}" + ("_baseline" if baseline else "")
            subprocess.run(
                [
                    "bash",
                    str(ROOT / "tests/run_multichip_experiment.sh"),
                    name,
                    "--layer",
                    str(layer),
                    "--length",
                    str(length),
                    "--capacity",
                    "--repeats",
                    "5",
                    *(["--prefill-only"] if length == 262144 else []),
                    *(["--baseline"] if baseline else []),
                ],
                check=True,
            )
            print(name, "PASS", flush=True)

# Keep the original full-length input live while the second layer consumes a
# different tensor, validating the conservative full-model activation lifetime.
for baseline in (True, False):
    name = "capacity_stack_s262143" + ("_baseline" if baseline else "")
    subprocess.run(
        [
            "bash",
            str(ROOT / "tests/run_multichip_experiment.sh"),
            name,
            "--layer",
            "0",
            "--length",
            "262143",
            "--stack",
            "--capacity",
            "--repeats",
            "5",
            *(["--baseline"] if baseline else []),
        ],
        check=True,
    )
    print(name, "PASS", flush=True)
