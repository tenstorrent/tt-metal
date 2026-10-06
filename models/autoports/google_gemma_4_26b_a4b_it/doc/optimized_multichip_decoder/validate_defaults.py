# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Final default correctness, capacity and separate Watcher runs, serialized."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--only", nargs="+", help="Run only the named validation cases")
selected_cases = parser.parse_args().only
root = Path(__file__).resolve().parent
module = "models.autoports.google_gemma_4_26b_a4b_it.tests."
cases = []
for layer, kind in ((0, "sliding"), (5, "full")):
    base = ["--layer", str(layer), "--trace", "--check-cache"]
    cases.append(
        (
            "final_" + kind,
            "run_multichip_decoder",
            base + ["--length", "4096", "--steps", "128", "--duplicate-replays", "8"],
            {},
        )
    )
for layer, kind in ((0, "sliding"), (5, "full")):
    cases.append(("final_contracts_" + kind, "test_multichip_contracts", ["--layer", str(layer), "--batch", "32"], {}))
for pair, label in (([0, 1], "samekind"), ([4, 5], "mixed")):
    for length in (33, 4096):
        cases.append(
            (
                f"final_stack_{label}_{length}",
                "test_multichip_stack",
                ["--layers", *map(str, pair), "--length", str(length), "--steps", "128"],
                {},
            )
        )
for layer, kind in ((0, "sliding"), (5, "full")):
    cases.append(
        (
            "final_nonaligned_" + kind,
            "run_multichip_decoder",
            ["--layer", str(layer), "--trace", "--check-cache", "--length", "4095", "--steps", "128"],
            {},
        )
    )
    for length, steps in ((262143, 1), (262144, 0)):
        cases.append(
            (
                f"final_max_{length}_{kind}",
                "run_multichip_decoder",
                [
                    "--layer",
                    str(layer),
                    "--length",
                    str(length),
                    "--steps",
                    str(steps),
                    "--trace",
                    "--check-cache",
                    "--repeat-input",
                    "--reserve-full-stack",
                    "--prefill-timing-samples",
                    "1",
                ],
                {},
            )
        )
for layer, kind in ((0, "sliding"), (5, "full")):
    cases.append(
        (
            "final_watcher_" + kind,
            "run_multichip_decoder",
            [
                "--layer",
                str(layer),
                "--length",
                "4096",
                "--steps",
                "128",
                "--trace",
                "--check-cache",
                "--duplicate-replays",
                "8",
            ],
            {"TT_METAL_WATCHER": "10", "TT_METAL_WATCHER_NOINLINE": "1"},
        )
    )
for pair, label in (([0, 1], "samekind"), ([4, 5], "mixed")):
    cases.append(
        (
            f"final_watcher_stack_{label}",
            "test_multichip_stack",
            ["--layers", *map(str, pair), "--length", "4096", "--steps", "128"],
            {"TT_METAL_WATCHER": "10", "TT_METAL_WATCHER_NOINLINE": "1"},
        )
    )
for name, runner, flags, env in cases:
    if selected_cases and name not in selected_cases:
        continue
    record = root / (name + ".command.json")
    sources = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            root.parents[1] / "tt/multichip_decoder.py",
            root.parents[1] / "tests" / (runner + ".py"),
            Path(
                "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp"
            ),
            Path("ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp"),
        )
    }
    if record.exists():
        previous = json.loads(record.read_text())
        if previous["exit_code"] == 0 and previous.get("source_sha256") == sources:
            continue
    command = [sys.executable, "-m", module + runner] + flags + ["--output", str(root / (name + ".json"))]
    print("START", name, flush=True)
    with (root / (name + ".log")).open("w") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", **env},
            timeout=1800,
        )
    record.write_text(
        json.dumps(dict(command=command, environment=env, exit_code=result.returncode, source_sha256=sources), indent=2)
        + "\n"
    )
    print("END", name, result.returncode, flush=True)
    if result.returncode:
        break
