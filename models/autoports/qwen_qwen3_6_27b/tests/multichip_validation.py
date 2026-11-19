# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serial context, stack and sustained-decode gates; stop on the first failure."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["context-smoke", "context", "stack", "stress"], required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--layers", default="0,3")
    p.add_argument("--batch", type=int)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    doc = root / "doc/multichip_decoder"
    sha = hashlib.sha256((root / "tt/multichip_decoder.py").read_bytes()).hexdigest()
    env = dict(
        os.environ,
        TT_METAL_TRACE_ALLOC_TRACKING="1",
        TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
        MULTICHIP_JOB_TIMEOUT="1200",
    )
    env.pop("TT_METAL_DEVICE_PROFILER", None)
    layers = [0] if a.suite == "stack" else list(map(int, a.layers.split(",")))
    shapes = {
        "context-smoke": [(1, 513)],
        "context": [(1, 262143), (1, 262144)],
        "stack": [(2, 65), (32, 33)],
        "stress": [(1, 128), (32, 33)],
    }[a.suite]
    if a.batch:
        shapes = [s for s in shapes if s[0] == a.batch]
    for layer in layers:
        for batch, length in shapes:
            stem = f"{a.tag}_l{layer}_b{batch}_s{length}"
            reference = doc / (stem + ".pt")
            for baseline in [True, False]:
                tag = stem + ("_baseline" if baseline else "_tp4")
                module = "context" if a.suite.startswith("context") else ("decoder" if a.suite == "stress" else "stack")
                command = [
                    "python",
                    "-m",
                    f"models.autoports.qwen_qwen3_6_27b.tests.run_multichip_{module}",
                    "--length",
                    str(length),
                    "--reference",
                    str(reference),
                    "--output",
                    str(doc / (tag + ".json")),
                ]
                if baseline:
                    command += ["--baseline"]
                if module == "stack":
                    command += [
                        "--activation-file",
                        str(root / "doc/optimized_decoder/text_activations/text_l0_b32_s254.pt"),
                    ]
                if module != "stack":
                    command += ["--layer", str(layer)]
                if module == "context" and not baseline:
                    command += ["--reserve-gib", "20"]
                if module != "context":
                    command += ["--batch", str(batch)]
                if module == "decoder":
                    command += ["--trace", "--repeats", "100", "--stress", "128" if batch == 1 else "512"]
                assert hashlib.sha256((root / "tt/multichip_decoder.py").read_bytes()).hexdigest() == sha
                code = subprocess.run(["bash", str(doc / "run_job.sh"), tag, *command], env=env).returncode
                with (doc / (a.tag + "_commands.jsonl")).open("a") as stream:
                    stream.write(json.dumps(dict(command=command, source_sha256=sha, exit_code=code)) + "\n")
                if code:
                    raise SystemExit(code)


if __name__ == "__main__":
    main()
