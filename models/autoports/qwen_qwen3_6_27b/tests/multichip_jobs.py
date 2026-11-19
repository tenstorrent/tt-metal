# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serialized mesh jobs: exact commands, source hashes and exit codes."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "doc/multichip_decoder"
p = argparse.ArgumentParser()
p.add_argument("--suite", choices=["correctness", "performance", "watcher"], required=True)
p.add_argument("--tag", required=True)
p.add_argument("--record-row-dispatch", action="store_true")
p.add_argument("--shapes", help="Comma-separated BxS pairs")
p.add_argument("--layers", default="0,3")
p.add_argument("--recorded-shapes", default="", help="BxS cases using matching text_lL_bB_sS.pt fixtures")
p.add_argument("--activation-dir", type=Path, default=D)
a = p.parse_args()
sha = hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest()
env = dict(os.environ)
for key in ["TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"]:
    env.pop(key, None)
env.update(
    TT_METAL_TRACE_ALLOC_TRACKING="0" if a.suite == "performance" else "1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0"
)
shapes = [(2, 33), (32, 33), (3, 513), (1, 31), (1, 1025)] if a.suite == "correctness" else [(1, 512), (32, 33)]
if a.suite == "watcher":
    env["TT_METAL_WATCHER"] = "10"
    shapes = [(32, 33), (3, 513)]
if a.shapes:
    shapes = [tuple(map(int, s.split("x"))) for s in a.shapes.split(",")]
for layer in map(int, a.layers.split(",")):
    for batch, length in shapes:
        ref = D / f"{a.tag}_ref_l{layer}_b{batch}_s{length}.pt"
        for mode in ["baseline", "sharded", "replicated"] if a.suite == "performance" else ["baseline", "sharded"]:
            tag = f"{a.tag}_{mode}_l{layer}_b{batch}_s{length}"
            cmd = [
                "python",
                "-m",
                "models.autoports.qwen_qwen3_6_27b.tests.run_multichip_"
                + ("boundaries" if a.record_row_dispatch and mode != "baseline" else "decoder"),
                "--trace",
                "--layer",
                str(layer),
                "--batch",
                str(batch),
                "--length",
                str(length),
                "--repeats",
                "100" if a.suite == "performance" else "20",
                "--reference",
                str(ref),
                "--output",
                str(D / (tag + ".json")),
            ]
            if f"{batch}x{length}" in a.recorded_shapes.split(","):
                activation = a.activation_dir / f"text_l{layer}_b{batch}_s{length}.pt"
                assert activation.is_file(), activation
                cmd += ["--activation-file", str(activation)]
            cmd += ["--baseline"] if mode == "baseline" else ["--layout", mode]
            assert hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest() == sha
            result = subprocess.run(["bash", str(D / "run_job.sh"), tag, *cmd], env=env)
            with (D / (a.tag + "_commands.jsonl")).open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "command": cmd,
                            "environment": {k: v for k, v in env.items() if k.startswith("TT_METAL_")},
                            "source_sha256": sha,
                            "exit_code": result.returncode,
                        }
                    )
                    + "\n"
                )
            if result.returncode:
                raise SystemExit(result.returncode)
