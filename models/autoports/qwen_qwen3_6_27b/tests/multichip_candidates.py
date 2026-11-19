# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer candidate comparison, serialized and stopped at the first failure."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parents[1]
D = R / "doc/multichip_decoder"
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--layers", default="0,3")
p.add_argument("--shapes", default="1x512,32x33")
p.add_argument("--policies", type=Path)
p.add_argument("--text-activations", action="store_true")
a = p.parse_args()
sha = hashlib.sha256((R / "tt/multichip_decoder.py").read_bytes()).hexdigest()
env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="0", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
env.pop("TT_METAL_WATCHER", None)
env.pop("TT_METAL_DEVICE_PROFILER", None)
for layer in map(int, a.layers.split(",")):
    for shape in a.shapes.split(","):
        batch, length = map(int, shape.split("x"))
        ref = D / f"{a.tag}_ref_l{layer}_{shape}.pt"
        policies = {
            "baseline": {},
            "base": {},
            "flat": {"flat_ccl": True, "flat_norm": True},
            "fused": {"flat_ccl": True, "flat_norm": True, "fused_mmrs": True},
            "replicated": {},
        }
        if a.policies:
            policies = {"baseline": {}, **json.loads(a.policies.read_text())}
        for name, policy in policies.items():
            assert hashlib.sha256((R / "tt/multichip_decoder.py").read_bytes()).hexdigest() == sha
            tag = f"{a.tag}_{name}_l{layer}_{shape}"
            cmd = [
                "python",
                "-m",
                "models.autoports.qwen_qwen3_6_27b.tests.run_multichip_decoder",
                "--trace",
                "--repeats",
                "100",
                "--layer",
                str(layer),
                "--batch",
                str(batch),
                "--length",
                str(length),
                "--reference",
                str(ref),
                "--output",
                str(D / (tag + ".json")),
                "--policy-json",
                json.dumps(policy),
            ]
            if a.text_activations:
                cmd += [
                    "--activation-file",
                    str(R / f"doc/optimized_decoder/text_activations/text_l{layer}_b{batch}_s{length}.pt"),
                ]
            if name == "baseline":
                cmd += ["--baseline"]
            if name == "replicated":
                cmd += ["--layout", "replicated"]
            code = subprocess.run(["bash", str(D / "run_job.sh"), tag, *cmd], env=env).returncode
            with (D / (a.tag + "_commands.jsonl")).open("a") as f:
                f.write(
                    json.dumps(
                        {"command": cmd, "source_sha256": sha, "exit_code": code, "regime": "serial", "tracking": 0}
                    )
                    + "\n"
                )
            if code:
                raise SystemExit(code)
