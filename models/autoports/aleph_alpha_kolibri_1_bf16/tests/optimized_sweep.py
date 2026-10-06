# SPDX-License-Identifier: Apache-2.0
"""Serialized fresh-process decoder candidates, with exact command/status records."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cases, layers=(0,), real=True, module="optimized_profile"):
    assert module in ("optimized_profile", "optimized_context_edges")
    evidence = ROOT / "doc/optimized_decoder"
    manifest = evidence / "sweep_commands.jsonl"
    for case in cases:
        tag, policy = case[:2]
        extras = case[2] if len(case) > 2 else {}
        for layer in layers:
            env = dict(
                os.environ,
                TT_METAL_TRACE_ALLOC_TRACKING="1" if module == "optimized_context_edges" else "0",
                OPT_REAL_INPUT=str(int(real)),
                OPT_TAG=tag,
                OPT_POLICY=json.dumps(policy or {}),
            )
            env.update(
                {key: json.dumps(value) if isinstance(value, dict) else str(value) for key, value in extras.items()}
            )
            command = [
                sys.executable,
                "-m",
                "models.autoports.aleph_alpha_kolibri_1_bf16.tests." + module,
                "--layer",
                str(layer),
            ]
            if module == "optimized_profile":
                command.extend(["--repetitions", "100"])
            if policy is None:
                command.append("--baseline")
            if len(case) > 3:
                command.extend(item.replace("{layer}", str(layer)) for item in case[3])
            log = evidence / f"{tag}_{layer}.log"
            print("START", tag, layer, flush=True)
            with log.open("w") as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            record = dict(
                tag=tag,
                layer=layer,
                command=command,
                policy=policy,
                real=real,
                exit_code=result.returncode,
                log=str(log),
                extra_environment=extras,
            )
            with manifest.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print("DONE", tag, layer, result.returncode, flush=True)
            # Host validation errors can be diagnosed after the sweep; don't continue after a signal.
            if result.returncode < 0:
                raise SystemExit("Device process signal: inspect and recover before continuing")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("matrix", type=Path)
    p.add_argument("--layers", type=int, nargs="+", default=[0])
    p.add_argument("--module", default="optimized_profile")
    a = p.parse_args()
    run(json.loads(a.matrix.read_text()), a.layers, module=a.module)
