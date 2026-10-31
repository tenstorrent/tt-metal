"""Serial acceptance matrix. Abort on failure before any subsequent device job."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--group", choices=["boundaries", "batches", "stress", "context", "watcher"], required=True)
p.add_argument("--layer", type=int, choices=[0, 5])
a = p.parse_args()
root = Path("models/autoports/google_gemma_4_31b_it/doc/multichip_decoder")
for layer in [a.layer] if a.layer is not None else [0, 5]:
    cases = []
    if a.group == "boundaries":
        cases = [(f"length{s}", s, 1, []) for s in [31, 32, 33, 1023, 1024, 1025, 2049]]
    if a.group == "batches":
        cases = [(f"batch{b}", 4096, b, ["--cyclic"]) for b in [1, 32, 16, 8]]
    if a.group == "stress":
        cases = [
            ("stress32", 33, 32, ["--stress"]),
            ("cyclic4097", 4097, 1, ["--cyclic", "--stress"]),
            ("wrap2047", 2047, 32, ["--cyclic", "--stress", "--distinct-offsets"]),
        ]
    if a.group == "context":
        cases = [(f"context{s}", s, 1, ["--cyclic"]) for s in [262143, 262144]]
    if a.group == "watcher":
        cases = [("watcher32", 33, 32, ["--stress", "--cyclic"])]
    for name, seq, batch, extra in cases:
        path = root / f"final_layer{layer}_{name}"
        fixture = (
            f"bringup/artifacts/optimized_decoder_layer{layer}_inputs.pt"
            if name in ["stress32", "watcher32"]
            else f"bringup/artifacts/optimized_decoder_layer{layer}_inputs262144.pt"
        )
        cmd = [
            sys.executable,
            "-m",
            "models.autoports.google_gemma_4_31b_it.tests.run_multichip_decoder",
            "--layer",
            str(layer),
            "--seq",
            str(seq),
            "--batch",
            str(batch),
            "--inputs",
            fixture,
            "--replicated",
            "--result",
            str(path.with_suffix(".json")),
            *extra,
        ]
        env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1")
        env.pop("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", None)
        if a.group == "watcher":
            env["TT_METAL_WATCHER"] = "10"
        print("RUN", path.name, flush=True)
        with path.with_suffix(".log").open("w") as stream:
            run = subprocess.run(
                cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=900 if a.group == "context" else 300
            )
        path.with_suffix(".exit.json").write_text(json.dumps({"exit_code": run.returncode, "command": cmd}) + "\n")
        if a.group == "watcher":
            source = Path("generated/watcher")
            assert source.exists(), "Watcher log directory missing"
            shutil.copytree(source, root / (path.name + "_watcher_logs"), dirs_exist_ok=True)
        if run.returncode:
            raise SystemExit(f"FAILED {path.name}; inspect and recover before continuing.")
        print("PASS", path.name, flush=True)
