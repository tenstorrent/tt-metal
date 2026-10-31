"""Serial stage05 acceptance queue; abort before another device job on failure."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument(
    "--groups",
    nargs="+",
    default=["boundaries", "norm_boundaries", "batches", "stress", "watcher", "stack", "context", "profiles"],
)
a = p.parse_args()
root = Path("models/autoports/google_gemma_4_31b_it/doc/optimized_multichip_decoder")
records = []
for group in a.groups:
    env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
    env.pop("TT_METAL_DEVICE_PROFILER", None)
    env.pop("TT_METAL_WATCHER", None)
    if group == "watcher":
        env["TT_METAL_WATCHER_DISABLE_ETH"] = "1"
    if group == "stack":
        cmd = [
            sys.executable,
            "-m",
            "models.autoports.google_gemma_4_31b_it.tests.check_multichip_stack",
            "--ring",
            "--result",
            str(root / "stack.json"),
        ]
    elif group == "profiles":
        cmd = [
            sys.executable,
            "-m",
            "models.autoports.google_gemma_4_31b_it.tests.run_multichip_profiles",
            "--output-root",
            str(root),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "models.autoports.google_gemma_4_31b_it.tests.run_multichip_validation",
            "--group",
            group,
            "--output-root",
            str(root),
        ]
    print("RUN_GROUP", group, flush=True)
    with (root / f"acceptance_{group}.log").open("w") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    records.append(dict(group=group, command=cmd, exit_code=r.returncode))
    (root / "acceptance_queue.json").write_text(json.dumps(records, indent=2) + "\n")
    if r.returncode:
        raise SystemExit(f"FAILED {group}: inspect before resuming")
    print("PASS_GROUP", group, flush=True)
