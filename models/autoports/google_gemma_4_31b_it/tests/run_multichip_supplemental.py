"""Final-source lifetime and advice controls, with serialized device ownership."""

import json
import os
import subprocess
import sys
from pathlib import Path

root = Path("models/autoports/google_gemma_4_31b_it/doc/optimized_multichip_decoder")
module = "models.autoports.google_gemma_4_31b_it.tests."
commands = [
    ("shared_workspace_final", "check_multichip_shared_workspace", ["--result", str(root / "shared_workspace.json")]),
    *[
        (
            f"prefill_trace_layer{layer}",
            "check_multichip_prefill_trace",
            ["--layer", str(layer), "--result", str(root / f"prefill_trace_layer{layer}.json")],
        )
        for layer in (0, 5)
    ],
    ("readers", "run_multichip_readers", []),
]
env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
env.pop("TT_METAL_WATCHER", None)
env.pop("TT_METAL_DEVICE_PROFILER", None)
records = []
for label, runner, args in commands:
    command = [sys.executable, "-m", module + runner, *args]
    print("RUN", label, flush=True)
    with (root / f"{label}.log").open("w") as f:
        result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=1500)
    records.append(dict(label=label, command=command, exit_code=result.returncode))
    (root / "supplemental_queue.json").write_text(json.dumps(records, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(f"FAILED {label}: inspect before continuing")
    print("PASS", label, flush=True)
