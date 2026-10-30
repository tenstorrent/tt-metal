"""Serialize the review-requested batched-fill experiment on real layer weights."""
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path("models/autoports/google_gemma_4_31b_it/doc/fused_decoder/batched_fill")
root.mkdir(exist_ok=True)
records = []
for layer in (0, 5):
    for timing in (False, True):
        name = f"layer{layer}_" + ("timing" if timing else "correctness")
        args = [
            sys.executable,
            "-m",
            "models.autoports.google_gemma_4_31b_it.tests.batched_fill_candidate",
            "--layer",
            str(layer),
            "--batch",
            "32",
            "--seq",
            "1025",
            "--result",
            str(root / (name + ".json")),
        ]
        if timing:
            args.append("--timing")
        env = dict(
            os.environ,
            TT_METAL_TRACE_ALLOC_TRACKING="0" if timing else "1",
            TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
        )
        print("START", name, flush=True)
        with (root / (name + ".log")).open("w") as f:
            r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
        records.append(dict(command=args, exit_code=r.returncode))
        (root / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
        print("END", name, r.returncode, flush=True)
        if r.returncode:
            raise SystemExit(r.returncode)
