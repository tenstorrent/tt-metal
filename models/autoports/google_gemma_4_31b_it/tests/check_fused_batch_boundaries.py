"""Check selection boundaries and preserve batches outside the tuned set."""
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path("models/autoports/google_gemma_4_31b_it/doc/fused_decoder")
records = []
for layer in (0, 5):
    # Full D512 cache updates inherit an S33 per-core L1 limit above B36;
    # check_full_batch_capacity compares the functional and fused boundary.
    for batch in (15, 17, 31, 33, 55, 56) if layer == 0 else (15, 17, 31, 33, 34, 36):
        name = f"batch_boundary_{layer}_{batch}"
        args = [
            sys.executable,
            "-m",
            "models.autoports.google_gemma_4_31b_it.tests.run_fused_decoder",
            "--layer",
            str(layer),
            "--batch",
            str(batch),
            "--seq",
            "33",
            "--result",
            str(root / (name + ".json")),
        ]
        env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
        if layer == 5 and batch == 36:
            env.update(TT_METAL_WATCHER="10", TT_METAL_LOGS_PATH=str(root / "watcher_l1_relocation"))
        print("START", name, flush=True)
        with (root / (name + ".log")).open("w") as f:
            r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
        records.append(dict(command=args, exit_code=r.returncode))
        (root / "batch_boundary_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
        print("END", name, r.returncode, flush=True)
        if r.returncode:
            raise SystemExit(r.returncode)
