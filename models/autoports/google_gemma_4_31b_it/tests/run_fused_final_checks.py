"""Finish affected boundary and profile gates after the diagnosed L1 relocation."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc/fused_decoder"
BASE = "models.autoports.google_gemma_4_31b_it.tests."
records = [r for r in json.loads((OUT / "batched_acceptance_manifest.json").read_text()) if r["exit_code"] == 0]
for name, module, extra in [
    ("batch_boundaries", "check_fused_batch_boundaries", []),
    ("prefill_index_boundaries", "check_prefill_index_boundary", []),
    (
        "batched_profiles",
        "run_fused_profile_queue",
        ["--artifact-root", "tracy_batched_fill", "--batches", "32", "16", "8", "--retain-batch1"],
    ),
]:
    args = [sys.executable, "-m", BASE + module, *extra]
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        env.pop(key, None)
    print("START", name, flush=True)
    with (OUT / (name + ".log")).open("w") as f:
        r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    records.append(
        dict(
            name=name,
            command=args,
            exit_code=r.returncode,
            source_hashes={
                str(ROOT / "tt/fused_decoder.py"): hashlib.sha256(
                    (ROOT / "tt/fused_decoder.py").read_bytes()
                ).hexdigest()
            },
        )
    )
    (OUT / "batched_acceptance_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print("END", name, r.returncode, flush=True)
    if r.returncode:
        raise SystemExit(r.returncode)
