"""Compare the inherited D512 cache-update L1 boundary in both implementations."""
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path("models/autoports/google_gemma_4_31b_it/doc/fused_decoder")
records = []
for batch in (36, 37):
    for functional in (True, False):
        name = f"full_capacity_{batch}_" + ("functional" if functional else "fused")
        module = "run_fusion_candidate" if functional else "run_fused_decoder"
        args = [
            sys.executable,
            "-m",
            "models.autoports.google_gemma_4_31b_it.tests." + module,
            "--layer",
            "5",
            "--batch",
            str(batch),
            "--seq",
            "33",
            "--result",
            str(root / (name + ".json")),
        ]
        if functional:
            args += ["--functional"]
        print("START", name, flush=True)
        log = root / (name + ".log")
        with log.open("w") as f:
            r = subprocess.run(
                args,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0"),
            )
        physical_limit = r.returncode != 0 and (
            "Statically allocated dataflow buffers" in log.read_text() or "overlap" in log.read_text()
        )
        records.append(dict(command=args, exit_code=r.returncode, physical_l1_limit=physical_limit))
        (root / "full_batch_capacity_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
        print("END", name, r.returncode, physical_limit, flush=True)
        if r.returncode and not physical_limit:
            raise SystemExit(r.returncode)
assert records[0]["exit_code"] == records[1]["exit_code"] == 0
assert records[2]["physical_l1_limit"] and records[3]["physical_l1_limit"]
