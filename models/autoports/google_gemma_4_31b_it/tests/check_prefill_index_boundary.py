"""Check prepared-index capacity and larger-batch prefill fallback, without decode."""
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path("models/autoports/google_gemma_4_31b_it/doc/fused_decoder")
records = []
for layer in (0, 5):
    for batch in (110, 111):
        name = f"prefill_index_boundary_{layer}_{batch}"
        args = [
            sys.executable,
            "-m",
            "models.autoports.google_gemma_4_31b_it.tests.test_fused_equivalence",
            "--layer",
            str(layer),
            "--batch",
            str(batch),
            "--seq",
            "33",
            "--prefill-only",
            "--result",
            str(root / (name + ".json")),
        ]
        print("START", name, flush=True)
        with (root / (name + ".log")).open("w") as f:
            r = subprocess.run(
                args,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0"),
            )
        records.append(dict(command=args, exit_code=r.returncode))
        (root / "prefill_index_boundary_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
        print("END", name, r.returncode, flush=True)
        if r.returncode:
            raise SystemExit(r.returncode)
