"""Resolve batch-dependent head-transform selection without weight replicas."""
import json
import os
import subprocess
import sys
from pathlib import Path

out = Path("models/autoports/google_gemma_4_31b_it/doc/fused_decoder/candidates_batches")
out.mkdir(parents=True, exist_ok=True)
base = "keep_rope_sharded,concat_heads,scalar_merge,split_unpacked,fused_cache"
records = []
for batch in (32, 16, 8, 1):
    for layer in (0, 5):
        opts = base + (",pack_qkv,shared_kv_norm" if layer == 5 else "")
        options = {
            "separate": opts,
            "split": opts + ",split_heads",
            "split_sharded": opts + ",split_heads,keep_split_sharded",
        }
        for name, option in options.items():
            for mode in ("correctness", "timing"):
                path = out / f"{layer}_b{batch}_{name}_{mode}"
                args = [
                    sys.executable,
                    "-m",
                    "models.autoports.google_gemma_4_31b_it.tests.run_fusion_candidate",
                    "--shared-rope",
                    "--layer",
                    str(layer),
                    "--batch",
                    str(batch),
                    "--options",
                    option,
                    "--result",
                    str(path.with_suffix(".json")),
                ]
                if mode == "timing":
                    args += ["--timing"]
                env = dict(
                    os.environ,
                    TT_METAL_TRACE_ALLOC_TRACKING="1" if mode == "correctness" else "0",
                    TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
                )
                print("START", layer, batch, name, mode, flush=True)
                with path.with_suffix(".log").open("w") as f:
                    r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
                records.append(dict(command=args, exit_code=r.returncode, log=str(path.with_suffix(".log"))))
                (out / "manifest.json").write_text(json.dumps(records, indent=2))
                print("END", layer, batch, name, mode, r.returncode, flush=True)
                if r.returncode:
                    break
