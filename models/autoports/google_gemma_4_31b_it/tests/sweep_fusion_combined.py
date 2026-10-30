"""Combine compatible graph rewrites and check the largest configured batch."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

out = Path("models/autoports/google_gemma_4_31b_it/doc/fused_decoder/candidates_combined")
out.mkdir(parents=True, exist_ok=True)
base = "keep_rope_sharded,concat_heads,scalar_merge,split_unpacked"
records = []
for batch in (1, 32):
    for layer in (0, 5):
        options = {"base": base, "cache": base + ",fused_cache"}
        if layer == 5:
            options.update(
                packed=base + ",pack_qkv",
                packed_cache=base + ",pack_qkv,fused_cache",
                packed_cache_norm=base + ",pack_qkv,fused_cache,shared_kv_norm",
                split_decode=base + ",pack_qkv,fused_cache,shared_kv_norm,split_heads",
            )
        if batch == 32:
            options.update(
                packed_mlp=base + ",fused_cache,pack_mlp" + (",pack_qkv,shared_kv_norm" if layer == 5 else "")
            )
        for name, opts in options.items():
            for mode in ("correctness", "timing"):
                path = out / f"{layer}_b{batch}_{name}_{mode}"
                args = [
                    sys.executable,
                    "-m",
                    "models.autoports.google_gemma_4_31b_it.tests.run_fusion_candidate",
                    "--extended-candidate",
                    "--layer",
                    str(layer),
                    "--batch",
                    str(batch),
                    "--options",
                    opts,
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
                sources = {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path("models/autoports/google_gemma_4_31b_it/tests").glob("*fusion*.py")
                }
                with path.with_suffix(".log").open("w") as f:
                    r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
                records.append(
                    dict(command=args, exit_code=r.returncode, log=str(path.with_suffix(".log")), sources=sources)
                )
                (out / "manifest.json").write_text(json.dumps(records, indent=2))
                print("END", layer, batch, name, mode, r.returncode, flush=True)
                if r.returncode:
                    break
