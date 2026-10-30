"""Serialized candidate correctness and host trace ranking; failures remain evidence."""
import json
import os
import subprocess
import sys
from pathlib import Path

out = Path("models/autoports/google_gemma_4_31b_it/doc/fused_decoder/candidates_adapted")
out.mkdir(parents=True, exist_ok=True)
base = "keep_rope_sharded,concat_heads,scalar_merge"
options = {
    "base": base,
    "split_unpacked": base + ",split_unpacked",
    "concat_decode": base + ",concat_decode",
    "both_heads": base + ",split_unpacked,concat_decode",
    "shared_kv_norm": base + ",shared_kv_norm",
    "qkv": base + ",pack_qkv",
    "mlp": base + ",pack_mlp",
    "all_packed": base + ",pack_qkv,pack_mlp",
    "matmul_gelu": base + ",matmul_gelu",
}
records = []
for layer in (0, 5):
    for name, opts in options.items():
        if name == "shared_kv_norm" and layer == 0:
            continue
        for mode in ("correctness", "timing"):
            path = out / f"{layer}_{name}_{mode}"
            args = [
                sys.executable,
                "-m",
                "models.autoports.google_gemma_4_31b_it.tests.run_fusion_candidate",
                "--extended-candidate",
                "--layer",
                str(layer),
                "--result",
                str(path.with_suffix(".json")),
            ]
            if opts is None:
                args += ["--functional"]
            else:
                args += ["--options", opts]
            if mode == "timing":
                args += ["--timing"]
            env = dict(
                os.environ,
                TT_METAL_TRACE_ALLOC_TRACKING="1" if mode == "correctness" else "0",
                TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
            )
            print("START", layer, name, mode, flush=True)
            with path.with_suffix(".log").open("w") as f:
                r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
            records.append(dict(command=args, exit_code=r.returncode, log=str(path.with_suffix(".log"))))
            (out / "manifest.json").write_text(json.dumps(records, indent=2))
            print("END", layer, name, mode, r.returncode, flush=True)
            if r.returncode:
                break
