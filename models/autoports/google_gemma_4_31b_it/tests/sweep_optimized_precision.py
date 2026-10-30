"""Serialized real-weight precision screening, preserving failed-candidate evidence."""

import json
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "doc/optimized_decoder/precision"
OUT.mkdir(parents=True, exist_ok=True)
manifest = []
for layer in (0, 5):
    for name, mlp, attn, down in [
        ("bfp8", "bfp8", "bfp8", "bfp8"),
        ("mlp_lofi", "bfp8_lofi", "bfp8", "bfp8_lofi"),
        ("gate4", "bfp4", "bfp8", "bfp8"),
        ("mlp4", "bfp4", "bfp8", "bfp4"),
        ("attn4", "bfp8", "bfp4", "bfp8"),
        ("attn_lofi", "bfp8", "bfp8_lofi", "bfp8"),
    ]:
        key = f"{layer}_{name}"
        options = ["--layer", str(layer), "--precision", mlp, "--attention-precision", attn, "--down-precision", down]
        for phase, module in [("correctness", "run_optimized_decoder"), ("timing", "time_optimized_candidate")]:
            command = [
                sys.executable,
                "-m",
                "models.autoports.google_gemma_4_31b_it.tests." + module,
                *options,
                "--result",
                str(OUT / f"{key}_{phase}.json"),
            ]
            if phase == "correctness":
                command += ["--seq", "33"]
            env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
            print("START", key, phase, flush=True)
            with (OUT / f"{key}_{phase}.log").open("w") as log:
                code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            manifest.append(dict(key=key, phase=phase, command=command, exit_code=code))
            (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print("END", key, phase, code, flush=True)
