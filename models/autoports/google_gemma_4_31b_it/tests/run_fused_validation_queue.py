"""Serialized stage validation; stops at the first failure for investigation."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc/fused_decoder"
MODULE = "models.autoports.google_gemma_4_31b_it.tests.run_fused_decoder"


def main():
    jobs = []
    if "--skip-context" not in sys.argv:
        jobs.append(("sliding_context_final", ["--layer", "0", "--seq", "262143"], {}))
        jobs.append(("full_context_final", ["--layer", "5", "--seq", "262143"], {}))
    for layer, kind in [(0, "sliding_attention"), (5, "full_attention")]:
        jobs.append(
            (
                kind + "_boundaries_final",
                ["--layer", str(layer), "--seq", "2049", "--extended", "--new-bucket-case"],
                {},
            )
        )
        for batch in [32, 16, 8]:
            jobs.append(
                (
                    kind + f"_batch{batch}_final",
                    ["--layer", str(layer), "--seq", "1025" if batch == 32 else "33", "--batch", str(batch)],
                    {},
                )
            )
        jobs.append(
            (
                kind + "_watcher_final",
                ["--layer", str(layer), "--seq", "2049", "--extended", "--new-bucket-case"],
                {"TT_METAL_WATCHER": "10", "TT_METAL_LOGS_PATH": str(OUT / "watcher" / (kind + "_final"))},
            )
        )
    results = []
    for name, args, env_extra in jobs:
        command = [sys.executable, "-m", MODULE, *args, "--result", str(OUT / (name + ".json"))]
        env = dict(
            os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0", **env_extra
        )
        print("START", name, flush=True)
        with (OUT / (name + ".log")).open("w") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        results.append(
            dict(
                name=name,
                command=command,
                environment={
                    k: env[k]
                    for k in ["TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", *env_extra]
                },
                exit_code=code,
            )
        )
        (OUT / "validation_queue.json").write_text(json.dumps(results, indent=2) + "\n")
        print("END", name, code, flush=True)
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
