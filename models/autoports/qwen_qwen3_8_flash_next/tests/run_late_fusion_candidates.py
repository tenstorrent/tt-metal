# SPDX-License-Identifier: Apache-2.0
"""Record and serialize the last source-audit fusion experiments."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "doc/fused_decoder"
MODULE = "models.autoports.qwen_qwen3_8_flash_next.tests."
CANDIDATES = {
    "addcmul_write": 0,
    "addcmul_read": 0,
    "l2_activation": 0,
    "softplus_activation": 0,
    "position_mac": 3,
    "default": 0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["benchmark", "correctness"])
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("TT_METAL_WATCHER"):
            del env[key]
    env.update(
        QWEN38_WEIGHT_SOURCE="real",
        QWEN38_WEIGHT_CACHE="/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next",
        OMP_NUM_THREADS="8",
    )
    for candidate in args.candidates.split(","):
        layer = CANDIDATES[candidate]
        tag = f"late_{candidate}_{args.mode}" + args.suffix
        if args.mode == "benchmark":
            command = [
                sys.executable,
                "-m",
                MODULE + "benchmark_fused_decoder",
                "--impl",
                "fused",
                "--layer",
                str(layer),
                "--tag",
                tag,
                "--candidate",
                candidate,
            ]
            env["TT_METAL_TRACE_ALLOC_TRACKING"] = "0"
        else:
            command = [sys.executable, "-m", MODULE + "validate_late_fusion", candidate, "-k", f"layer{layer}"]
            env.update(
                TT_METAL_TRACE_ALLOC_TRACKING="1",
                QWEN38_DECODER_BATCHES="1,2",
                QWEN38_DECODER_LENGTHS="1,3,4,5,31,32,33,127,128,129,259",
            )
        sources = [ROOT / "tt/fused_decoder.py", ROOT / "tests/late_fusion_candidates.py"]
        manifest = dict(
            command=command,
            environment={
                k: v for k, v in env.items() if k.startswith(("TT_METAL_", "QWEN38_")) or k == "OMP_NUM_THREADS"
            },
            source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            started_unix=time.time(),
        )
        for source in sources:
            (EVIDENCE / f"{tag}_{source.name}.txt").write_bytes(source.read_bytes())
        target = EVIDENCE / f"{tag}_command.json"
        target.write_text(json.dumps(manifest, indent=2) + "\n")
        print("START", tag, flush=True)
        with (EVIDENCE / f"{tag}.log").open("w") as handle:
            result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        manifest.update(returncode=result.returncode, elapsed_seconds=time.time() - manifest["started_unix"])
        target.write_text(json.dumps(manifest, indent=2) + "\n")
        print("END", tag, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
