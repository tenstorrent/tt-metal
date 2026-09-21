# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serial, fail-fast suite launcher. Never silently substitutes an attention mode."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["D", "C", "B", "A", "E", "F", "G", "stock"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--block-bench", action="store_true")
    parser.add_argument("--append", action="store_true", help="Append only new variants; never overwrite an existing variant")
    args = parser.parse_args()
    if any(v not in ("A", "B", "C", "D", "E", "F", "G", "stock") for v in args.variants):
        parser.error("Unknown variant")
    args.output.mkdir(parents=True, exist_ok=args.append)
    for variant in args.variants:
        env = dict(
            os.environ,
            FLUX2_OUTPUT=str(args.output / variant),
            FLUX2_VARIANT=variant,
            FLUX2_STEPS=str(args.steps),
            FLUX2_PROMPTS=str(args.prompts),
            FLUX2_SEEDS=str(args.seeds),
            FLUX2_TRACED="1",
            FLUX2_BLOCK_BENCH="1" if args.block_bench else "0",
        )
        command = [sys.executable, "-m", "pytest", "-s", "-q", str(Path(__file__).with_name("test_pipeline.py"))]
        print("START_VARIANT", variant, flush=True)
        with (args.output / f"{variant}.log").open("x") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Variant {variant} failed with status {result.returncode}; see {log.name}")
        manifest = json.loads((args.output / variant / "manifest.json").read_text())
        if manifest["status"] != "completed":
            raise RuntimeError(f"Variant {variant} did not complete")
        print("COMPLETED_VARIANT", variant, len(manifest["results"]), flush=True)


if __name__ == "__main__":
    main()
