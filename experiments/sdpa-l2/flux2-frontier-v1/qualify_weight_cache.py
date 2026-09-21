# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify fresh-process FLUX.2 cold/warm cache loads before the image suite."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    args = parser.parse_args()
    if (args.cache / args.checkpoint.name).exists():
        parser.error("Cold qualification requires a fresh checkpoint cache namespace; existing caches are not deleted")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.embeddings, args.output / "prompt-embeddings")
    results = {}
    for phase in ("cold", "warm"):
        env = dict(
            os.environ,
            TT_DIT_CACHE_DIR=str(args.cache),
            TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES="0",
            FLUX2_CHECKPOINT=str(args.checkpoint),
            FLUX2_OUTPUT=str(args.output / phase),
            FLUX2_VARIANT="stock",
            FLUX2_STEPS="2",
            FLUX2_PROMPTS="1",
            FLUX2_SEEDS="1",
            FLUX2_BLOCK_BENCH="0",
            FLUX2_TRACED="1",
            FLUX2_EXPLORATORY="1",
            FLUX2_MODEL_REPAIR="main_fused",
            FLUX2_CONDITIONING="stock",
            FLUX2_VERIFY_CACHED_WEIGHT="1",
            FLUX2_REQUIRE_WEIGHT_CACHE="1" if phase == "warm" else "0",
        )
        print("START_CACHE_QUALIFICATION", phase, flush=True)
        with (args.output / f"{phase}.log").open("x") as log:
            subprocess.run(
                [sys.executable, "-m", "pytest", "-s", "-q", str(Path(__file__).with_name("test_pipeline.py"))],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
        manifest = json.loads((args.output / phase / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        loads = [r for r in manifest["weight_loads"] if not r["already_loaded"]]
        assert len(loads) >= 3, loads
        assert all(r["torch_state_dict_requested"] == (phase == "cold") for r in loads), loads
        results[phase] = {
            "pipeline_setup_seconds": manifest["pipeline_setup_seconds"],
            "weight_loads": loads,
            "weight_load_seconds_including_cache_write": sum(r["seconds_including_cache_write"] for r in loads),
            "validation_weight": manifest["cache_validation_weight"],
        }
        print("COMPLETED_CACHE_QUALIFICATION", phase, json.dumps(results[phase]), flush=True)
    assert results["cold"]["validation_weight"] == results["warm"]["validation_weight"]
    assert len(results["warm"]["validation_weight"]["device_shard_sha256"]) == 8
    results["status"] = "completed"
    results["precision_scope"] = (
        "All-shard exact hash of the original stall-site weight; synthetic probe covers sharded weights"
    )
    (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
