# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded all-layer readiness entrypoint with compact JSON evidence.

Uses the installed readiness runners without modifying their accuracy logic.
One runner owns one model lifetime. Use --mode to rerun a failing gate directly.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from readiness_check.run_prefill_check import run_prefill_check
from readiness_check.run_teacher_forcing import run_teacher_forcing

import ttnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prefill", "teacher"], required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    torch.set_num_threads(8)
    provenance = {
        "command": sys.argv,
        "time_unix": time.time(),
        "layers": 40,
        "mesh": [1, 4],
        "measurement_regime": "serial",
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted((root / "tt").glob("*.py"))]
        },
        "environment": {
            k: os.environ.get(k) for k in ["TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"]
        },
    }
    Path(args.output + ".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    try:
        fn = run_prefill_check if args.mode == "prefill" else run_teacher_forcing
        rows = fn(
            model_dir=root,
            reference_path=root / "readiness_aime24_chat.refpt",
            mesh_device=mesh,
            build_kwargs={"max_seq_len": args.context, "prepare": args.mode == "teacher"},
        )
        result = {
            "mode": args.mode,
            "rows": rows,
            "passed": all(r["top5"] >= 0.98 and r["top100"] == 1.0 for r in rows),
        }
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        assert result["passed"], "Full-model top-k gate failed"
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
