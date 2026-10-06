# SPDX-License-Identifier: Apache-2.0
"""Run unchanged packaged readiness gates and retain their returned statistics."""

import argparse
import json
import os
from pathlib import Path

import torch
from readiness_check.run_prefill_check import run_prefill_check
from readiness_check.run_teacher_forcing import run_teacher_forcing

import ttnn

from .full_provenance import provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prefill", "decode"])
    a = p.parse_args()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    source = provenance()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    try:
        runner = run_prefill_check if a.mode == "prefill" else run_teacher_forcing
        rows = runner(model_dir=root, reference_path=root / "readiness_aime24_chat.refpt", mesh_device=mesh)
        result = dict(
            provenance=source,
            runner=runner.__module__,
            rows=rows,
            passed=all(row["top5"] >= 0.98 and row["top100"] == 1.0 for row in rows),
        )
        (Path(os.environ.get("FULL_ARTIFACT_DIR", root / "doc/full_model")) / f"accuracy_{a.mode}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print("READINESS_RESULT", json.dumps(result), flush=True)
        assert result["passed"], result
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
