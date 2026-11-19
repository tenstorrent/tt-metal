# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small device positive/negative controls for the max-context scalar check."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn

from .run_full_model_acceptance import terminal_logits_finite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    mesh = None
    report = {
        "passed": False,
        "scope": "terminal scalar-check controls; no model or max-context execution",
        "cases": [],
    }
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=32768)
        vocab = 248320
        for name, row, rank, poison, expected in (
            ("finite", 0, 0, 1.0, True),
            ("active_nan_rank1", 0, 1, float("nan"), False),
            ("active_posinf_rank2", 0, 2, float("inf"), False),
            ("active_neginf_rank3", 0, 3, -float("inf"), False),
            ("inactive_nan_ignored", 17, 0, float("nan"), True),
        ):
            host = torch.ones((1, 1, 32, vocab), dtype=torch.bfloat16)
            host[0, 0, row, rank * (vocab // 4) + 13] = poison
            logits = ttnn.from_torch(
                host,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            )
            generator = SimpleNamespace(
                sample_logits=logits, model=SimpleNamespace(config=SimpleNamespace(vocab_size=vocab))
            )
            result = terminal_logits_finite(generator)
            report["cases"].append({"name": name, "expected_finite": expected, **result})
            assert result["all_finite"] == expected, report["cases"][-1]
            del generator, logits
        report["passed"] = True
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
