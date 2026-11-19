# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Focused Watcher regression for the native INT32 fill/slice alignment repair."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.evidence import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0])
    checks = []
    try:
        for batch in (1, 2, 15, 16, 17, 31, 32):
            for absolute in (1, 31, 33, 511, 262143):
                with patch.object(ttnn, "from_torch", side_effect=AssertionError("runtime upload")), patch.object(
                    ttnn, "to_torch", side_effect=AssertionError("runtime download")
                ):
                    position = ttnn.reshape(
                        ttnn.moreh_full((1, 32), absolute, mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)[
                            :, :batch
                        ],
                        (batch,),
                    )
                    ttnn.synchronize_device(mesh)
                actual = ttnn.to_torch(position, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                assert torch.equal(actual, torch.full((batch,), absolute, dtype=torch.int32))
                checks.append({"batch": batch, "absolute_position": absolute, "exact": True})
        args.output.write_text(json.dumps({"provenance": provenance(), "checks": checks}, indent=2) + "\n")
        print("PASS", len(checks), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
