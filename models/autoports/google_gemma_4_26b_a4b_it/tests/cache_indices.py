# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regression for physical cache row addresses beyond Float32 integer precision."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.precise_attention import PrecisePagedAttention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    results = []
    try:
        for heads in (8, 2):
            operation = PrecisePagedAttention(mesh, SimpleNamespace(num_key_value_heads=heads), 262144)
            ids = torch.tensor([[0, 65535, 65536, 65537, 262143, 262144]], dtype=torch.int32)
            physical = ttnn.from_torch(ids, device=mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            with device_only():
                rows = operation.cache_row_indices(physical)
            actual = ttnn.to_torch(rows).long()
            expected = (ids.long()[..., None] * (heads * 32) + torch.arange(heads * 32)).reshape(1, -1)
            equal = torch.equal(actual, expected)
            results.append(dict(kv_heads=heads, max_physical_page=int(ids.max()), exact=equal))
            assert equal, results
        args.output.write_text(json.dumps(dict(results=results, runtime_audit="clean"), indent=2) + "\n")
        print(results, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
