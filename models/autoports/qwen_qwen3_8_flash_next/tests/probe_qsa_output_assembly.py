# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate full-context slicing and paged output assembly without attention."""
import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    fixture = torch.load(args.fixture, weights_only=True, mmap=True)
    expected = fixture["hidden"].contiguous()
    length, width = expected.shape[-2:]
    assert length == 262144 and width == 2560
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = {"scope": __doc__, "rows": length, "runtime_audit": True}
    try:
        source = ttnn.from_torch(expected.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        pages = ttnn.from_torch(
            torch.arange(length // 32).reshape(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
        )
        output = ttnn.allocate_tensor_on_device(
            ttnn.Shape((length // 32, 1, 32, width)), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        with runtime_audit():
            for start in range(0, length, 32):
                chunk = source[:, :, start : start + 32, :]
                page_table = pages[:, start // 32 : start // 32 + 1]
                ttnn.experimental.paged_fill_cache(output, chunk, page_table, batch_idx=0)
            result = ttnn.reshape(output, (1, 1, length, width))
        actual = ttnn.to_torch(result).reshape(expected.shape)
        mismatches = (expected != actual).reshape(length, width).sum(-1)
        bad = torch.nonzero(mismatches).flatten()
        report.update(
            exact=torch.equal(expected, actual),
            mismatched_rows=bad.numel(),
            first_bad_rows=bad[:32].tolist(),
            max_abs_error=(expected.float() - actual.float()).abs().max().item(),
        )
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)
    print(json.dumps(report), flush=True)
    assert report["exact"]


if __name__ == "__main__":
    main()
