# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify exact output assembly, including partial-tile writes and canaries."""
import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_output import write_token_chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(2701)
    report = {"cases": []}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        shape = (1, 1, 97, 10240)
        target = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        expected = torch.zeros(shape, dtype=torch.bfloat16)
        specs = [(1, 0), (31, 1), (32, 32), (1, 65), (7, 70), (16, 80)]
        sources = []
        for count, start in specs:
            value = torch.randn(1, 1, count, 10240).bfloat16()
            source = ttnn.from_torch(value, layout=ttnn.TILE_LAYOUT, device=mesh)
            sources.append((value, source))
            with runtime_audit():
                write_token_chunk(source, target, start)
            expected[:, :, start : start + count] = value
        assert torch.equal(ttnn.to_torch(target), expected)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with runtime_audit():
            for (_, source), (_, start) in zip(sources, specs):
                write_token_chunk(source, target, start)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for repetition in range(2):
            for (value, source), (count, start) in zip(sources, specs):
                changed = value * (repetition + 2)
                host = ttnn.from_torch(changed, layout=ttnn.TILE_LAYOUT)
                ttnn.copy_host_to_device_tensor(host, source)
                expected[:, :, start : start + count] = changed
            with runtime_audit():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            equal = torch.equal(ttnn.to_torch(target), expected)
            report["cases"].append({"replay": repetition, "exact_equal_with_canaries": equal})
            assert equal
        report["passed"] = True
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
