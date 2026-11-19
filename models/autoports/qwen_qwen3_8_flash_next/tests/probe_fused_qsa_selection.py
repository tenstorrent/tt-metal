# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact FP32 selection/expansion with changing device bounds and stale tails."""
import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import fused_selected_token_ids


def expected_ids(scores, positions):
    result = torch.full((len(positions), 2080), 0xFFFFFFFF, dtype=torch.int64)
    for row, position in enumerate(positions.tolist()):
        complete = (position + 1) // 4
        selected = torch.argsort(scores[0, 0, row, :complete], descending=True, stable=True)[:512]
        tokens = (selected[:, None] * 4 + torch.arange(4)).flatten()
        result[row, : len(tokens)] = tokens
        tail = torch.arange(complete * 4, position + 1)
        result[row, 2048 : 2048 + len(tail)] = tail
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=Path("/home/ttuser/qwen38-flash/qsa_score_windows_205504.pt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    captured = torch.load(args.capture, mmap=True, weights_only=True, map_location="cpu")
    real_scores = (captured["query"].float() @ captured["keys"].float().transpose(-1, -2)).relu().sum(
        1, keepdim=True
    ) * 128**-0.5
    report = dict(
        passed=False,
        cases=[],
        source_sha256=hashlib.sha256(Path(inspect.getfile(fused_selected_token_ids)).read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        for count in (1, 3, 7, 16, 17, 31, 32):
            positions = [torch.arange(count), torch.arange(262144 - count, 262144), torch.full((count,), 262143)]
            values = [
                real_scores[:, :, :count].clone(),
                real_scores[:, :, -count:].clone(),
                torch.ones((1, 1, count, 65536)),
            ]
            for row, position in enumerate(positions[0].tolist()):
                values[0][0, 0, row, (position + 1) // 4 :] = float("nan")
            # Scores differing by one FP32 ULP must retain their full ordering.
            values[2][..., 600:1100] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
            values[2][..., 1200] = -0.0
            expected = [expected_ids(value, position) for value, position in zip(values, positions)]
            host_scores = [ttnn.from_torch(value, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT) for value in values]
            host_positions = [
                ttnn.from_torch(position.float().reshape(1, 1, count, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
                for position in positions
            ]
            scores_tt = ttnn.to_device(host_scores[0], mesh)
            positions_tt = ttnn.to_device(host_positions[0], mesh)
            with runtime_audit():
                warm = fused_selected_token_ids(scores_tt, positions_tt)
            ttnn.synchronize_device(mesh)
            del warm
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            with runtime_audit():
                output = fused_selected_token_ids(scores_tt, positions_tt)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            trace_allocation_tracker.acknowledge_corruptible(output)
            previous = None
            for branch in (0, 1, 2, 0):
                ttnn.copy_host_to_device_tensor(host_scores[branch], scores_tt)
                ttnn.copy_host_to_device_tensor(host_positions[branch], positions_tt)
                start = time.perf_counter()
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                milliseconds = (time.perf_counter() - start) * 1000
                actual = ttnn.to_torch(output).to(torch.int64)
                exact = torch.equal(actual, expected[branch])
                repeated = previous is None or branch != 0 or torch.equal(actual, previous)
                if previous is None:
                    previous = actual.clone()
                row = dict(count=count, branch=branch, exact=exact, repeated=repeated, traced_ms=milliseconds)
                report["cases"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print("SELECTION", row, flush=True)
                assert exact and repeated
            ttnn.release_trace(mesh, trace)
            trace = None
            del output, scores_tt, positions_tt
        report["passed"] = True
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
