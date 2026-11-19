# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prototype: fold packed gate/up slices into the SiLU/multiply reader."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import fused_packed_silu as packed_silu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(819)
    torch.set_num_threads(8)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = dict(rows=[], passed=False)
    trace = None
    try:
        for groups in (1, 10, 512):
            for count in (1, 7, 31, 32):
                host = (torch.randn(1, 1, 1, groups, count, 1280) * 3).bfloat16()
                x = ttnn.from_torch(host, device=mesh, layout=ttnn.TILE_LAYOUT)

                def baseline():
                    return ttnn.multiply(x[..., :640], x[..., 640:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU])

                expected = ttnn.to_torch(baseline())
                actual = ttnn.to_torch(packed_silu(x))
                pcc = float(torch.corrcoef(torch.stack([expected.float().flatten(), actual.float().flatten()]))[0, 1])
                row = dict(
                    groups=groups,
                    count=count,
                    exact=torch.equal(expected, actual),
                    pcc=pcc,
                    max_abs=float((expected.float() - actual.float()).abs().max()),
                    times={},
                )
                for name, fn in [("slices_binary", baseline), ("packed_reader", lambda: packed_silu(x))]:
                    fn()
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    out = fn()
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    trace_allocation_tracker.acknowledge_corruptible(out)
                    samples = []
                    for _ in range(12):
                        start = time.perf_counter()
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        samples.append((time.perf_counter() - start) * 1000)
                    replay = ttnn.to_torch(out)
                    ref = expected if name == "slices_binary" else actual
                    row[name + "_replay_exact"] = torch.equal(replay, ref)
                    row[name + "_replay_max_abs"] = float((replay.float() - ref.float()).abs().max())
                    row["times"][name] = statistics.median(samples[2:])
                    ttnn.release_trace(mesh, trace)
                    trace = None
                report["rows"].append(row)
                print(row, flush=True)
        report["passed"] = all(
            r["pcc"] >= 0.995 and r["slices_binary_replay_exact"] and r["packed_reader_replay_exact"]
            for r in report["rows"]
        )
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
