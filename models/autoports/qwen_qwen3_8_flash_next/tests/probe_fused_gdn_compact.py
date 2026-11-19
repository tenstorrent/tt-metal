# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prototype: fold GDN head replication and padded token layout into device I/O."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import fused_recurrent_gated_delta_rule


def compact_recurrent_gated_delta_rule(*args, **kwargs):
    return fused_recurrent_gated_delta_rule(*args, compact_io=True, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    torch.set_num_threads(8)
    torch.manual_seed(9173)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = dict(rows=[], passed=False)
    trace = None
    try:

        def upload(x):
            return ttnn.from_torch(x, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

        for n in (1, 2, 7, 16, 17, 31, 32):
            q, k = [upload(torch.randn(1, n, 16, 128) * 0.05) for _ in range(2)]
            v = upload(torch.randn(1, n, 48, 128) * 0.02)
            g = upload(-torch.rand(1, n, 48) * 0.2)
            beta = upload(torch.rand(1, n, 48))
            initial = upload(torch.randn(1, 48, 128, 128) * 0.01)
            args = (q, k, v, g, beta)
            expected = [ttnn.to_torch(x) for x in fused_recurrent_gated_delta_rule(*args, initial_state=initial)]
            actual = [ttnn.to_torch(x) for x in compact_recurrent_gated_delta_rule(*args, initial_state=initial)]
            row = dict(
                tokens=n,
                exact=[torch.equal(x, y) for x, y in zip(expected, actual)],
                max_abs=[float((x - y).abs().max()) for x, y in zip(expected, actual)],
                times={},
            )
            for name, fn in [
                ("expanded", fused_recurrent_gated_delta_rule),
                ("compact", compact_recurrent_gated_delta_rule),
            ]:
                trace = ttnn.begin_trace_capture(device, cq_id=0)
                outputs = fn(*args, initial_state=initial)
                ttnn.end_trace_capture(device, trace, cq_id=0)
                for output in outputs:
                    trace_allocation_tracker.acknowledge_corruptible(output)
                times = []
                for _ in range(10):
                    start = time.perf_counter()
                    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                    times.append((time.perf_counter() - start) * 1000)
                replay = [ttnn.to_torch(x) for x in outputs]
                assert all(torch.equal(x, y) for x, y in zip(replay, expected if name == "expanded" else actual))
                row["times"][name] = statistics.median(times[2:])
                ttnn.release_trace(device, trace)
                trace = None
            report["rows"].append(row)
            print(row, flush=True)
        report["passed"] = all(all(r["exact"]) for r in report["rows"])
    finally:
        if trace is not None:
            ttnn.release_trace(device, trace)
        ttnn.close_mesh_device(device)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
