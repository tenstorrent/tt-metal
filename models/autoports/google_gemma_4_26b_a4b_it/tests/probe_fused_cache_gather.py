# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare cache embedding, flattened gather, and page-axis gather movement."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=("all", "embedding", "tiled_gather", "page_axis_gather", "page_axis_gather_dynamic"),
        default="all",
    )
    parser.add_argument("--physical-pages", type=int, default=160)
    parser.add_argument("--selected-pages", type=int, default=33)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timing-samples", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.selected_pages <= args.physical_pages:
        parser.error("selected-pages must be between 1 and physical-pages")
    if args.kv_heads < 1 or args.head_dim < 32 or args.head_dim % 32:
        parser.error("kv-heads must be positive and head-dim must be a positive multiple of 32")
    if args.repeats < 1 or args.timing_samples < 1:
        parser.error("repeats and timing-samples must be positive")

    torch.manual_seed(1)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        # Defaults preserve the original sliding control's pool, pages and data.
        host = torch.randn(args.physical_pages, args.kv_heads, 32, args.head_dim).bfloat16()
        cache = ttnn.from_torch(host, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ids = torch.randperm(args.physical_pages)[: args.selected_pages]
        rows_per_page = args.kv_heads * 32
        rows = (ids[:, None] * rows_per_page + torch.arange(rows_per_page)[None, :]).flatten().int()
        expected = host[ids].reshape(-1, args.head_dim)
        flat = ttnn.reshape(cache, (args.physical_pages * rows_per_page, args.head_dim))

        def upload_index(value, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value.contiguous(), device=mesh, dtype=ttnn.uint32, layout=layout)

        candidates = []
        if args.candidate in ("all", "embedding"):
            rm = upload_index(rows[None], ttnn.ROW_MAJOR_LAYOUT)
            candidates.append(("embedding", lambda: ttnn.embedding(rm, flat, layout=ttnn.TILE_LAYOUT), False))
        if args.candidate in ("all", "tiled_gather"):
            tiled = upload_index(rows[:, None].expand(-1, args.head_dim))
            candidates.append(("tiled_gather", lambda: ttnn.gather(flat, dim=0, index=tiled), False))
        if args.candidate in ("all", "page_axis_gather"):
            page_index = upload_index(ids[:, None, None, None].expand(-1, args.kv_heads, 32, args.head_dim))
            candidates.append(("page_axis_gather", lambda: ttnn.gather(cache, dim=0, index=page_index), False))
        if args.candidate in ("all", "page_axis_gather_dynamic"):
            page_ids = upload_index(ids[:, None, None, None])

            def page_axis_gather_dynamic():
                index = ttnn.repeat(page_ids, (1, args.kv_heads, 32, args.head_dim))
                return ttnn.gather(cache, dim=0, index=index)

            candidates.append(("page_axis_gather_dynamic", page_axis_gather_dynamic, True))

        report = []
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for name, fn, timed_index_expansion in candidates:
            output = fn()
            actual = ttnn.to_torch(output).reshape(-1, args.head_dim)
            exact_equal = torch.equal(actual, expected)
            assert exact_equal, {"candidate": name, "max_abs": float((actual.float() - expected.float()).abs().max())}
            output.deallocate(True)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            output = fn()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            try:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                times = []
                for _ in range(args.timing_samples):
                    start = time.perf_counter_ns()
                    for _ in range(args.repeats):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    times.append((time.perf_counter_ns() - start) / (args.repeats * 1000))
                replay_equal = torch.equal(ttnn.to_torch(output).reshape(-1, args.head_dim), expected)
                assert replay_equal, {"candidate": name, "trace_replay_equal": False}
            finally:
                ttnn.release_trace(mesh, trace)
                output.deallocate(True)
            report.append(
                dict(
                    candidate=name,
                    exact_equal=exact_equal,
                    trace_replay_equal=replay_equal,
                    timed_index_expansion=timed_index_expansion,
                    traced_host_us=statistics.median(times),
                    traced_host_us_samples=times,
                )
            )
            args.output.write_text(
                json.dumps(
                    dict(
                        cache_shape=list(host.shape),
                        selected_pages=args.selected_pages,
                        page_ids=ids.tolist(),
                        repeats=args.repeats,
                        timing_samples=args.timing_samples,
                        results=report,
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(report[-1], flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
