# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Do several traces of the *same program* keep their own runtime arguments?

`TracedDecodeStepInPlace` rests on this and nothing else. It captures 32 traces that
differ only in two integers -- `update_cache`'s write row and `slice`'s start column --
and neither integer is part of a program-cache key: `UpdateKVCacheOperation::compute_
program_hash` hashes the op type and the tensors, not `update_idx`. So all 32 traces
share one compiled program, and each is supposed to carry its own copy of the dispatch
commands that set that program's runtime args.

"Supposed to" is doing real work in that sentence. If instead a trace referenced a
single shared runtime-arg region, capturing trace 31 would silently rewrite what trace
0 replays, and every slot would behave like the last one captured. In the full model
that surfaces as attention drifting a few rows out -- fluent, plausible, wrong output
several hundred tokens later, which is the worst way to find out.

So: capture three traces at three different indices, then replay them **one at a time
from a freshly zeroed buffer** and ask which row actually moved. Separate state per
replay is the point; sharing it is how a probe of this kind talks itself into a
false pass.

    python models/demos/cosyvoice/scripts/probe_multi_trace_args.py
"""
from __future__ import annotations

import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

H, DK, W = 16, 64, 288
ROWS = (256, 269, 287)  # tile-interior, mid-tile and last row of the scratch zone


def written_rows(t):
    """Which rows of a [1, h, W, dk] tensor are non-zero, per row index."""
    back = ttnn.to_torch(t).float()
    per_row = back.abs().sum(dim=(0, 1, 3))
    return (per_row > 0).nonzero().flatten().tolist()


def main() -> int:
    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=402653184)
    try:
        mk = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # noqa: E731
        buf = mk(torch.zeros(1, H, W, DK))
        tok = mk(torch.ones(1, H, 1, DK))
        zero = mk(torch.zeros(1, H, W, DK))

        # --- capture one trace per row, all sharing the single update_cache program
        traces = {}
        for row in ROWS:
            ttnn.update_cache(buf, tok, row)  # warm the runtime-arg variant
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            try:
                ttnn.update_cache(buf, tok, row)
            finally:
                ttnn.end_trace_capture(device, tid, cq_id=0)
            traces[row] = tid
        ttnn.synchronize_device(device)

        print(f"\n  == update_cache: 3 traces, rows {ROWS} ==")
        ok = True
        for row, tid in traces.items():
            ttnn.copy(zero, buf)  # fresh state per replay -- see the module docstring
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            got = written_rows(buf)
            good = got == [row]
            ok &= good
            print(f"    trace for row {row:3d} wrote {got}  {'ok' if good else '<-- WRONG'}")
            ttnn.release_trace(device, tid)

        # --- the same question for a slice whose start offset varies
        src = mk(torch.arange(W * 4, dtype=torch.float32).reshape(1, 1, 1, W * 4).repeat(1, H, 1, 1))
        dst = mk(torch.zeros(1, H, 1, W))
        offs = (0, 17, 31)
        straces = {}
        for off in offs:
            # Warm the *whole* body, not just the op under test. The first attempt
            # warmed the slice and left the `copy` behind it cold, and capture failed
            # with "Cannot load new binaries during trace capture" -- which is the
            # reassuring failure mode: an unwarmed program raises at capture rather
            # than quietly recording something else.
            s = ttnn.slice(src, [0, 0, 0, off], [1, H, 1, off + W])
            ttnn.copy(s, dst)
            ttnn.deallocate(s)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            try:
                s = ttnn.slice(src, [0, 0, 0, off], [1, H, 1, off + W])
                ttnn.copy(s, dst)
                ttnn.deallocate(s)
            finally:
                ttnn.end_trace_capture(device, tid, cq_id=0)
            straces[off] = tid
        ttnn.synchronize_device(device)

        print(f"\n  == slice: 3 traces, start offsets {offs} ==")
        for off, tid in straces.items():
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            first = float(ttnn.to_torch(dst).float()[0, 0, 0, 0])
            good = abs(first - off) < 0.5
            ok &= good
            print(f"    trace for offset {off:2d} starts at {first:6.1f}  {'ok' if good else '<-- WRONG'}")
            ttnn.release_trace(device, tid)

        print(f"\n  verdict: {'per-trace runtime args hold' if ok else 'TRACES SHARE ARGS -- design is not viable'}")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
