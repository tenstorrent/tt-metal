# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Can a KV slot be written in place, at an index a trace can bake?

This is the linchpin for the tile-aligned cache. The shift costs 78 + 207 us a tensor
a layer purely because slicing from row 1 and appending one row re-tiles a 32-row-tiled
buffer. The way out is to stop shifting: keep a 288-row buffer, write the new token at
row 256+i with `update_cache`, and let a 32-row tile-aligned shift happen once every 32
steps instead of a 1-row shift every step.

That only works if the write is (a) in place, (b) cheap, and (c) expressible with an
index that a trace can capture. If the index has to be a device tensor read at replay
time, one trace suffices; if it is a Python int baked at capture, the design needs 32
traces, one per sub-step. Either is workable, and which one decides the shape of the
implementation -- so it is worth knowing before writing any of it.

    python models/demos/cosyvoice/scripts/probe_update_cache.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

H, DK, W = 16, 64, 288
REPS = 40
CHAIN = 16


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main() -> int:
    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=402653184)
    try:
        print("\n  == candidate entry points ==")
        for path in (
            "update_cache",
            "fill_cache",
            "kv_cache.update_cache_for_token_",
            "experimental.paged_update_cache",
        ):
            obj = ttnn
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            print(f"    {path:44s} {'present' if obj is not None else 'MISSING'}")

        mk = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # noqa: E731

        print("\n  == correctness: write row 260, read it back ==")
        wrote = None
        # The op's expected layouts are not documented in the python binding, so try the
        # shapes tt-metal's own LLM demos use as well as this model's.
        variants = [
            ("cache[1,h,W,dk] tok[1,1,h,dk]", (1, H, W, DK), (1, 1, H, DK)),
            ("cache[1,h,W,dk] tok[1,h,1,dk]", (1, H, W, DK), (1, H, 1, DK)),
            ("cache[h,1,W,dk] tok[1,h,1,dk]", (H, 1, W, DK), (1, H, 1, DK)),
            ("cache[1,h,W,dk] tok[1,1,1,dk]", (1, H, W, DK), (1, 1, 1, DK)),
        ]
        for label, cshape, tshape in variants:
            b2 = mk(torch.zeros(*cshape))
            t2 = mk(torch.randn(*tshape))
            try:
                ttnn.update_cache(b2, t2, 260)
                back = ttnn.to_torch(b2).float()
                axis = 2 if len(cshape) == 4 else 1
                rows = back.abs().sum(dim=tuple(i for i in range(back.dim()) if i != axis)) > 0
                print(f"    {label:34s} OK, rows {rows.nonzero().flatten().tolist()[:5]}")
                wrote = label
            except Exception as e:  # noqa: BLE001
                print(f"    {label:34s} {str(e)[:150]}")
            ttnn.deallocate(b2)
            ttnn.deallocate(t2)

        buf = mk(torch.zeros(1, H, W, DK))
        tok = mk(torch.randn(1, H, 1, DK))  # the variant that passed above

        if wrote is None:
            print("\n  No in-place write available -- the design needs a different primitive.")
            return 0

        print("\n  == cost, traced, vs the shift it replaces ==")

        def timed(make):
            for _ in range(2):
                for _ in range(CHAIN):
                    make()
            ttnn.synchronize_device(device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(CHAIN):
                make()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(REPS):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            us = (time.perf_counter() - t0) * 1e6 / REPS / CHAIN
            ttnn.release_trace(device, tid)
            return us

        try:
            print(
                f"    update_cache at a baked index          {timed(lambda: ttnn.update_cache(buf, tok, 260)):8.1f} us"
            )
        except Exception as e:  # noqa: BLE001
            print(f"    update_cache in a trace                RAISED {type(e).__name__}: {str(e)[:70]}")

        big = mk(torch.randn(1, H, 256, DK))
        one = mk(torch.randn(1, H, 1, DK))

        def shift():
            t = ttnn.slice(big, [0, 0, 1, 0], [1, H, 256, DK])
            o = ttnn.concat([t, one], dim=2)
            ttnn.deallocate(t)
            ttnn.deallocate(o)

        print(f"    the 1-row shift it replaces            {timed(shift):8.1f} us")

        print("\n  == is the index bakeable in a trace, or does it need a tensor? ==")
        print("    (a baked int means 32 traces, one per sub-step; a device tensor means one)")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
