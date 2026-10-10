#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""notify and wait as two separate generic_op programs captured in one trace and replayed N times with
execute_trace(blocking=False) back to back (the host only enqueues). The wait program copies the reply line into
an output tensor. Checks: req_seq == done_seq == start + N, wait status 0, output == reply of the last request.
Also reports the per-program cost in a trace (100 notify programs)."""
import argparse
import sys
import time

from link_setup import ops, open_link, stop_responder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=10_000)
    a = ap.parse_args()
    import torch
    import ttnn

    dev, hw, link = open_link()
    out = ttnn.from_torch(
        torch.zeros(1, 1, 1, 32, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pn, pw = link.notify_program(), link.wait_program(out=out, n_words=16)
    link.run(pn)
    link.run(pw, out=out)
    ttnn.synchronize_device(dev)  # compile + warm the program cache before capture
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    link.run(pn)
    link.run(pw, out=out)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    r0 = hw.pa_read32(link.base + ops.LINK_OFF_REQ_SEQ)
    t0 = time.perf_counter()
    for _ in range(a.iters):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    dt = time.perf_counter() - t0
    req, done = hw.pa_read32(link.base + ops.LINK_OFF_REQ_SEQ), hw.pa_read32(link.base + ops.LINK_OFF_DONE_SEQ)
    st = hw.pa_read32(link.base + ops.LINK_OFF_WAIT_STATUS)
    got = ttnn.to_torch(out).flatten().tolist()[:16]
    exp = [(req * 64 + i) & 0xFFFFFFFF for i in range(16)]
    ok = req == done == r0 + a.iters and st == 0 and got == exp
    print(
        f"trace notify + wait: {a.iters} replays, {dt / a.iters * 1e6:.2f} us per iteration; req {r0} -> {req}, done {done}, "
        f"wait status 0x{st:x}, output == last reply: {got == exp}"
    )
    ttnn.release_trace(dev, tid)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(100):
        link.run(pn)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(20):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    print(
        f"program launch + notify kernel inside a trace: {(time.perf_counter() - t0) / 2000 * 1e6:.2f} us per program"
    )
    ttnn.release_trace(dev, tid)
    stop_responder(hw, link)
    print("PASS" if ok else "FAIL", flush=True)
    ttnn.close_device(dev)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
