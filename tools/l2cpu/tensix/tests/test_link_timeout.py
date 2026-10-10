#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""Forced wait timeout: the responder is stopped, then a trace of [notify, wait] is replayed N times. Every wait must
hit its bound, write the timeout status word and return, so the trace completes (no device hang) in about
N x timeout. Checks: the replays finish, wait status = 0xDEAD0000 | (last req & 0xFFFF), done_seq did not move,
wall time per iteration ~ the bound (reported)."""
import argparse
import sys
import time

from link_setup import ops, open_link, stop_responder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--timeout-us", type=int, default=ops.WAIT_TIMEOUT_US)
    a = ap.parse_args()
    import ttnn

    dev, hw, link = open_link()
    pn, pw = link.notify_program(), link.wait_program(timeout_us=a.timeout_us)
    link.run(pn)
    link.run(pw)
    ttnn.synchronize_device(dev)  # warm-up while the responder still serves (compiles both programs)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    link.run(pn)
    link.run(pw)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    stop_responder(hw, link)
    t0 = time.time()
    while hw.pa_read32(link.base + ops.LINK_OFF_RESP_STATUS) != 0:  # responder clears its magic when it stops
        if time.time() - t0 > 2.0:
            print("FAIL: responder did not stop")
            sys.exit(1)
    done0 = hw.pa_read32(link.base + ops.LINK_OFF_DONE_SEQ)
    t0 = time.perf_counter()
    for _ in range(a.iters):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    dt = time.perf_counter() - t0
    req = hw.pa_read32(link.base + ops.LINK_OFF_REQ_SEQ)
    done = hw.pa_read32(link.base + ops.LINK_OFF_DONE_SEQ)
    st = hw.pa_read32(link.base + ops.LINK_OFF_WAIT_STATUS)
    ok = st == (ops.WAIT_STATUS_TIMEOUT | (req & 0xFFFF)) and done == done0
    print(
        f"forced timeout: {a.iters} replays completed in {dt:.3f} s = {dt / a.iters * 1e3:.1f} ms per iteration "
        f"(bound {a.timeout_us / 1e3:.1f} ms); wait status 0x{st:08x}, req {req}, done {done} (unchanged: {done == done0})"
    )
    print("PASS" if ok else "FAIL", flush=True)
    ttnn.release_trace(dev, tid)
    ttnn.close_device(dev)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
