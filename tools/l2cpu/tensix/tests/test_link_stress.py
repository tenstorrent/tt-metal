#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""Ordering stress of the Tensix <-> L2CPU link: one Tensix core runs N rounds of notify -> x280 responder -> wait ->
read the reply line; every reply word must carry the round's request number (0 stale reads allowed).
Exit code 0 only if all rounds completed with 0 stale replies and 0 timeouts."""
import argparse
import struct
import sys
import time

from link_setup import ops, open_link, stop_responder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=100_000)
    a = ap.parse_args()
    import ttnn

    dev, hw, link = open_link()
    hw.pa_write(link.base + ops.LINK_OFF_DIAG, bytes(64))
    prog = link.stress_program(a.rounds)
    t0 = time.perf_counter()
    link.run(prog)
    ttnn.synchronize_device(dev)
    wall = time.perf_counter() - t0
    k, stale, tmo, ticks, worst, first_bad = struct.unpack("<6I", hw.pa_read(link.base + ops.LINK_OFF_DIAG, 24))
    served = hw.pa_read64(link.base + ops.LINK_OFF_RESP_STATUS + 8)
    print(
        f"link stress: {k}/{a.rounds} rounds, stale {stale}, timeouts {tmo}, first bad {first_bad}, responder served "
        f"{served}; {wall / max(k, 1) * 1e6:.2f} us per round trip (wall, incl. one program launch); worst round "
        f"{worst} device ticks"
    )
    stop_responder(hw, link)
    ok = k == a.rounds and stale == 0 and tmo == 0
    print("PASS" if ok else "FAIL", flush=True)
    ttnn.close_device(dev)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
