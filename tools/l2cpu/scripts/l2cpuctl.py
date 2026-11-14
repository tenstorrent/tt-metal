#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""l2cpuctl: command-line front end of the l2cpu host package (start / stop / restart / status / log / hold).

Every chip access goes through the clock guard. This process is itself a clock holder while it runs (a UMD open
sets the driver's L2CPU power flag); when it exits and no other process holds the chip, the L2CPU clock stops and
the harts freeze (state kept). Run it under the same lock as every other device user (scripts/l2cpu_run.sh).

    l2cpuctl.py --region PA status
    l2cpuctl.py --region PA start tools/l2cpu/fw/build/bh-irq/fw.bin      # fresh chip reset only
    l2cpuctl.py --region PA stop
    l2cpuctl.py --region PA restart [IMAGE] [--cold] [--slot A|B]
    l2cpuctl.py --region PA log
    l2cpuctl.py --region PA hold SECONDS                                   # keep the clock on
--region is the x280 PA of the region (l2cpu_boot.h), e.g. 0x4000_3000_0000 + a 64 KiB aligned offset in the
tile's local DRAM that nothing else uses.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "host"))

from l2cpu import L2cpuCtl, L2cpuHw, layout as L, make_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", type=lambda s: int(s, 0), required=True)
    ap.add_argument("--backend", default=None, help="umd (default) or ttnn")
    ap.add_argument("--mhz", type=int, default=1750)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("image")
    s.add_argument("--slot", default="A")
    sub.add_parser("stop")
    s = sub.add_parser("restart")
    s.add_argument("image", nargs="?")
    s.add_argument("--cold", action="store_true")
    s.add_argument("--slot", default=None)
    sub.add_parser("status")
    sub.add_parser("log")
    s = sub.add_parser("hold")
    s.add_argument("seconds", type=float)
    a = ap.parse_args()
    ctl = L2cpuCtl(L2cpuHw(make_backend(a.backend), guard=True), a.region, mhz=a.mhz)
    slot = {"A": L.L2CPU_SLOT_A, "B": L.L2CPU_SLOT_B, None: None}
    if a.cmd == "start":
        out = {k: v for k, v in ctl.start(open(a.image, "rb").read(), slot=slot[a.slot]).items() if k != "hw"}
    elif a.cmd == "stop":
        recs, used = ctl.stop()
        out = dict(levels=used, records=recs)
    elif a.cmd == "restart":
        img = open(a.image, "rb").read() if a.image else None
        out = ctl.restart(img, warm=not a.cold, slot=slot[a.slot])
    elif a.cmd == "status":
        ctl.ensure_clock()
        out = ctl.status()
    elif a.cmd == "log":
        print(ctl.log_text()[0], end="")
        return 0
    else:
        t_end = time.time() + a.seconds
        while time.time() < t_end:
            time.sleep(min(1.0, t_end - time.time()))
        out = ctl.status()
    print(json.dumps(out, indent=1, default=lambda v: hex(v) if isinstance(v, int) else str(v)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
