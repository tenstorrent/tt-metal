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
    l2cpuctl.py --tile 0,1 --region PA0,PA1 start IMAGE                    # several tiles, one release
--region is the x280 PA of the region (l2cpu_boot.h), e.g. 0x4000_3000_0000 + a 64 KiB aligned offset in the
tile's local DRAM that nothing else uses. --tile (default 0) selects the L2CPU tile: 0 (8,3) DRAM bank 5, 1 (8,9)
bank 6, 2 (8,5) bank 7, 3 (8,7) bank 7 (tiles 2 and 3 share bank 7: give them non-overlapping regions). With a list
of tiles every command runs on each of them (start: one release for all) and prints one result per tile.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "host"))

from l2cpu import L2cpuCtl, L2cpuHw, layout as L, make_backend, start_tiles  # noqa: E402


def int_list(s):
    return [int(x, 0) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", type=int_list, required=True, help="region PA, one per tile")
    ap.add_argument("--tile", type=int_list, default=[0], help="L2CPU tile(s) 0-3 (default 0)")
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
    if len(a.region) != len(a.tile):
        ap.error("--region needs one PA per --tile")
    backend = make_backend(a.backend)
    ctls = [L2cpuCtl(L2cpuHw(backend, tile=t, guard=True), r, mhz=a.mhz) for t, r in zip(a.tile, a.region)]
    slot = {"A": L.L2CPU_SLOT_A, "B": L.L2CPU_SLOT_B, None: None}
    if a.cmd == "start":
        infos = start_tiles(ctls, open(a.image, "rb").read(), slot=slot[a.slot])
        outs = [{k: v for k, v in i.items() if k != "hw"} for i in infos]
        print(
            json.dumps(
                outs if len(outs) > 1 else outs[0], indent=1, default=lambda v: hex(v) if isinstance(v, int) else str(v)
            )
        )
        return 0
    if a.cmd == "log":
        for c in ctls:
            print(c.log_text()[0], end="")
        return 0
    outs = [run(c, a, slot) for c in ctls]
    print(
        json.dumps(
            outs if len(outs) > 1 else outs[0], indent=1, default=lambda v: hex(v) if isinstance(v, int) else str(v)
        )
    )
    return 0


def run(ctl, a, slot):
    if a.cmd == "stop":
        recs, used = ctl.stop()
        out = dict(levels=used, records=recs)
    elif a.cmd == "restart":
        img = open(a.image, "rb").read() if a.image else None
        out = ctl.restart(img, warm=not a.cold, slot=slot[a.slot])
    elif a.cmd == "status":
        ctl.ensure_clock()
        out = ctl.status()
    else:
        t_end = time.time() + a.seconds
        while time.time() < t_end:
            time.sleep(min(1.0, t_end - time.time()))
        out = ctl.status()
    out["tile"] = ctl.hw.tile
    return out


if __name__ == "__main__":
    sys.exit(main())
