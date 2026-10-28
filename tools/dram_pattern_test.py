#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone DRAM pattern test through tt-exalens (no tt-metal process, no model).

For each pattern, writes it over [addr, addr+len) of one DRAM bank on one chip, reads it back at once, then polls
every --period seconds for --seconds and reports every byte that ever differed from what was written (first time,
bits ever cleared / set, final value). Run with no other process holding the chips; it overwrites that DRAM range.

Defaults target the cell behind #62682: bh-glx-120-b06u08 tray 2 / ASIC 5 (exalens/UMD device 12), DRAM bank 7,
around 0x1400A2D0. Measured 2026-10-28 on b06u08: `ff` -> 0x1400a2d0 reads 9f (bits 5,6 lost) within 1 s, `55`
-> 6 bytes lose bits, all at byte 16 of a 64-byte block; on a healthy galaxy every pattern reports 0 bytes.

  python3 tools/dram_pattern_test.py                                  # defaults: dev 12, bank 7, 4 KB @ 0x1400A000
  python3 tools/dram_pattern_test.py --dev 12 --bank 7 --addr 0x13F80000 --len 1048576 --seconds 60 ff 55
"""
import argparse
import sys
import time

import numpy as np
from ttexalens import tt_exalens_init
from ttexalens import tt_exalens_lib as lib


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="*", default=["ff", "00", "aa", "55"], help="hex byte, 'rand' or 'orig'")
    ap.add_argument("--dev", type=int, default=12, help="exalens/UMD device id")
    ap.add_argument("--bank", type=int, default=7, help="DRAM bank (channel)")
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=0x1400A000, help="bank-local start address")
    ap.add_argument("--len", type=int, default=4096, help="bytes")
    ap.add_argument("--seconds", type=float, default=45, help="observation time per pattern")
    ap.add_argument("--period", type=float, default=0.2, help="poll period")
    a = ap.parse_args()

    ctx = tt_exalens_init.init_ttexalens()
    # Endpoint 1: exalens' "ch<n>" shorthand is endpoint 0, which reads zeros for banks 0 and 4-7 on a live chip.
    loc = f"d{a.bank},1"

    def rd():
        return np.frombuffer(
            bytes(lib.read_from_device(loc, a.addr, device_id=a.dev, num_bytes=a.len, context=ctx)), np.uint8
        )

    orig = rd().copy()
    print(f"dev {a.dev} bank {a.bank} [{a.addr:#x}, {a.addr + a.len:#x}) {a.seconds:.0f} s per pattern")
    total = 0
    for pat in a.patterns:
        if pat == "orig":
            w = orig.copy()
        elif pat == "rand":
            w = np.random.default_rng(1).integers(0, 256, a.len, dtype=np.uint8)
        else:
            w = np.full(a.len, int(pat, 16), np.uint8)
        lib.write_to_device(loc, a.addr, w.tobytes(), device_id=a.dev, context=ctx)
        t0 = time.time()
        first = {}
        ever_clr = np.zeros(a.len, np.uint8)
        ever_set = np.zeros(a.len, np.uint8)
        reads = 0
        while True:
            cur = rd()
            dt = time.time() - t0
            reads += 1
            for i in np.nonzero(cur != w)[0]:
                first.setdefault(int(i), dt)
            ever_clr |= w & ~cur
            ever_set |= ~w & cur
            if dt > a.seconds:
                break
            time.sleep(a.period)
        print(f"pattern {pat}: {reads} reads over {dt:.1f} s, {len(first)} bytes ever differed")
        for i in sorted(first):
            print(
                f"   {a.addr + i:#x} wrote {w[i]:02x} final {cur[i]:02x} first@{first[i] * 1000:.0f}ms "
                f"ever-cleared {ever_clr[i]:08b} ever-set {ever_set[i]:08b}"
            )
        sys.stdout.flush()
        total += len(first)
    print("FAIL" if total else "PASS", f"({total} failing byte observations)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
