#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-card check of the ethernet counters with nothing running: are the queue registers readable and monotonic,
how fast do they move at idle, does the firmware's L1 block refresh, and do its 64-bit queue copies agree with the
queue registers.

  python eth_idle_drift.py --seconds 20

Run on an idle card that nothing else has open (a Blackhole card with a connected or looped port, or any chip with
connected links). Reads only. Prints per ethernet core with a link up (Blackhole) or every core (Wormhole):

  - READABLE: no register read returned 0xFFFFFFFF
  - MONOTONIC: no 32-bit register went backwards between samples
  - idle rate: counts per second of every queue register that moved
  - REFRESH: Blackhole, FW_LINK_POLL_ALIVE_COUNT delta over the run (0 means the L1 block is stale and the other FW_
    deltas mean nothing)
  - COPIES: Blackhole, FW_TXQ_RESEND_CNT and FW_RXQ_PKT_DROP deltas against TXQ_RESEND_CNT and RXQ_PKT_DROP_CNT

An idle link that shows no movement of TXQ_WORD_CNT at all has no keepalive, which is a fact worth knowing for the idle
correction of the runtime.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--device", default=c.default_device_path())
    args = p.parse_args()
    if c.card_busy(args.device):
        c.fail(f"{args.device} is open by another process; not running")

    cards = c.Cards()
    ok = True
    for device in cards.devices:
        arch = cards.arch(device)
        for core in cards.eth_cores(device):
            up = c.link_up(arch, core)
            if up is False:
                continue
            samples = []
            t0 = time.monotonic()
            while True:
                samples.append((time.monotonic(), c.snapshot(arch, core)))
                if time.monotonic() - t0 >= args.seconds:
                    break
                time.sleep(args.interval)
            first, last = samples[0], samples[-1]
            span = last[0] - first[0]
            readable = all(
                v != c.FAILED_READ
                for _, snap in samples
                for k, v in snap.items()
                if not k[0].startswith("FW_")
            )
            monotonic = True
            for (_, a), (_, b) in zip(samples, samples[1:]):
                for key, value in b.items():
                    if value < a[key] and not (
                        value < 1 << 8 and a[key] > (1 << 32) - (1 << 8)
                    ):
                        monotonic = False
            tag = f"device {device.id} core {core.to_str('noc0')}"
            print(f"{tag}: READABLE {readable} MONOTONIC {monotonic}")
            ok = ok and readable and monotonic
            for key, value in sorted(last[1].items()):
                delta = (value - first[1][key]) & (
                    0xFFFFFFFF if not key[0].startswith("FW_") else (1 << 64) - 1
                )
                if delta and not key[0].startswith("FW_"):
                    print(f"  {key[0]} q{key[1]}: {delta / span:.1f}/s")
            if arch == "blackhole":
                alive = (
                    last[1][("FW_LINK_POLL_ALIVE_COUNT", -1)]
                    - first[1][("FW_LINK_POLL_ALIVE_COUNT", -1)]
                )
                print(f"  REFRESH: FW_LINK_POLL_ALIVE_COUNT +{alive} in {span:.1f} s")
                for fw, reg in (
                    ("FW_TXQ_RESEND_CNT", "TXQ_RESEND_CNT"),
                    ("FW_RXQ_PKT_DROP", "RXQ_PKT_DROP_CNT"),
                ):
                    for q in range(3):
                        d_fw = last[1][(fw, q)] - first[1][(fw, q)]
                        d_reg = (last[1][(reg, q)] - first[1][(reg, q)]) & 0xFFFFFFFF
                        print(
                            f"  COPIES {fw} q{q}: firmware +{d_fw}, register +{d_reg}"
                        )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
