#!/usr/bin/env python3
"""Force Blackhole's AICLK to a fixed frequency, and read back what the chip reports.

Blackhole idles at 800 MHz and ramps to 1350 MHz under load, while Wormhole runs at 1000 MHz.
Comparing energy per FLOP between the two boards at their own default clocks therefore mixes an
architecture difference with a 1.35x frequency difference. Pinning Blackhole to 1000 MHz makes
the comparison iso-clock, so what is left is the architecture: NoC width, core count, voltage.

Uses the ARC FORCE_AICLK message (0x33) through tt_umd's TTDevice.arc_msg. The force has to be
re-applied after anything that resets the chip, and possibly after a tt-metal process opens the
device -- tt-metal asks the ARC for the busy power state on open, which is what normally ramps
the clock to 1350. Whether the force survives that is exactly what --verify checks, so always
confirm against a running workload rather than trusting the value read at idle.
"""
import argparse
import sys
import time

import tt_umd

FORCE_AICLK = 0x33          # blackhole::ArcMessageType::FORCE_AICLK
AICLK_GO_BUSY = 0x52
# Passing 0 releases the override and returns the chip to firmware-managed clocking.
RELEASE = 0


def open_device(device_id: int):
    infos = tt_umd.PCIDevice.enumerate_devices()
    if device_id not in infos:
        raise SystemExit(f"device {device_id} not present; found {sorted(infos)}")
    dev = tt_umd.TTDevice.create(device_id)
    # The telemetry reader and the ARC messenger are both unavailable until this runs.
    dev.init_tt_device()
    return dev


def read_aiclk(dev) -> int:
    r = dev.get_arc_telemetry_reader()
    return r.read_entry(tt_umd.TelemetryTag.AICLK)


def read_limits(dev):
    r = dev.get_arc_telemetry_reader()
    out = {}
    for name in ("AICLK", "AICLK_LIMIT_MAX", "AICLK_ARB_MAX", "AICLK_ARB_MIN"):
        tag = getattr(tt_umd.TelemetryTag, name)
        try:
            out[name] = r.read_entry(tag) if r.is_entry_available(tag) else None
        except Exception as e:
            out[name] = f"err: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mhz", type=int, nargs="?", default=None,
                    help="Frequency to force, in MHz. Omit to only report. 0 releases the force.")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--busy", action="store_true",
                    help="Send AICLK_GO_BUSY first, so the reading is not the idle clock.")
    ap.add_argument("--watch", type=float, default=0.0,
                    help="Poll AICLK for this many seconds instead of exiting, to see what the "
                         "clock does while a workload runs in another process.")
    args = ap.parse_args()

    dev = open_device(args.device_id)
    print(f"arch: {dev.get_arch()}")
    print("before:", read_limits(dev))

    if args.busy:
        rc = dev.arc_msg(AICLK_GO_BUSY, True, [], 5000)
        print(f"AICLK_GO_BUSY -> {rc}")
        time.sleep(0.5)
        print("after busy:", read_limits(dev))

    if args.mhz is not None:
        rc = dev.arc_msg(FORCE_AICLK, True, [args.mhz], 5000)
        print(f"FORCE_AICLK({args.mhz}) -> exit_code={rc[0]} ret={rc[1:]}")
        if rc[0] != 0:
            print("WARNING: non-zero exit code; the force may not have been applied",
                  file=sys.stderr)
        time.sleep(0.5)
        print("after force:", read_limits(dev))

    if args.watch > 0:
        t0 = time.time()
        seen = []
        while time.time() - t0 < args.watch:
            seen.append(read_aiclk(dev))
            time.sleep(0.1)
        print(f"watched {len(seen)} samples over {args.watch:.1f}s: "
              f"min={min(seen)} max={max(seen)} mode={max(set(seen), key=seen.count)}")
        # A workload that ramps the clock back up shows as a max well above the forced value.
        print(f"distinct values: {sorted(set(seen))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
