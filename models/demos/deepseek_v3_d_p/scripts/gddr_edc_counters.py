#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-instance GDDR EDC counters read straight from the Blackhole memory-controller registers.

These are the full-width hardware counters, not the saturating u8 copy ARC telemetry exposes,
and they are clearable without a DRAM retrain. Reached over NoC through the AXI window on
translated DRAM coords. Read-only, so it is safe against a live hang.

One line per instance on stdout, greppable as "dev <n> inst <n>:".
"""

import argparse

import ttnn

AXI_WINDOW = 0x100_00000000
MC_BASE3 = 0xFC104800
REGS = {"corr_wr": 0x168, "corr_rd": 0x16C, "uncorr_wr": 0x170, "uncorr_rd": 0x174}
ADDR_REGS = {"rd_err_addr": 0x1A4, "wr_err_addr": 0x1A0}
CLEAR_REG = 0x160
NUM_INSTANCES = 8


def instance_core(instance):
    return 17 + (instance // 4), 12 + 3 * (instance % 4)


def read_register(device, instance, offset):
    x, y = instance_core(instance)
    return int.from_bytes(ttnn._ttnn.cluster.read_from_core(device, x, y, AXI_WINDOW + MC_BASE3 + offset, 4), "little")


def clear_counters(device, instance):
    x, y = instance_core(instance)
    for value in (1, 0):
        ttnn._ttnn.cluster.write_to_core_immediate(
            device, x, y, AXI_WINDOW + MC_BASE3 + CLEAR_REG, value.to_bytes(4, "little")
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("devices", nargs="*", type=int, default=list(range(32)))
    parser.add_argument("--all", action="store_true", help="print clean instances too")
    parser.add_argument("--clear", action="store_true", help="clear the counters after reading")
    args = parser.parse_args()

    dirty = 0
    for device in args.devices:
        for instance in range(NUM_INSTANCES):
            try:
                values = {name: read_register(device, instance, off) for name, off in (REGS | ADDR_REGS).items()}
            except Exception as e:
                print(f"dev {device:2d} inst {instance}: read failed ({type(e).__name__})")
                continue
            bad = any(values[name] for name in REGS)
            dirty += bad
            if args.all or bad:
                print(
                    f"dev {device:2d} inst {instance}: "
                    + " ".join(f"{name}={values[name]}" for name in REGS)
                    + f" rd_addr=0x{values['rd_err_addr']:08x} wr_addr=0x{values['wr_err_addr']:08x}"
                    + ("   *** EDC ***" if bad else "")
                )
            if args.clear:
                clear_counters(device, instance)

    print(f"\n{len(args.devices) * NUM_INSTANCES} instances checked, {dirty} with EDC errors")


if __name__ == "__main__":
    main()
