#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Usage:
    dump_l1_diff

Description:
    Raw L1 of chosen worker cores against a reference core on the same chip (the firmware is the same on every worker,
    so a diff over the firmware region is what was written there). TRIAGE_L1_DIFF="tx,ty/rx,ry;..." (logical
    coordinates), TRIAGE_L1_DIFF_END (bytes from 0, default 0x10000). Writes generated/moe_block/l1_diff/
    dev<N>_<tx>_<ty>_vs_<rx>_<ry>.bin (target, then reference) and l1_diff.txt: per device and pair, the differing 32-bit
    words in the firmware region (MEM_BRISC_FIRMWARE_BASE .. MEM_NOC_COUNTER_BASE) with both values, and the count elsewhere. Runs only when
    TRIAGE_L1_DIFF is set.

Owner:
    mstaletovic
"""

import os
import struct

from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_from_device
from triage import ScriptConfig, run_script

script_config = ScriptConfig(depends=["run_checks"])

FW_LO, FW_HI = 0x3BE0, 0x85E0  # (Blackhole: BRISC firmware .. the NOC counters - BRISC, NCRISC, TRISC0-2 firmware)


def run(args, context: Context):
    spec = os.environ.get("TRIAGE_L1_DIFF")
    if not spec:
        return
    end = int(os.environ.get("TRIAGE_L1_DIFF_END", "0x10000"), 0)
    d = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/moe_block/l1_diff")
    os.makedirs(d, exist_ok=True)
    pairs = [tuple(tuple(int(v) for v in c.split(",")) for c in p.split("/")) for p in spec.split(";") if p]
    out = open(os.path.join(d, "l1_diff.txt"), "w")
    for dev in context.devices.values():
        for (tx, ty), (rx, ry) in pairs:
            try:
                t = OnChipCoordinate(tx, ty, "logical", dev, "tensix")
                r = OnChipCoordinate(rx, ry, "logical", dev, "tensix")
                bt = bytes(read_from_device(t, 0, device_id=dev.id, num_bytes=end, context=context))
                br = bytes(read_from_device(r, 0, device_id=dev.id, num_bytes=end, context=context))
            except Exception as e:  # noqa: BLE001
                print(f"L1_DIFF dev {dev.id} ({tx},{ty}) vs ({rx},{ry}): read failed {e}", file=out)
                continue
            open(os.path.join(d, f"dev{dev.id}_{tx}_{ty}_vs_{rx}_{ry}.bin"), "wb").write(bt + br)
            wt = struct.unpack(f"<{end // 4}I", bt)
            wr = struct.unpack(f"<{end // 4}I", br)
            fw = [i for i in range(FW_LO // 4, FW_HI // 4) if wt[i] != wr[i]]
            other = sum(1 for i in range(end // 4) if wt[i] != wr[i]) - len(fw)
            print(
                f"L1_DIFF dev {dev.id} ({tx},{ty}) vs ({rx},{ry}): {len(fw)} words differ in the firmware, {other} "
                f"elsewhere below 0x{end:x}" + (f" - firmware 0x{4 * fw[0]:x}..0x{4 * fw[-1] + 4:x}" if fw else ""),
                file=out,
            )
            for i in fw[:512]:
                print(f"   0x{4 * i:05x}: {wt[i]:08x} (ref {wr[i]:08x})", file=out)
    out.close()


if __name__ == "__main__":
    run_script()
