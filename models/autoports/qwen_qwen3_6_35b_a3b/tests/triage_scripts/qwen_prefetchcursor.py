# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Capture the BRISC-private cursor in the preserved hang's exact prefetch ELF.

Usage:
    qwen_prefetchcursor

Owner:
    qwen_qwen3_6_35b_a3b functional-decoder bringup
"""

import json
import time
from pathlib import Path

from triage import ScriptConfig
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.memory_access import RiscDebugMemoryAccess

script_config = ScriptConfig()


def run(args, context):
    device = context.devices[0]
    location = OnChipCoordinate.create("16-2", device)
    debug = device.get_block(location).get_risc_debug("brisc")
    if not debug.is_halted():
        raise RuntimeError("This diagnostic is only for an already halted core; do not interrupt a running test")
    access = RiscDebugMemoryAccess(debug)
    data = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # readelf -sW cq_prefetch/10398389770575116831/brisc/brisc.elf
    addresses = {
        "issue_fence": 0xFFB00D00,
        "prefetch_q_rd_ptr": 0xFFB00D04,
        "inflight_count": 0xFFB00D58,
        "stall_state": 0xFFB00D68,
    }
    data["initially_halted"] = debug.is_halted()
    with debug.ensure_halted():
        data["pc"] = hex(debug.get_pc())
        data["gpr"] = [hex(debug.read_gpr(i)) for i in range(32)]
        for name, address in addresses.items():
            buffer = bytearray(4)
            access.read(address, buffer)
            data[name] = hex(int.from_bytes(buffer, "little"))
    data["finally_halted"] = debug.is_halted()
    output = Path(__file__).parents[2] / "doc/functional_decoder/triage/prefetch_private.json"
    output.write_text(json.dumps(data, indent=2) + "\n")
    print("PREFETCH_CURSOR_CAPTURE", data)
