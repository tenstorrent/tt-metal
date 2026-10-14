#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# Load with: tt-triage.py --additional-scripts-directory=<stress>/triage --run=dump_moe_swiglu [-v]
"""
Usage:
    dump_moe_swiglu

Description:
    One row per Tensix core that is running a moe_fused_swiglu program (kernel path contains "moe_fused_swiglu"):
    its 15 program semaphores, the L1 mailbox words (CB 21, 64 B), the idx scratch (CB 7) and counts scratch (CB 8)
    pages, and pushed/popped counters of the protocol CBs. The semaphore ids are moe_fused_swiglu_geometry.hpp's:
      0 x_ready  1 x_free  2/3 h-mcast (unused)  4 GO (invites received)  5 DATA  6 HSLICE
      7 XSTAGED (= reader gb+1 of last staged block)  8..10 H_RDY slots  11 H_FREE  12 WDSPLIT (= writer (gb+1)*HGROUPS)
      13 PHASE_FREE (= writer gb)  14 HROW_FREE
    Mailbox words: 0 count 1 m_t 2 m_blocks 3 magic (= MAILBOX_MAGIC + local_expert_id of the reader's last publish)
      4 start_row 5 HSEND_DONE 6 UP_SCATTER_DONE (= reader gb+1 of last up scatter).

Owner:
    MiniMax-M3 prefill stress test (models/demos/minimax_m3/tests/stress)
"""

import struct
from dataclasses import dataclass

from dispatcher_data import DispatcherData
from dispatcher_data import run as get_dispatcher_data
from inspector_data import run as get_inspector_data
from run_checks import run as get_run_checks
from triage import ScriptConfig, run_script, triage_field
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.hardware.risc_debug import RiscLocation
from ttexalens.tt_exalens_lib import read_from_device, read_word_from_device

script_config = ScriptConfig(depends=["run_checks", "dispatcher_data", "inspector_data"])

STREAM_REG_BASE = 0xFFB40000
STREAM_REG_STRIDE = 0x1000
BH_RECEIVED_REG, BH_ACKED_REG = 10, 8
CB_NAMES = {
    0: "x_in",
    1: "x_tiles",
    2: "x_stage",
    3: "w_gate",
    4: "w_up",
    5: "w_down",
    6: "h",
    9: "gath_g",
    10: "gath_u",
    11: "sl_g",
    12: "sl_u",
    13: "h_slice",
    14: "out",
    15: "gate_acc",
    16: "up_acc",
    17: "silu",
    19: "out_int",
    20: "mb_wr",
    21: "mb_cmp",
}
SEM_NAMES = [
    "x_ready",
    "x_free",
    "h2",
    "h3",
    "GO",
    "DATA",
    "HSLICE",
    "XSTAGED",
    "HRDY0",
    "HRDY1",
    "HRDY2",
    "H_FREE",
    "WDSPLIT",
    "PHASE_FREE",
    "HROW_FREE",
]


@dataclass
class MoeRow:
    sems: str = triage_field("Semaphores")
    mailbox: str = triage_field("Mailbox w0..15")
    cbs: str = triage_field("CB pushed/popped")
    idx: str = triage_field("idx scratch[0:16]")
    counts: str = triage_field("counts scratch[0:128]", verbose=1)


def words(location, address, n):
    return struct.unpack(f"<{n}I", read_from_device(location, address, num_bytes=4 * n))


def moe_programs(inspector_data):
    """watcher kernel id -> semaphores of a moe_fused_swiglu program."""
    out = {}
    for program in inspector_data.getPrograms().programs:
        if any("moe_fused_swiglu" in (str(k.path) + str(k.name)) for k in program.kernels):
            sems = list(program.semaphores)
            for k in program.kernels:
                out[int(k.watcherKernelId)] = sems
    return out


def read_core(location: OnChipCoordinate, dispatcher_data: DispatcherData, by_kernel) -> list | None:
    core = dispatcher_data.get_cached_core_data(RiscLocation(location, None, "brisc"))
    if core.go_message == "DONE" or core.mailboxes is None:
        return None
    kc = core.mailboxes.launch[core.launch_msg_rd_ptr].kernel_config
    kids = [int(kc.watcher_kernel_ids[i]) for i in range(len(kc.watcher_kernel_ids))]
    sems = next((by_kernel[k] for k in kids if k in by_kernel), None)
    if sems is None:
        return None
    sem_base = core.kernel_config_base + int(kc.sem_offset[core.programmable_core_type])
    vals = {int(s.id): read_word_from_device(location, sem_base + int(s.offset)) for s in sems}
    sem_str = " ".join(f"{SEM_NAMES[i] if i < len(SEM_NAMES) else i}={vals[i]}" for i in sorted(vals))
    cb_base = core.kernel_config_base + int(kc.local_cb_offset)
    mask = int(kc.local_cb_mask)
    cb_addr = {}
    cb_parts = []
    for cb in range(64):
        if not mask >> cb & 1:
            continue
        address, size, pages, page_size = words(location, cb_base + 16 * cb, 4)
        cb_addr[cb] = address
        if cb in CB_NAMES:
            stream = STREAM_REG_BASE + STREAM_REG_STRIDE * cb
            pushed = read_word_from_device(location, stream + 4 * BH_RECEIVED_REG) & 0xFFFF
            popped = read_word_from_device(location, stream + 4 * BH_ACKED_REG) & 0xFFFF
            cb_parts.append(f"{CB_NAMES[cb]}={pushed}/{popped}")
    mb = words(location, cb_addr[21], 16) if 21 in cb_addr else ()
    idx = words(location, cb_addr[7], 16) if 7 in cb_addr else ()
    cnt = words(location, cb_addr[8], 128) if 8 in cb_addr else ()
    return [
        MoeRow(
            sem_str,
            " ".join(hex(w) if i == 3 else str(w) for i, w in enumerate(mb)),
            " ".join(cb_parts),
            " ".join(map(str, idx)),
            " ".join(map(str, cnt)),
        )
    ]


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    by_kernel = moe_programs(get_inspector_data(args, context))
    return run_checks.run_per_block_check(
        lambda location: read_core(location, dispatcher_data, by_kernel), block_filter="tensix"
    )


if __name__ == "__main__":
    run_script()
