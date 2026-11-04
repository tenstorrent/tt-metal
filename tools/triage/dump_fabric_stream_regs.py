#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_fabric_stream_regs

Description:
    Stream registers the fabric router uses as credit counters, on every active ethernet core running it,
    one row per stream role.

    The host assigns stream ids per router and passes them as named compile-time args, so each role's stream is
    read from the router ELF's constants. One stream can carry two roles (30 and 31 do).
    In 2-ERISC mode each RISC's ELF may keep only the constants its half uses, so the roles of both are merged.

Owner:
    onenezicTT
"""

from dataclasses import dataclass

from dispatcher_data import DispatcherData, run as get_dispatcher_data
from elfs_cache import ElfsCache, run as get_elfs_cache
from run_checks import run as get_run_checks
from triage import ScriptConfig, hex_serializer, log_warning_location, run_script, triage_field
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_word_from_device
from ttexalens.umd_device import TimeoutDeviceRegisterError

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
)

# (STREAM_REMOTE_SRC register, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE register, get_ptr_val mask)
Arch = tuple[int, int, int]
WORMHOLE: Arch = (0, 64, 0xFFFFFFFF)
BLACKHOLE: Arch = (4, 297, (1 << 17) - 1)
STREAM_REG_BASE = 0xFFB40000
STREAM_REG_STRIDE = 0x1000
NUM_STREAMS = 32  # a role the router does not use gets k_unused_stream_id, 32

# The router's stream-id constants, fabric_erisc_router_ct_args.hpp.
STREAM_ID_NAMES = (
    [f"to_receiver_{i}_pkts_sent_id" for i in range(3)]
    + [f"to_sender_{i}_pkts_acked_id" for i in range(5)]
    + [f"to_sender_{i}_pkts_completed_id" for i in range(9)]
    + [f"vc_{vc}_free_slots_from_downstream_edge_{edge}_stream_id" for vc in (0, 1) for edge in range(1, 5)]
    + [f"sender_channel_{i}_free_slots_stream_id" for i in range(10)]
    + ["vc2_receiver_free_slots_stream_id", "tensix_relay_local_free_slots_stream_id"]
    + ["MULTI_RISC_TEARDOWN_SYNC_STREAM_ID", "ETH_RETRAIN_LINK_SYNC_STREAM_ID"]
)


@dataclass
class StreamRegsRow:
    role: str = triage_field("Role")
    stream: int = triage_field("Stream")
    value: int = triage_field("Value")
    scratch: int = triage_field("Scratch", hex_serializer, verbose=1)


def read_core(
    location: OnChipCoordinate, dispatcher_data: DispatcherData, elfs_cache: ElfsCache
) -> list[StreamRegsRow] | None:
    arch = WORMHOLE if location.device.is_wormhole() else BLACKHOLE if location.device.is_blackhole() else None
    if arch is None:
        return None
    elfs = []
    for risc_debug in location.device.get_block(location).all_riscs:
        core = dispatcher_data.get_cached_core_data(risc_debug.risc_location)
        if core.kernel_name == "fabric_erisc_router" and core.kernel_path is not None:
            elfs.append(elfs_cache[core.kernel_path])
    if not elfs:
        return None

    roles: dict[int, set[str]] = {}
    for elf in elfs:
        for name in STREAM_ID_NAMES:
            try:
                stream = elf.get_constant(name)
            except TimeoutDeviceRegisterError:
                raise
            except Exception:
                continue  # constants a RISC's half does not use can be left out of its ELF
            if isinstance(stream, int) and stream < NUM_STREAMS:
                roles.setdefault(stream, set()).add(name)
    if not roles:
        log_warning_location(location, "Fabric router ELF has no stream-id constants")
        return None

    src_reg, space_reg, mask = arch
    rows = []
    for stream, names in sorted(roles.items()):
        base = STREAM_REG_BASE + STREAM_REG_STRIDE * stream
        value = read_word_from_device(location, base + 4 * space_reg) & mask
        value = value - (1 << 32) if value >= 1 << 31 else value  # get_ptr_val returns int32_t
        scratch = read_word_from_device(location, base + 4 * src_reg)
        rows += [StreamRegsRow(name, stream, value, scratch) for name in sorted(names)]
    return rows


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    return run_checks.run_per_block_check(
        lambda location: read_core(location, dispatcher_data, elfs_cache), block_filter="active_eth"
    )


if __name__ == "__main__":
    run_script()
