#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_sdpa_ring_iters

Description:
    Dump (ring_iter, ring_id) pairs that ring_joint_reader.cpp wrote into the
    watcher ring buffer for every tensix worker core on every device.

    Pre-fill is 0xffffffff. Each loop iteration in the kernel pushes
        WATCHER_RING_BUFFER_PUSH(ring_iter)
        WATCHER_RING_BUFFER_PUSH(ring_id)
    so even-indexed slots are ring_iter and odd-indexed slots are ring_id.

    Cores whose ring buffer is empty (current_ptr == -1) are skipped.

Owner:
    ppopovic
"""

from dataclasses import dataclass
from triage import ScriptConfig, triage_field, run_script
from run_checks import run as get_run_checks, RunChecks
from elfs_cache import run as get_elfs_cache, ElfsCache
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.umd_device import TimeoutDeviceRegisterError


script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
)


@dataclass
class SdpaRingIterRow:
    proc: str = triage_field("Proc")
    pairs: str = triage_field("(ring_iter, ring_id) pairs")


def _format_pair(v: int) -> str:
    return "??" if v == 0xFFFFFFFF else str(v)


def read_ring_pairs(
    location: OnChipCoordinate,
    block_type: str,
    risc_name: str,
    dispatcher_data: DispatcherData,
    elf_cache: ElfsCache,
):
    try:
        fw_path = dispatcher_data.get_cached_core_data(location, risc_name).firmware_path
    except TimeoutDeviceRegisterError:
        raise
    except Exception:
        return None

    fw_elf = elf_cache[fw_path]
    mailboxes = dispatcher_data.get_cached_core_data(location, risc_name).mailboxes

    current_ptr = mailboxes.watcher.debug_ring_buf.current_ptr
    if current_ptr == 65535:  # int16 -1, never written
        return None

    ring_elements = fw_elf.get_constant("DEBUG_RING_BUFFER_ELEMENTS")
    assert isinstance(ring_elements, int)

    # Read in physical order: data[0], data[1], ..., data[N-1].
    # The kernel resets current_ptr to -1 before pushing, and push increments
    # before write, so values are in physical-order pairs:
    #   data[0]=ring_iter_0, data[1]=ring_id_0, data[2]=ring_iter_1, ...
    raw = [int(mailboxes.watcher.debug_ring_buf.data[i]) for i in range(ring_elements)]

    # Truncate to the range that was actually written (current_ptr is the last
    # written index; values beyond it are the 0xffffffff pre-fill).
    last_written = current_ptr
    written = raw[: last_written + 1]

    pairs = []
    for i in range(0, len(written), 2):
        ri = _format_pair(written[i])
        rid = _format_pair(written[i + 1]) if i + 1 < len(written) else "<missing>"
        pairs.append(f"({ri},{rid})")

    if not pairs:
        return None

    return SdpaRingIterRow(proc=block_type, pairs=" ".join(pairs))


def read_ring_pairs_for_block(
    location: OnChipCoordinate,
    dispatcher_data: DispatcherData,
    elf_cache: ElfsCache,
    run_checks: RunChecks,
):
    try:
        block_type = run_checks.get_block_type(location)
    except Exception:
        return None

    risc_name = location.noc_block.risc_names[0]

    if not dispatcher_data.risc_enabled(risc_name):
        return None

    return read_ring_pairs(location, block_type, risc_name, dispatcher_data, elf_cache)


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    return run_checks.run_per_block_check(
        lambda location: read_ring_pairs_for_block(location, dispatcher_data, elfs_cache, run_checks),
        block_filter=["tensix"],
    )


if __name__ == "__main__":
    run_script()
