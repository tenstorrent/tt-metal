#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_semaphores [--include-global-unchanged]

Options:
    --include-global-unchanged   Also show global semaphores whose Value equals Initial.

Description:
    Semaphore values on Tensix and Ethernet cores, one row per semaphore per core.

    Kind is program for a semaphore of the program running on the core (CreateSemaphore). It is read
    from the kernel config buffer at the core type's sem_offset, and only on cores its core range
    covers: the host initializes a slot on those cores only, so any other core holds a stale value
    there. Kind is global for a GlobalSemaphore, one word at the same L1 address on every Tensix core
    of its range, shown whether or not the core is running.

    Initial is the value the host last wrote: at launch for a program semaphore, at creation or the
    latest reset_semaphore_value for a global one, N/A if it never wrote one. A kernel resetting a
    semaphore on device is not reflected.

    Without --include-global-unchanged, global rows still at Initial (idle or never signaled) are
    left out, since models can declare ~100 GlobalSemaphores per core. A global semaphore missing
    from a core it covers is therefore at Initial. Program rows are always shown.

    Both lists come from Inspector, which records each program's semaphores and every live
    GlobalSemaphore.

Owner:
    onenezicTT
"""

from dataclasses import dataclass

from dispatcher_data import DispatcherData, run as get_dispatcher_data
from inspector_data import run as get_inspector_data
from metal_device_id_mapping import run as get_metal_device_id_mapping
from run_checks import run as get_run_checks
from triage import ScriptConfig, hex_serializer, run_script, triage_field
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_word_from_device

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "inspector_data", "metal_device_id_mapping"],
)


@dataclass
class SemaphoreRow:
    kind: str = triage_field("Kind")
    id: int | None = triage_field("Id")
    address: int = triage_field("Address", hex_serializer)
    value: int = triage_field("Value")
    initial: int | None = triage_field("Initial")


def covers(core_ranges, x: int, y: int) -> bool:
    return any(r.start.x <= x <= r.end.x and r.start.y <= y <= r.end.y for r in core_ranges)


def program_semaphores_by_kernel(inspector_data) -> dict[int, list]:
    """Watcher kernel id -> semaphores of that kernel's program."""
    by_kernel = {}
    for program in inspector_data.getPrograms().programs:
        semaphores = list(program.semaphores)
        for kernel in program.kernels:
            by_kernel[kernel.watcherKernelId] = semaphores
    return by_kernel


def global_semaphores(inspector_data, id_mapping, run_checks) -> list[tuple[int, list, set[int], int | None]]:
    """(address, core ranges, exalens device ids, host-written value) per live GlobalSemaphore this rank reaches."""
    result = []
    for record in inspector_data.getGlobalSemaphores().semaphores:
        unique_ids = [id_mapping.get_unique_id(c) for c in record.chipIds if id_mapping.has_metal_device_id(c)]
        devices = [run_checks.get_device_by_unique_id(u) for u in unique_ids]
        device_ids = {d.id for d in devices if d is not None}
        initial = int(record.resetValue) if record.resetValue >= 0 else None
        result.append((int(record.address), list(record.coreRanges), device_ids, initial))
    return result


def program_rows(location: OnChipCoordinate, dispatcher_data: DispatcherData, by_kernel) -> list[SemaphoreRow]:
    core = dispatcher_data.get_cached_core_data(location, location.noc_block.risc_names[0])
    if core.go_message == "DONE" or core.mailboxes is None:
        return []
    kernel_config = core.mailboxes.launch[core.launch_msg_rd_ptr].kernel_config
    kernel_ids = [int(kernel_config.watcher_kernel_ids[i]) for i in range(len(kernel_config.watcher_kernel_ids))]
    semaphores: list = next((by_kernel[k] for k in kernel_ids if k in by_kernel), [])
    base = core.kernel_config_base + int(kernel_config.sem_offset[core.programmable_core_type])
    (x, y), core_type = location.to("logical")
    rows = []
    for s in semaphores:
        if s.coreType != core_type or not covers(s.coreRanges, x, y):
            continue
        address = base + s.offset
        value = read_word_from_device(location, address)
        rows.append(SemaphoreRow("program", s.id, address, value, s.initialValue))
    return rows


def global_rows(location: OnChipCoordinate, semaphores, include_unchanged: bool) -> list[SemaphoreRow]:
    (x, y), core_type = location.to("logical")
    if core_type != "tensix":
        return []  # GlobalSemaphores live on worker cores
    rows = []
    for address, core_ranges, device_ids, initial in semaphores:
        if location.device.id in device_ids and covers(core_ranges, x, y):
            value = read_word_from_device(location, address)
            if include_unchanged or value != initial:
                rows.append(SemaphoreRow("global", None, address, value, initial))
    return rows


def read_core(
    location: OnChipCoordinate, dispatcher_data: DispatcherData, by_kernel, globals_, include_unchanged: bool
) -> list | None:
    return (
        program_rows(location, dispatcher_data, by_kernel) + global_rows(location, globals_, include_unchanged) or None
    )


def run(args, context: Context):
    include_unchanged: bool = args["--include-global-unchanged"]
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    inspector_data = get_inspector_data(args, context)
    by_kernel = program_semaphores_by_kernel(inspector_data)
    globals_ = global_semaphores(inspector_data, get_metal_device_id_mapping(args, context), run_checks)
    return run_checks.run_per_block_check(
        lambda location: read_core(location, dispatcher_data, by_kernel, globals_, include_unchanged),
        block_filter=["tensix", "active_eth", "idle_eth"],
    )


if __name__ == "__main__":
    run_script()
