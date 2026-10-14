#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    semaphores_provider

Description:
    Data provider for semaphore values on Tensix and Ethernet cores. get_semaphores(location) reads a
    core's semaphores once and caches them: those of the program running on the core, then every
    GlobalSemaphore covering it, unchanged ones included.

Owner:
    onenezicTT
"""

import threading
from dataclasses import dataclass

from dispatcher_data import DispatcherData, run as get_dispatcher_data
from inspector_data import run as get_inspector_data
from metal_device_id_mapping import run as get_metal_device_id_mapping
from run_checks import run as get_run_checks
from triage import ScriptConfig, hex_serializer, run_script, triage_field, triage_singleton
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_word_from_device

script_config = ScriptConfig(
    data_provider=True,
    depends=["run_checks", "dispatcher_data", "inspector_data", "metal_device_id_mapping"],
)


@dataclass
class SemaphoreData:
    kind: str = triage_field("Kind")  # program or global
    id: int | None = triage_field("Id")  # program semaphores only
    address: int = triage_field("Address", hex_serializer)
    value: int = triage_field("Value")
    initial: int | None = triage_field("Initial")  # last host-written value, None if the host never wrote one


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


def read_program_semaphores(
    location: OnChipCoordinate, dispatcher_data: DispatcherData, by_kernel
) -> list[SemaphoreData]:
    core = dispatcher_data.get_cached_core_data(location.noc_block.all_riscs[0].risc_location)
    if core.go_message == "DONE" or core.mailboxes is None:
        return []
    kernel_config = core.mailboxes.launch[core.launch_msg_rd_ptr].kernel_config
    kernel_ids = [int(kernel_config.watcher_kernel_ids[i]) for i in range(len(kernel_config.watcher_kernel_ids))]
    semaphores: list = next((by_kernel[k] for k in kernel_ids if k in by_kernel), [])
    base = core.kernel_config_base + int(kernel_config.sem_offset[core.programmable_core_type])
    (x, y), core_type = location.to("logical")
    result = []
    for s in semaphores:
        if s.coreType != core_type or not covers(s.coreRanges, x, y):
            continue
        address = base + s.offset
        result.append(SemaphoreData("program", s.id, address, read_word_from_device(location, address), s.initialValue))
    return result


def read_global_semaphores(location: OnChipCoordinate, globals_) -> list[SemaphoreData]:
    (x, y), core_type = location.to("logical")
    if core_type != "tensix":
        return []  # GlobalSemaphores live on worker cores
    return [
        SemaphoreData("global", None, address, read_word_from_device(location, address), initial)
        for address, core_ranges, device_ids, initial in globals_
        if location.device.id in device_ids and covers(core_ranges, x, y)
    ]


class SemaphoresProvider:
    def __init__(self, dispatcher_data: DispatcherData, by_kernel: dict[int, list], globals_: list):
        self.dispatcher_data = dispatcher_data
        self.by_kernel = by_kernel
        self.globals_ = globals_
        self._cache: dict[OnChipCoordinate, list[SemaphoreData]] = {}
        self._lock = threading.Lock()

    def get_semaphores(self, location: OnChipCoordinate) -> list[SemaphoreData]:
        with self._lock:
            if location not in self._cache:
                program = read_program_semaphores(location, self.dispatcher_data, self.by_kernel)
                self._cache[location] = program + read_global_semaphores(location, self.globals_)
            return self._cache[location]


@triage_singleton
def run(args, context: Context) -> SemaphoresProvider:
    run_checks = get_run_checks(args, context)
    inspector_data = get_inspector_data(args, context)
    by_kernel = program_semaphores_by_kernel(inspector_data)
    globals_ = global_semaphores(inspector_data, get_metal_device_id_mapping(args, context), run_checks)
    return SemaphoresProvider(get_dispatcher_data(args, context), by_kernel, globals_)


if __name__ == "__main__":
    run_script()
