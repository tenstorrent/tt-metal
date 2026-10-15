#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    semaphore_provider

Description:
    Data provider for semaphore values on Tensix and Ethernet cores: those of the program running on
    a core, then every GlobalSemaphore covering it, unchanged ones included. get_semaphores(location)
    reads them, and get_cached_semaphores(location) reads them once and returns the same list after.

Owner:
    onenezicTT
"""

from dataclasses import dataclass
from functools import cache

from dispatcher_data import DispatcherData, run as get_dispatcher_data
from inspector_data import InspectorData, run as get_inspector_data
from metal_device_id_mapping import MetalDeviceIdMapping, run as get_metal_device_id_mapping
from run_checks import RunChecks, run as get_run_checks
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


class SemaphoreProvider:
    def __init__(
        self,
        dispatcher_data: DispatcherData,
        inspector_data: InspectorData,
        id_mapping: MetalDeviceIdMapping,
        run_checks: RunChecks,
    ):
        self.dispatcher_data = dispatcher_data
        # Watcher kernel id -> semaphores of that kernel's program.
        self.semaphores_by_kernel: dict[int, list] = {}
        for program in inspector_data.getPrograms().programs:
            semaphores = list(program.semaphores)
            for kernel in program.kernels:
                self.semaphores_by_kernel[kernel.watcherKernelId] = semaphores
        # (address, core ranges, exalens device ids, host-written value) per live GlobalSemaphore this rank reaches.
        self.global_semaphores: list[tuple[int, list, set[int], int | None]] = []
        for record in inspector_data.getGlobalSemaphores().semaphores:
            unique_ids = [id_mapping.get_unique_id(c) for c in record.chipIds if id_mapping.has_metal_device_id(c)]
            devices = [run_checks.get_device_by_unique_id(u) for u in unique_ids]
            device_ids = {d.id for d in devices if d is not None}
            initial = int(record.resetValue) if record.resetValue >= 0 else None
            self.global_semaphores.append((int(record.address), list(record.coreRanges), device_ids, initial))

    @cache
    def get_cached_semaphores(self, location: OnChipCoordinate) -> list[SemaphoreData]:
        return self.get_semaphores(location)

    def get_semaphores(self, location: OnChipCoordinate) -> list[SemaphoreData]:
        return self._read_program_semaphores(location) + self._read_global_semaphores(location)

    @staticmethod
    def _covers(core_ranges, x: int, y: int) -> bool:
        return any(r.start.x <= x <= r.end.x and r.start.y <= y <= r.end.y for r in core_ranges)

    def _read_program_semaphores(self, location: OnChipCoordinate) -> list[SemaphoreData]:
        core = self.dispatcher_data.get_cached_core_data(location.noc_block.all_riscs[0].risc_location)
        if core.go_message == "DONE" or core.mailboxes is None:
            return []
        kernel_config = core.mailboxes.launch[core.launch_msg_rd_ptr].kernel_config
        kernel_ids = [int(kernel_config.watcher_kernel_ids[i]) for i in range(len(kernel_config.watcher_kernel_ids))]
        semaphores: list = next(
            (self.semaphores_by_kernel[k] for k in kernel_ids if k in self.semaphores_by_kernel), []
        )
        base = core.kernel_config_base + int(kernel_config.sem_offset[core.programmable_core_type])
        (x, y), core_type = location.to("logical")
        result = []
        for s in semaphores:
            if s.coreType != core_type or not self._covers(s.coreRanges, x, y):
                continue
            address = base + s.offset
            value = read_word_from_device(location, address)
            result.append(SemaphoreData("program", s.id, address, value, s.initialValue))
        return result

    def _read_global_semaphores(self, location: OnChipCoordinate) -> list[SemaphoreData]:
        (x, y), core_type = location.to("logical")
        if core_type != "tensix":
            return []  # GlobalSemaphores live on worker cores
        return [
            SemaphoreData("global", None, address, read_word_from_device(location, address), initial)
            for address, core_ranges, device_ids, initial in self.global_semaphores
            if location.device.id in device_ids and self._covers(core_ranges, x, y)
        ]


@triage_singleton
def run(args, context: Context) -> SemaphoreProvider:
    return SemaphoreProvider(
        get_dispatcher_data(args, context),
        get_inspector_data(args, context),
        get_metal_device_id_mapping(args, context),
        get_run_checks(args, context),
    )


if __name__ == "__main__":
    run_script()
