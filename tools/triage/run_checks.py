#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    run_checks [--dev=<device_id>]... [--loc=<location>]... [--execute-sequential]

Options:
    --dev=<device_id>      Specify the device id. Repeatable. 'all' is also an option  [default: in_use]
    --loc=<location>       Specify location/core. Repeatable. Logical coordinates only: R,C (tensix), eX,Y (eth), dX,Y / CHn (dram). Default: all locations
    --execute-sequential   Force fully sequential execution. By default checks run in parallel across MMIO devices: one worker thread per local MMIO device, with its remote devices on the same thread.

Description:
     Data provider script for running checks on devices, block locations and RISC cores. This script provides a single interface for:
    - Device selection and filtering
    - Block location extraction by block type and filtering
    - Running checks per device
    - Running checks per block location
    - Running checks per RISC core

    This enables other scripts to easily run comprehensive checks across all devices and their
    block locations and RISC cores without needing to depend on multiple separate scripts.

Owner:
    adjordjevic-TT
"""

import os
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast, get_args

from rich.progress import Progress, TaskID

from inspector_data import run as get_inspector_data, InspectorData
from triage import (
    triage_singleton,
    ScriptConfig,
    TTTriageError,
    triage_field,
    recurse_field,
    run_script,
    log_warning_device,
    log_warning_risc,
    create_progress,
    log_check,
)
from triage_session import get_triage_session
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.umd_device import TimeoutDeviceRegisterError
from ttexalens.exceptions import RiscHaltError
import utils
from metal_device_id_mapping import run as get_metal_device_id_mapping, MetalDeviceIdMapping

script_config = ScriptConfig(
    data_provider=True,
)

# Block and core types that scripts return.
BlockType: TypeAlias = Literal["idle_eth", "active_eth", "tensix", "eth", "dram"]
CoreType: TypeAlias = Literal["brisc", "trisc0", "trisc1", "trisc2", "ncrisc", "erisc", "erisc0", "erisc1", "drisc"]

BLOCK_TYPES: list[BlockType] = list(get_args(BlockType))
CORE_TYPES: set[CoreType] = set(get_args(CoreType))

# We need to map triage block types to inspector block types since we cannot use _ in capnp struct names
INSPECTOR_BLOCK_TYPES = {
    "idle_eth": "idleEth",
    "active_eth": "activeEth",
}


# Classes for storing check results for devices, blocks and cores


@dataclass
class CheckResult:
    result: object = recurse_field()

    # Hack to make result the last field to preserve header order
    def __post_init__(cls):
        cls.__dataclass_fields__["result"] = cls.__dataclass_fields__.pop("result")


@dataclass
class DeviceDescription:
    device: Device
    use_unique_id: bool


def device_description_serializer(device_desc: DeviceDescription) -> str:
    return hex(device_desc.device.unique_id) if device_desc.use_unique_id else str(device_desc.device.id)


@dataclass
class PerDeviceCheckResult(CheckResult):
    device_description: DeviceDescription = triage_field("Dev", device_description_serializer)


@dataclass
class PerBlockCheckResult(PerDeviceCheckResult):
    location: OnChipCoordinate = triage_field("Loc")


@dataclass
class PerCoreCheckResult(PerBlockCheckResult):
    risc_name: str = triage_field("RiscV")


def get_devices(
    devices: list[str],
    inspector_data: InspectorData | None,
    metal_device_id_mapping: MetalDeviceIdMapping | None,
    context: Context,
) -> list[Device]:
    if len(devices) == 1 and devices[0].lower() == "in_use":
        if inspector_data is None or metal_device_id_mapping is None:
            # No Inspector. Fall back to TT_METAL_VISIBLE_DEVICES - exalens sees the same subset.
            if os.environ.get("TT_METAL_VISIBLE_DEVICES"):
                utils.WARN(
                    f"  Inspector unavailable; using the {len(context.devices)} device(s) "
                    f"exposed via TT_METAL_VISIBLE_DEVICES."
                )
                return list(context.devices.values())
            raise TTTriageError(
                "Triage (with --dev=in_use) needs Inspector data or TT_METAL_VISIBLE_DEVICES set; "
                "pass --dev=<id> or --dev=all to override and use specific or all devices correspondingly."
            )
        metal_device_ids = list(inspector_data.getDevicesInUse().metalDeviceIds)

        if len(metal_device_ids) == 0:
            # Live "in use" list is empty — most often because firmware init failed and
            # devices were torn down. Fall back to the SystemMesh's configured local set.
            system_mesh = inspector_data.getSystemMesh().systemMesh
            metal_device_ids = [m.localChipId for m in system_mesh.mappedDevices if m.isLocal]
            if len(metal_device_ids) > 0:
                utils.WARN(
                    f"  No devices in use found in inspector data — firmware init likely failed. "
                    f"Falling back to the {len(metal_device_ids)} device(s) configured in the System Mesh."
                )
            else:
                raise TTTriageError(
                    "Cannot determine which devices to inspect: no active devices in metal and the "
                    "System Mesh has no host-local devices."
                )
        device_ids = [
            metal_device_id_mapping.get_device_id(metal_device_id)
            for metal_device_id in metal_device_ids
            if metal_device_id_mapping.get_device_id(metal_device_id) is not None
        ]
    elif len(devices) == 1 and devices[0].lower() == "all":
        device_ids = [int(id) for id in context.devices.keys()]
    else:
        device_ids = [int(id) for id in devices]

    return [context.devices[id] for id in device_ids]


def _convert_to_on_chip_coordinates(
    device: Device, block_locations: list, block_type: BlockType
) -> list[OnChipCoordinate]:
    on_chip_coordinates: list[OnChipCoordinate] = []
    for location in block_locations:
        coord_str = f"e{location.x},{location.y}" if "eth" in block_type.lower() else f"{location.x},{location.y}"
        # We skip e0,15 on wormhole devices since it is reserved for syseng use
        if coord_str == "e0,15" and device.is_wormhole() and device.is_local:
            continue
        on_chip_coordinates.append(OnChipCoordinate.create(coord_str, device))
    return on_chip_coordinates


def _make_device_map(devices: list[Device]) -> dict[int, Device]:
    return {device.id: device for device in devices}


def _exalens_block_locations(device: Device, block_type: BlockType) -> list[OnChipCoordinate]:
    if block_type == "active_eth":
        return device.active_eth_block_locations
    if block_type == "idle_eth":
        return device.idle_eth_block_locations
    if block_type == "dram" and not device.is_blackhole():
        return []
    return device.get_block_locations("functional_workers" if block_type == "tensix" else block_type)


def get_block_locations(
    devices: list[Device],
    locations: list[str],
    inspector_data: InspectorData | None,
    metal_device_id_mapping: MetalDeviceIdMapping | None,
) -> dict[Device, dict[BlockType, list[OnChipCoordinate]]]:
    block_locations: dict[Device, dict[BlockType, list[OnChipCoordinate]]] = defaultdict(dict)
    covered: set[Device] = set()

    if inspector_data is not None and metal_device_id_mapping is not None:
        device_map = _make_device_map(devices)
        chip_blocks_list = inspector_data.getBlocksByType().chips
        for i in range(len(chip_blocks_list)):
            metal_device_id = chip_blocks_list[i].chipId
            device_id = metal_device_id_mapping.get_device_id(metal_device_id)
            if device_id in device_map:
                device = device_map[device_id]
                covered.add(device)
                for block_type in BLOCK_TYPES:
                    if block_type in INSPECTOR_BLOCK_TYPES:
                        block_locations[device][block_type] = _convert_to_on_chip_coordinates(
                            device, getattr(chip_blocks_list[i].blocks, INSPECTOR_BLOCK_TYPES[block_type]), block_type
                        )
                    else:
                        block_locations[device][block_type] = _exalens_block_locations(device, block_type)

    # Exalens fallback for any device Inspector didn't cover (no inspector at all, or
    # Inspector's getBlocksByType is missing this device).
    for device in devices:
        if device in covered:
            continue
        for block_type in BLOCK_TYPES:
            block_locations[device][block_type] = _exalens_block_locations(device, block_type)

    # Keep only the requested locations. Only logical coordinates are accepted — physical (noc0)
    # layout shifts with harvesting, so a logical string maps to the same core on every device.
    if locations:
        for loc in locations:
            if "-" in loc:
                raise TTTriageError(f"--loc expects a logical coordinate (R,C / eX,Y / dX,Y / CHn), got '{loc}'")
        for device in devices:
            wanted = {OnChipCoordinate.create(loc, device) for loc in locations}
            for block_type in BLOCK_TYPES:
                block_locations[device][block_type] = [
                    location for location in block_locations[device][block_type] if location in wanted
                ]

    return block_locations


class RunChecks:
    def __init__(
        self,
        devices: list[Device],
        block_locations: dict[Device, dict[BlockType, list[OnChipCoordinate]]],
        metal_device_id_mapping: MetalDeviceIdMapping | None,
        execute_in_parallel: bool = True,
    ):
        self.devices = devices
        self.metal_device_id_mapping = metal_device_id_mapping
        self.block_locations: dict[Device, dict[BlockType, list[OnChipCoordinate]]] = block_locations
        # If any device has a metal<->exalens mismatch, show all devices as hex unique_id.
        if metal_device_id_mapping is not None:
            self._use_unique_id = metal_device_id_mapping.mismatch_exists()
        else:
            self._use_unique_id = bool(os.environ.get("TT_METAL_VISIBLE_DEVICES"))
        # Pre-compute unique_id to device mapping for fast lookup
        self._unique_id_to_device: dict[int, Device] = {device.unique_id: device for device in devices}
        self._session = get_triage_session()
        self._execute_in_parallel = execute_in_parallel and len(self._mmio_groups) > 1

    @cached_property
    def _mmio_groups(self) -> list[list[Device]]:
        """Group devices by their MMIO access. Each group runs on one worker thread."""
        groups: dict[int, list[Device]] = defaultdict(list)
        for device in self.devices:
            groups[device.local_device.id].append(device)
        return list(groups.values())

    @cached_property
    def _location_to_block_type_map(self) -> dict[OnChipCoordinate, BlockType]:
        map: dict[OnChipCoordinate, BlockType] = {}
        for device in self.devices:
            for block_type in BLOCK_TYPES:
                for location in self.block_locations[device][block_type]:
                    if location not in map:
                        map[location] = block_type
        return map

    def get_device_by_unique_id(self, unique_id: int) -> Device | None:
        return self._unique_id_to_device.get(unique_id)

    def get_block_type(self, location: OnChipCoordinate):
        log_check(
            location in self._location_to_block_type_map,
            f"Location {location.to_user_str()} not found in location to block type map",
        )
        return self._location_to_block_type_map[location]

    def _collect_results(
        self, result: list[CheckResult], check_result: object, result_type: type[CheckResult], **kwargs
    ) -> list[CheckResult]:
        """Helper to collect and wrap check results consistently."""
        if check_result is None:
            return result
        if isinstance(check_result, list):
            for item in check_result:
                if not isinstance(item, CheckResult):
                    result.append(result_type(result=item, **kwargs))
                else:
                    result.append(item)
        else:
            if not isinstance(check_result, CheckResult):
                result.append(result_type(result=check_result, **kwargs))
            else:
                result.append(check_result)
        return result

    def run_per_device_check(
        self,
        check: Callable[[Device], object],
        print_broken_devices: bool = True,
        *,
        progress: Progress | None = None,
        device_task: TaskID | None = None,
    ) -> list[PerDeviceCheckResult] | None:
        """Run a check function on each device, collecting results."""

        # If progress is not provided, create our own progress and task, and call recursively with it.
        if progress is None:
            with create_progress() as own_progress:
                own_task = own_progress.add_task(
                    "Processing devices", total=len(self.devices), visible=len(self.devices) > 1
                )
                try:
                    return self.run_per_device_check(
                        check, print_broken_devices, progress=own_progress, device_task=own_task
                    )
                finally:
                    own_progress.remove_task(own_task)

        assert device_task is not None, "device_task must be provided when progress is provided"

        def process_one(device: Device) -> list[PerDeviceCheckResult]:
            """Run the check on a single device, applying the broken-cascade rules."""
            local_result: list[PerDeviceCheckResult] = []
            if self._session.is_device_broken(device):
                return local_result
            try:
                check_result = check(device)
            except TimeoutDeviceRegisterError as e:
                self._session.add_broken_device(device)
                if print_broken_devices:
                    log_warning_device(
                        device, f"Triage broke device with: {e}. This device will be skipped from now on."
                    )
                if device.is_local:
                    # Cascade: cannot reach remote devices once their local MMIO parent is broken.
                    for remote_device in device.remote_devices:
                        self._session.add_broken_device(remote_device)
                        if print_broken_devices:
                            log_warning_device(
                                remote_device,
                                f"Will be skipped from now on due to its local device (device {device.id}) being broken.",
                            )
                return local_result
            except Exception as e:
                log_warning_device(device, f"Skipping: {str(e)}")
                return local_result
            self._collect_results(
                local_result,
                check_result,
                PerDeviceCheckResult,
                device_description=DeviceDescription(device, self._use_unique_id),
            )
            return local_result

        if not self._execute_in_parallel:
            # Sequential: iterate self.devices on the main thread.
            result: list[PerDeviceCheckResult] = []
            for device in self.devices:
                result.extend(process_one(device))
                progress.advance(device_task)
            return result if len(result) > 0 else None

        # Parallel: one worker per MMIO group.
        def process_group(group: list[Device]) -> list[tuple[int, list[PerDeviceCheckResult]]]:
            per_device: list[tuple[int, list[PerDeviceCheckResult]]] = []
            for device in group:
                items = process_one(device)
                if items:
                    per_device.append((device.id, items))
            return per_device

        groups = self._mmio_groups
        per_device_results: list[tuple[int, list[PerDeviceCheckResult]]] = []
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = {executor.submit(process_group, g): g for g in groups}
            for future in as_completed(futures):
                per_device_results.extend(future.result())
                for _ in futures[future]:
                    progress.advance(device_task)

        per_device_results.sort(key=lambda x: x[0])
        result = []
        for _, items in per_device_results:
            result.extend(items)
        return result if len(result) > 0 else None

    def run_per_block_check(
        self,
        check: Callable[[OnChipCoordinate], object],
        block_filter: list[str] | str | None = None,
        *,
        progress: Progress | None = None,
        device_task: TaskID | None = None,
        item_task: TaskID | None = None,
    ) -> list[PerBlockCheckResult] | None:
        """Run a check function on each block location, collecting results."""

        # If progress is not provided, create our own progress and tasks, and call recursively with it.
        block_types_to_check = cast(
            "list[BlockType]",
            BLOCK_TYPES if block_filter is None else [block_filter] if isinstance(block_filter, str) else block_filter,
        )

        if progress is None:
            total_blocks = sum(
                len(self.block_locations[device][bt]) for device in self.devices for bt in block_types_to_check
            )
            with create_progress() as own_progress:
                own_device_task = own_progress.add_task(
                    "Processing devices", total=len(self.devices), visible=len(self.devices) > 1
                )
                own_item_task = own_progress.add_task("Processing NOC locations", total=total_blocks)
                try:
                    return self.run_per_block_check(
                        check,
                        block_filter,
                        progress=own_progress,
                        device_task=own_device_task,
                        item_task=own_item_task,
                    )
                finally:
                    own_progress.remove_task(own_device_task)
                    own_progress.remove_task(own_item_task)

        assert device_task is not None, "device_task must be provided when progress is provided"

        def per_device_blocks_check(device: Device) -> list[PerBlockCheckResult] | None:
            result: list[PerBlockCheckResult] = []
            for block_type in block_types_to_check:
                for location in self.block_locations[device][block_type]:
                    check_result = check(location)
                    if item_task is not None:
                        progress.advance(item_task)
                    self._collect_results(
                        result,
                        check_result,
                        PerBlockCheckResult,
                        device_description=DeviceDescription(device, self._use_unique_id),
                        location=location,
                    )
            return result if len(result) > 0 else None

        return cast(
            "list[PerBlockCheckResult] | None",
            self.run_per_device_check(per_device_blocks_check, progress=progress, device_task=device_task),
        )

    def run_per_core_check(
        self,
        check: Callable[[OnChipCoordinate, CoreType], object],
        block_filter: list[str] | str | None = None,
        core_filter: list[str] | str | None = None,
        print_broken_cores: bool = True,
    ) -> list[PerCoreCheckResult] | None:
        """Run a check function on each RISC core in each block location, collecting results."""

        # Filtering cores to check
        cores_to_check = (
            CORE_TYPES
            if core_filter is None
            else set([core_filter])
            if isinstance(core_filter, str)
            else set(core_filter)
        )

        def per_block_cores_check(location: OnChipCoordinate) -> list[PerCoreCheckResult] | None:
            """Check all RISC cores for a single block location."""
            result: list[PerCoreCheckResult] = []

            # Get the block and its available RISC cores
            noc_block = location._device.get_block(location)
            risc_names = noc_block.risc_names

            for risc_name in risc_names:
                # Skipping cores we do not want to check
                if risc_name not in cores_to_check:
                    continue
                risc = cast("CoreType", risc_name)
                try:
                    check_result = check(location, risc)
                except RiscHaltError as e:
                    self._session.add_broken_core(location, risc)
                    if print_broken_cores:
                        log_warning_risc(risc, location, f"Broken: {e}.")
                    continue
                except Exception as e:
                    log_warning_risc(risc, location, f"Skipping: {str(e)}")
                    continue

                self._collect_results(
                    cast("list[CheckResult]", result),
                    check_result,
                    PerCoreCheckResult,
                    device_description=DeviceDescription(location._device, self._use_unique_id),
                    location=location,
                    risc_name=risc,
                )

            return result if len(result) > 0 else None

        # Cast: items are PerCoreCheckResult, narrowed from the parent return type.
        return cast(
            "list[PerCoreCheckResult] | None",
            self.run_per_block_check(per_block_cores_check, block_filter),
        )


@triage_singleton
def run(args, context: Context):
    devices_to_check = args["--dev"]
    locs_to_check = args["--loc"]
    execute_in_parallel = not bool(args["--execute-sequential"])
    try:
        inspector_data = get_inspector_data(args, context)
        metal_device_id_mapping = get_metal_device_id_mapping(args, context)
    except Exception:
        inspector_data = None
        metal_device_id_mapping = None
    devices = get_devices(devices_to_check, inspector_data, metal_device_id_mapping, context)
    block_locations = get_block_locations(devices, locs_to_check, inspector_data, metal_device_id_mapping)
    return RunChecks(devices, block_locations, metal_device_id_mapping, execute_in_parallel=execute_in_parallel)


if __name__ == "__main__":
    run_script()
