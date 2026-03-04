#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    run_checks [--dev=<device_id>]...

Options:
    --dev=<device_id>   Specify the device id. 'all' is also an option  [default: in_use]

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

from collections import defaultdict
from collections.abc import Callable
from functools import cached_property
import threading
from dataclasses import dataclass
from typing import Literal, TypeAlias

from inspector_data import run as get_inspector_data, InspectorData
from triage import (
    triage_singleton,
    ScriptConfig,
    triage_field,
    recurse_field,
    run_script,
    log_warning,
    create_progress,
    log_check,
)
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.umd_device import TimeoutDeviceRegisterError
from ttexalens.hardware.risc_debug import RiscHaltError
import utils
from metal_device_id_mapping import run as get_metal_device_id_mapping, MetalDeviceIdMapping

script_config = ScriptConfig(
    data_provider=True,
    depends=["inspector_data", "metal_device_id_mapping"],
)

# List of block types that script returns, can be extended if other block types are needed
BLOCK_TYPES = [
    "idle_eth",
    "active_eth",
    "tensix",
    "eth",
]

# We need to map triage block types to inspector block types since we cannot use _ in capnp struct names
INSPECTOR_BLOCK_TYPES = {
    "idle_eth": "idleEth",
    "active_eth": "activeEth",
}

# List of RISC cores currently supported
CORE_TYPES = {
    "brisc",
    "trisc0",
    "trisc1",
    "trisc2",
    "ncrisc",
    "erisc",
    "erisc0",
    "erisc1",
}

BlockType: TypeAlias = Literal[BLOCK_TYPES]
CoreType: TypeAlias = Literal[CORE_TYPES]


# Classes for storing check results for devices, blocks and cores


@dataclass
class CheckResult:
    result: object = recurse_field()

    # Hack to make result the last filed to perserve header order
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
    metal_device_id_mapping: MetalDeviceIdMapping,
    context: Context,
) -> list[Device]:
    if len(devices) == 1 and devices[0].lower() == "in_use":
        if inspector_data is not None:
            metal_device_ids = list(inspector_data.getDevicesInUse().metalDeviceIds)

            if len(metal_device_ids) == 0:
                utils.WARN(
                    f"  No devices in use found in inspector data. Switching to use all available devices. If you are using ttnn check if you have enabled program cache."
                )
                device_ids = [int(id) for id in context.devices.keys()]
            else:
                device_ids = [
                    metal_device_id_mapping.get_device_id(metal_device_id)
                    for metal_device_id in metal_device_ids
                    if metal_device_id_mapping.get_device_id(metal_device_id) is not None
                ]
        else:
            utils.WARN(f"  Using all available devices.")
            device_ids = [int(id) for id in context.devices.keys()]
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


def get_block_locations(
    devices: list[Device],
    inspector_data: InspectorData,
    metal_device_id_mapping: MetalDeviceIdMapping,
) -> dict[Device, dict[BlockType, list[OnChipCoordinate]]]:
    device_map = _make_device_map(devices)
    block_locations: dict[Device, dict[BlockType, list[OnChipCoordinate]]] = defaultdict(dict)
    chip_blocks_list = inspector_data.getBlocksByType().chips

    for i in range(len(chip_blocks_list)):
        metal_device_id = chip_blocks_list[i].chipId
        device_id = metal_device_id_mapping.get_device_id(metal_device_id)
        if device_id in device_map:
            device = device_map[device_id]
            for block_type in BLOCK_TYPES:
                if block_type in INSPECTOR_BLOCK_TYPES:
                    block_locations[device][block_type] = _convert_to_on_chip_coordinates(
                        device, getattr(chip_blocks_list[i].blocks, INSPECTOR_BLOCK_TYPES[block_type]), block_type
                    )
                else:
                    block_locations[device][block_type] = device.get_block_locations(
                        "functional_workers" if block_type == "tensix" else block_type
                    )

    return block_locations


@dataclass(frozen=True)
class BrokenCore:
    location: OnChipCoordinate
    risc_name: str

    def __hash__(self):
        return hash((self.location, self.risc_name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BrokenCore):
            return False
        return self.location == other.location and self.risc_name == other.risc_name

    def __str__(self) -> str:
        return f"{self.risc_name} at {self.location.to_user_str()}"


class RunChecks:
    def __init__(
        self,
        devices: list[Device],
        block_locations: dict[Device, dict[BlockType, list[OnChipCoordinate]]],
        metal_device_id_mapping: MetalDeviceIdMapping,
    ):
        self.devices = devices
        self.metal_device_id_mapping = metal_device_id_mapping
        self.block_locations: dict[Device, dict[BlockType, list[OnChipCoordinate]]] = block_locations
        # If any device has a metal<->exalens mismatch, show all devices as hex unique_id
        self._use_unique_id = metal_device_id_mapping.mismatch_exists()
        # Pre-compute unique_id to device mapping for fast lookup
        self._unique_id_to_device: dict[int, Device] = {device.unique_id: device for device in devices}
        self._broken_devices: set[Device] = set()
        self._broken_cores: dict[Device, set[BrokenCore]] = {}
        self._skip_lock = threading.Lock()

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

    def is_device_broken(self, device: Device) -> bool:
        with self._skip_lock:
            return device in self._broken_devices

    def is_device_in_broken_cores(self, device: Device) -> bool:
        with self._skip_lock:
            return device in self._broken_cores

    def get_device_broken_cores(self, device: Device) -> set[BrokenCore] | None:
        with self._skip_lock:
            return self._broken_cores.get(device).copy()

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
        self, check: Callable[[Device], object], print_broken_devices: bool = True
    ) -> list[PerDeviceCheckResult] | None:
        """Run a check function on each device, collecting results."""
        result: list[PerDeviceCheckResult] = []
        with create_progress() as progress:
            device_task = progress.add_task(
                "Processing devices", total=len(self.devices), visible=len(self.devices) > 1
            )
            try:
                for device in self.devices:
                    # Skipping broken devices
                    with self._skip_lock:
                        if device in self._broken_devices:
                            continue
                    try:
                        check_result = check(device)
                    except TimeoutDeviceRegisterError as e:
                        with self._skip_lock:
                            self._broken_devices.add(device)
                            if print_broken_devices:
                                log_warning(
                                    f"Triage broke device {device.id} with: {e}. This device will be skipped from now on."
                                )
                            if device.is_local:
                                # We are classifying remote devices as broken since we cannot access them if their local device is broken
                                for remote_device in device.remote_devices:
                                    # Broken remote devices will inherit the error from the local device
                                    self._broken_devices.add(remote_device)
                                    if print_broken_devices:
                                        log_warning(
                                            f"Device {remote_device.id} will be skipped from now on due to its local device (device {device.id}) being broken."
                                        )
                        continue
                    except Exception as e:
                        log_warning(f"Skipping device {device.id}: {str(e)}")
                        continue
                    # Use the common result collection helper
                    self._collect_results(
                        result,
                        check_result,
                        PerDeviceCheckResult,
                        device_description=DeviceDescription(device, self._use_unique_id),
                    )
                    progress.advance(device_task)
                return result if len(result) > 0 else None
            finally:
                progress.remove_task(device_task)

    def run_per_block_check(
        self, check: Callable[[OnChipCoordinate], object], block_filter: list[str] | str | None = None
    ) -> list[PerBlockCheckResult] | None:
        """Run a check function on each block location, collecting results."""
        block_types_to_check = (
            BLOCK_TYPES if block_filter is None else [block_filter] if isinstance(block_filter, str) else block_filter
        )

        def per_device_blocks_check(device: Device) -> list[PerBlockCheckResult] | None:
            """Check all block locations for a single device."""
            result: list[PerBlockCheckResult] = []
            with create_progress() as progress:
                progress_count = 0
                for block_type in block_types_to_check:
                    for location in self.block_locations[device][block_type]:
                        progress_count += 1
                device_task = progress.add_task(f"Processing NOC locations", total=progress_count)
                try:
                    for block_type in block_types_to_check:
                        for location in self.block_locations[device][block_type]:
                            check_result = check(location)
                            progress.advance(device_task)
                            # Use the common result collection helper
                            self._collect_results(
                                result,
                                check_result,
                                PerBlockCheckResult,
                                device_description=DeviceDescription(device, self._use_unique_id),
                                location=location,
                            )
                    return result if len(result) > 0 else None
                finally:
                    progress.remove_task(device_task)

        # Reuse the device iteration from run_per_device_check
        return self.run_per_device_check(per_device_blocks_check)

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
                try:
                    check_result = check(location, risc_name)
                except RiscHaltError as e:
                    with self._skip_lock:
                        if (
                            location._device in self._broken_cores.keys()
                            and BrokenCore(location, risc_name) in self._broken_cores[location._device]
                        ):
                            # If the core is already broken we do not need to add it again
                            continue
                        if location._device in self._broken_cores.keys():
                            self._broken_cores[location._device].add(BrokenCore(location, risc_name))
                        else:
                            self._broken_cores[location._device] = {BrokenCore(location, risc_name)}
                    if print_broken_cores:
                        log_warning(
                            f"Triage broke {risc_name} at {location.to_user_str()} at device {location.device_id} with: {e}."
                        )
                    continue
                except Exception as e:
                    log_warning(
                        f"Skipping {risc_name} at {location.to_user_str()} at device {location.device_id}: {str(e)}"
                    )
                    continue

                # Use the common result collection helper
                self._collect_results(
                    result,
                    check_result,
                    PerCoreCheckResult,
                    device_description=DeviceDescription(location._device, self._use_unique_id),
                    location=location,
                    risc_name=risc_name,
                )

            return result if len(result) > 0 else None

        # Reuse the block iteration from run_per_block_check
        return self.run_per_block_check(per_block_cores_check, block_filter)


@triage_singleton
def run(args, context: Context):
    devices_to_check = args["--dev"]
    inspector_data = get_inspector_data(args, context)
    metal_device_id_mapping = get_metal_device_id_mapping(args, context)
    devices = get_devices(devices_to_check, inspector_data, metal_device_id_mapping, context)
    block_locations = get_block_locations(devices, inspector_data, metal_device_id_mapping)
    return RunChecks(devices, block_locations, metal_device_id_mapping)


if __name__ == "__main__":
    run_script()
