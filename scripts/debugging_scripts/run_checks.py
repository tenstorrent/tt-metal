#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from inspector_data import run as get_inspector_data, InspectorData
from triage import triage_singleton, ScriptConfig, triage_field, recurse_field, run_script
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.hw.tensix.wormhole.wormhole import WormholeDevice
from utils import ORANGE, RST

script_config = ScriptConfig(
    data_provider=True,
    depends=["inspector_data"],
)

# List of block types that script returns, can be extended if other block types are needed
BLOCK_TYPES = [
    "idle_eth",
    "active_eth",
    "tensix",
    "eth",
]

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
class PerDeviceCheckResult(CheckResult):
    device: Device = triage_field("Dev")


@dataclass
class PerBlockCheckResult(PerDeviceCheckResult):
    location: OnChipCoordinate = triage_field("Loc")


@dataclass
class PerCoreCheckResult(PerBlockCheckResult):
    risc_name: str = triage_field("RiscV")


def is_galaxy(device: Device) -> bool:
    return device.cluster_desc["chip_to_boardtype"][device._id] == "GALAXY"


def get_idle_eth_block_locations(device: Device) -> list[OnChipCoordinate]:
    block_locations = device.idle_eth_block_locations
    # We remove idle eth blocks that are reserved for syseng use
    # These are blocks on wormhole mmio capable devices with connections to remote devices
    # If board type is galaxy, we remove idle eth blocks at locations e0,0 e0,1 e0,2 e0,3 and e0,15,
    # if not we just remove e0,15
    if isinstance(device, WormholeDevice) and device._has_mmio:
        locations_to_remove = {"e0,0", "e0,1", "e0,2", "e0,3", "e0,15"} if is_galaxy(device) else {"e0,15"}
        block_locations = [loc for loc in block_locations if loc.to_str("logical") not in locations_to_remove]

    return block_locations


def get_block_locations_to_check(block_type: BlockType, device: Device) -> list[OnChipCoordinate]:
    match block_type:
        case "idle_eth":
            return get_idle_eth_block_locations(device)
        case "active_eth":
            return device.active_eth_block_locations
        case _:
            # In exalens we call tensix blocks functional_workers
            block_type = "functional_workers" if block_type == "tensix" else block_type
            return device.get_block_locations(block_type)


def get_devices(devices: list[str], inspector_data: InspectorData | None, context: Context) -> list[Device]:
    if len(devices) == 1 and devices[0].lower() == "in_use":
        if inspector_data is not None:
            device_ids = list(inspector_data.devices_in_use)
            if len(device_ids) == 0:
                print(
                    f"  {ORANGE}No devices in use found in inspector data. Switching to use all available devices. If you are using ttnn check if you have enabled program cache.{RST}"
                )
                device_ids = [int(id) for id in context.devices.keys()]
        else:
            print(f"  {ORANGE}Using all available devices.{RST}")
            device_ids = [int(id) for id in context.devices.keys()]
    elif len(devices) == 1 and devices[0].lower() == "all":
        device_ids = [int(id) for id in context.devices.keys()]
    else:
        device_ids = [int(id) for id in devices]
    return [context.devices[id] for id in device_ids]


class RunChecks:
    def __init__(self, devices: list[Device]):
        self.devices = devices
        # Pre-compute block locations for all devices and block types
        self.block_locations: dict[Device, dict[BlockType, list[OnChipCoordinate]]] = {
            device: {block_type: get_block_locations_to_check(block_type, device) for block_type in BLOCK_TYPES}
            for device in devices
        }

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

    def run_per_device_check(self, check: Callable[[Device], object]) -> list[PerDeviceCheckResult] | None:
        """Run a check function on each device, collecting results."""
        result: list[PerDeviceCheckResult] = []
        for device in self.devices:
            check_result = check(device)
            # Use the common result collection helper
            self._collect_results(result, check_result, PerDeviceCheckResult, device=device)

        return result if len(result) > 0 else None

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
            for block_type in block_types_to_check:
                for location in self.block_locations[device][block_type]:
                    check_result = check(location)
                    # Use the common result collection helper
                    self._collect_results(
                        result,
                        check_result,
                        PerBlockCheckResult,
                        device=device,
                        location=location,
                    )
            return result if len(result) > 0 else None

        # Reuse the device iteration from run_per_device_check
        return self.run_per_device_check(per_device_blocks_check)

    def run_per_core_check(
        self,
        check: Callable[[OnChipCoordinate, CoreType], object],
        block_filter: list[str] | str | None = None,
        core_filter: list[str] | str | None = None,
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

                check_result = check(location, risc_name)
                # Use the common result collection helper
                self._collect_results(
                    result,
                    check_result,
                    PerCoreCheckResult,
                    device=location._device,
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
    devices = get_devices(devices_to_check, inspector_data, context)
    return RunChecks(devices)


if __name__ == "__main__":
    run_script()
