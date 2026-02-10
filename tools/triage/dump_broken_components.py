#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_broken_components [--user-view]

Options:
    --user-view      Draws broken cores instead of listing them.

Description:
    Probes devices by reading L1 address 0 and probes cores by attempting to halt them.
    Devices that time out are marked broken. Cores that fail to halt are marked broken.

Owner:
    adjordjevic-TT
"""

from dataclasses import dataclass
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.device import Device
from ttexalens.tt_exalens_lib import read_word_from_device

from run_checks import run as get_run_checks, BrokenDevice, BrokenCore, RunChecks
from triage import ScriptConfig, run_script, triage_field, collection_serializer

script_config = ScriptConfig(
    depends=["run_checks"],
)

_USER_VIEW = False


def probe_device(device: Device) -> None:
    # TESTING CODE
    # import tt_umd
    # from ttexalens.umd_device import TimeoutDeviceRegisterError
    # coord = tt_umd.CoreCoord(1, 1, tt_umd.CoreType.TENSIX, tt_umd.CoordSystem.NOC0)
    # if device.id == 0:
    #     raise TimeoutDeviceRegisterError(
    #         chip_id=device.id, coord=coord, address=0x1000, size=4, is_read=True, duration=1.0)
    location = device.get_block_locations()[0]
    assert location is not None
    read_word_from_device(location, 0)


def probe_core(location: OnChipCoordinate, risc_name: str) -> None:
    # TESTING CODE
    # from ttexalens.hardware.risc_debug import RiscHaltError

    # if risc_name == "erisc" and ("e0,10" in location.to_user_str() or "e0,0" in location.to_user_str()):
    #     raise RiscHaltError(risc_name, location)
    # if risc_name == "brisc" and ("0,0" in location.to_user_str() or "0,1" in location.to_user_str() or "0,2" in location.to_user_str() or "0,3" in location.to_user_str()):
    #     raise RiscHaltError(risc_name, location)
    noc_block = location._device.get_block(location)
    risc_debug = noc_block.get_risc_debug(risc_name)
    if risc_debug.is_in_reset():
        return
    with risc_debug.ensure_halted():
        pass


def error_serializer(value: Exception | None) -> str:
    return "N/A" if value is None else f"[error]{str(value)}[/]"


def group_broken_cores_by_risc_name(broken_cores: set[BrokenCore]) -> dict[str, set[OnChipCoordinate]]:
    broken_cores_by_risc_name: dict[str, set[OnChipCoordinate]] = {}
    for broken_core in broken_cores:
        if broken_core.risc_name not in broken_cores_by_risc_name:
            broken_cores_by_risc_name[broken_core.risc_name] = set()
        broken_cores_by_risc_name[broken_core.risc_name].add(broken_core.location)
    return broken_cores_by_risc_name


def draw_broken_cores(broken_cores: set[BrokenCore]) -> str:
    def location_render(location: OnChipCoordinate) -> str:
        s = ""
        for risc_name in location.device.get_block(location).risc_names:
            if BrokenCore(location, risc_name) in broken_cores:
                s += "x"
            else:
                # Skipping DRAM and ACTIVE_ETH blocks because we do not check them
                if (
                    not location in location.device.active_eth_block_locations
                    and not location.noc_block.block_type == "dram"
                ):
                    s += "-"
        return s

    return next(iter(broken_cores)).location.device.render(axis_coordinate="noc0", cell_renderer=location_render)


def broken_core_serializer(broken_cores: set[BrokenCore] | None, user_view: bool | None = None) -> str:
    if user_view is None:
        user_view = _USER_VIEW
    if broken_cores is None:
        return "N/A"
    elif not user_view:
        broken_cores_by_risc_name = group_broken_cores_by_risc_name(broken_cores)
        return "\n".join(
            [
                f"{risc_name}: {collection_serializer(', ')(broken_cores_by_risc_name[risc_name])}"
                for risc_name in broken_cores_by_risc_name
            ]
        )
    else:
        return draw_broken_cores(broken_cores)


@dataclass
class DeviceHealthSummary:
    device: Device = triage_field("Device")
    error: Exception | None = triage_field("Error", serializer=error_serializer)
    broken_cores: dict[Device, set[BrokenCore]] | None = triage_field("Broken Cores", serializer=broken_core_serializer)


def collect_device_health_summary(run_checks: RunChecks) -> DeviceHealthSummary:
    broken_devices = run_checks.get_broken_devices()
    for device in run_checks.devices:
        if BrokenDevice(device=device) in broken_devices:
            return DeviceHealthSummary(
                device=device, error=next(bd for bd in broken_devices if bd.device == device).error, broken_cores=None
            )
        else:
            broken_cores = run_checks.get_broken_cores()
            return DeviceHealthSummary(
                device=device, error=None, broken_cores=broken_cores[device] if device in broken_cores else None
            )


def run(args, context: Context):
    global _USER_VIEW
    _USER_VIEW = bool(args["--user-view"])
    RISC_CORES_TO_CHECK = ["brisc", "trisc0", "trisc1", "trisc2", "erisc", "erisc0", "erisc1"]
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth"]
    run_checks = get_run_checks(args, context)
    # These are used to test health of devices and cores and populate broken_devices and broken_cores sets in RunChecks
    run_checks.run_per_device_check(lambda device: probe_device(device))
    run_checks.run_per_core_check(
        lambda location, risc_name: probe_core(location, risc_name),
        block_filter=BLOCK_TYPES_TO_CHECK,
        core_filter=RISC_CORES_TO_CHECK,
    )
    broken_components = collect_device_health_summary(run_checks)
    return broken_components


if __name__ == "__main__":
    run_script()
