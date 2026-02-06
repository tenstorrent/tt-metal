#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_broken_components

Description:
    Probes devices by reading L1 address 0 and probes cores by attempting to halt them.
    Devices that time out are marked broken. Cores that fail to halt are marked broken.

Owner:
    adjordjevic-TT
"""

from collections import defaultdict
from dataclasses import dataclass
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.device import Device
from ttexalens.tt_exalens_lib import read_word_from_device

from run_checks import run as get_run_checks, BrokenDevice, BrokenCore
from triage import ScriptConfig, triage_field, run_script, log_warning

script_config = ScriptConfig(
    depends=["run_checks"],
)


def probe_device(device: Device) -> None:
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
    from ttexalens.hardware.risc_debug import RiscHaltError

    if risc_name == "erisc" and "e0,10" in location.to_user_str():
        raise RiscHaltError(risc_name, location)
    noc_block = location._device.get_block(location)
    risc_debug = noc_block.get_risc_debug(risc_name)
    if risc_debug.is_in_reset():
        return
    with risc_debug.ensure_halted():
        pass


def print_broken_components(broken_devices: list[BrokenDevice], broken_cores: list[BrokenCore]) -> None:
    if len(broken_devices) > 0:
        log_warning(f"The following devices are broken and will be skipped in triage:")
        for broken_device in broken_devices:
            log_warning(f"  Device {broken_device.device.id} due to {broken_device.error}")
    if len(broken_cores) > 0:
        log_warning(f"The following cores are broken and will be skipped in triage:")
        for broken_core in broken_cores:
            log_warning(
                f"  {broken_core.risc_name} at {broken_core.location.to_user_str()} at device {broken_core.location.device_id} due to {broken_core.error}"
            )


def run(args, context: Context):
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

    print(run_checks._broken_cores)

    def location_render(location: OnChipCoordinate) -> str:
        if location in location.device.active_eth_block_locations:
            return "A"
        if location.noc_block.block_type == "dram":
            return "D"
        if location.noc_block.block_type == "harvested_workers":
            return "H"
        s = ""
        for risc_name in location.device.get_block(location).risc_names:
            if BrokenCore(location, risc_name) in run_checks._broken_cores:
                s += "x"
            else:
                s += "-"
        return s

    # print_broken_components(run_checks.get_broken_devices(), run_checks.get_broken_cores())
    for device in run_checks.devices:
        print(f"Device {device.id} [BROKEN]")
        print(device.render(axis_coordinate="die", cell_renderer=location_render))

    return None


if __name__ == "__main__":
    run_script()
