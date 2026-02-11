#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_broken_components [--user-view]

Options:
    --user-view      Draws broken cores instead of listing them. Broken cores are marked with an 'x' others are marked with a '-'.

Description:
    This script checks what devices and cores are broken before starting triage.
    It does that by probing devices by reading L1 address 0 and probes cores by attempting to halt them.
    Devices that time out are marked broken. Cores that fail to halt are marked broken.

Owner:
    adjordjevic-TT
"""

from dataclasses import dataclass
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.device import Device
from ttexalens.tt_exalens_lib import read_word_from_device

from run_checks import run as get_run_checks, BrokenCore, RunChecks
from triage import ScriptConfig, run_script, triage_field, collection_serializer

script_config = ScriptConfig(
    depends=["run_checks"],
)

_USER_VIEW = False


def probe_device(device: Device) -> None:
    location = device.get_block_locations()[0]
    assert location is not None
    read_word_from_device(location, 0)


def probe_core(location: OnChipCoordinate, risc_name: str) -> None:
    noc_block = location._device.get_block(location)
    risc_debug = noc_block.get_risc_debug(risc_name)
    if risc_debug.is_in_reset():
        return
    with risc_debug.ensure_halted():
        pass


def group_broken_cores_by_risc_name(broken_cores: set[BrokenCore]) -> dict[str, set[OnChipCoordinate]]:
    broken_cores_by_risc_name: dict[str, set[OnChipCoordinate]] = {}
    for broken_core in broken_cores:
        if broken_core.risc_name not in broken_cores_by_risc_name:
            broken_cores_by_risc_name[broken_core.risc_name] = set()
        broken_cores_by_risc_name[broken_core.risc_name].add(broken_core.location)
    return broken_cores_by_risc_name


def draw_broken_cores(broken_cores: set[BrokenCore]) -> str:
    def location_render(location: OnChipCoordinate) -> str:
        riscs_string = ""
        for risc_name in location.device.get_block(location).risc_names:
            riscs_string += "x" if BrokenCore(location, risc_name) in broken_cores else "-"
        return riscs_string

    return next(iter(broken_cores)).location.device.render(axis_coordinate="noc0", cell_renderer=location_render)


def broken_core_serializer(broken_cores: set[BrokenCore] | None | str) -> str:
    if isinstance(broken_cores, str):
        return broken_cores
    if broken_cores is None:
        return "N/A"
    elif not _USER_VIEW:
        broken_cores_by_risc_name = group_broken_cores_by_risc_name(broken_cores)
        return "\n".join(
            [
                f"[error]{risc_name}[/]: {collection_serializer(', ')(broken_cores_by_risc_name[risc_name])}"
                for risc_name in broken_cores_by_risc_name
            ]
        )
    else:
        return draw_broken_cores(broken_cores)


@dataclass
class DeviceHealthSummary:
    device: Device = triage_field("Device")
    broken_cores: dict[Device, set[BrokenCore]] | None | str = triage_field(
        "Broken Cores", serializer=broken_core_serializer
    )


def collect_device_health_summary(run_checks: RunChecks) -> list[DeviceHealthSummary] | None:
    broken_devices = run_checks.get_broken_devices()
    device_health_summaries: list[DeviceHealthSummary] = []
    for device in run_checks.devices:
        if device in broken_devices:
            device_health_summaries.append(
                DeviceHealthSummary(device=device, broken_cores=f"[error]Device is broken so it is skipped.[/]")
            )
        else:
            broken_cores = run_checks.get_broken_cores()
            if device in broken_cores:
                device_health_summaries.append(DeviceHealthSummary(device=device, broken_cores=broken_cores[device]))
    return device_health_summaries if len(device_health_summaries) > 0 else None


def run(args, context: Context):
    global _USER_VIEW
    _USER_VIEW = bool(args["--user-view"])
    RISC_CORES_TO_CHECK = ["brisc", "trisc0", "trisc1", "trisc2", "erisc", "erisc0", "erisc1"]
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth"]
    run_checks = get_run_checks(args, context)
    # These are used to test health of devices and cores and populate broken_devices and broken_cores sets in RunChecks
    run_checks.run_per_device_check(lambda device: probe_device(device), print_broken_devices=False)
    run_checks.run_per_core_check(
        lambda location, risc_name: probe_core(location, risc_name),
        block_filter=BLOCK_TYPES_TO_CHECK,
        core_filter=RISC_CORES_TO_CHECK,
        print_broken_cores=False,
    )
    return collect_device_health_summary(run_checks)


if __name__ == "__main__":
    run_script()
