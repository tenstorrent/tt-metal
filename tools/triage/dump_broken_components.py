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
    This script prints a summary of broken devices and cores detected during triage.
    It relies on run_checks having already identified all broken components.

Owner:
    adjordjevic-TT
"""

from dataclasses import dataclass
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.device import Device

from run_checks import run as get_run_checks, RunChecks
from triage import ScriptConfig, run_script, triage_field, collection_serializer, ScriptPriority
from triage_session import BrokenCore, get_triage_session
import utils

script_config = ScriptConfig(
    depends=["run_checks"],
    priority=ScriptPriority.LOW,
)

_USER_VIEW = False


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
    broken_cores: set[BrokenCore] | str = triage_field("Broken Cores", serializer=broken_core_serializer)


def collect_device_health_summary(run_checks: RunChecks) -> list[DeviceHealthSummary] | None:
    session = get_triage_session()
    device_health_summaries: list[DeviceHealthSummary] = []
    for device in run_checks.devices:
        if session.is_device_broken(device):
            device_health_summaries.append(
                DeviceHealthSummary(device=device, broken_cores=f"[error]Device is broken so it is skipped.[/]")
            )
        else:
            if session.is_device_in_broken_cores(device):
                device_health_summaries.append(
                    DeviceHealthSummary(device=device, broken_cores=session.get_device_broken_cores(device))
                )
    return device_health_summaries if len(device_health_summaries) > 0 else None


def verify_halted_cores() -> None:
    """Verify that cores halted by triage are still halted.

    If a core was halted by us but is no longer halted, it was broken during
    triage (cont() is patched to no-op on affected architectures, so it
    cannot have been continued intentionally).
    """
    session = get_triage_session()
    for location, risc_name in session.halted_cores:
        device = location.device
        if session.is_device_broken(device):
            continue
        try:
            risc_debug = location.noc_block.get_risc_debug(risc_name)
            if not risc_debug.is_halted():
                utils.WARN(
                    f"Core {risc_name} at {location.to_user_str()} on device {device.id} "
                    f"was halted by triage but is no longer halted — core was broken during triage."
                )
                session.add_broken_core(device, BrokenCore(location, risc_name))
        except Exception as e:
            utils.WARN(
                f"Failed to verify halted state of {risc_name} at {location.to_user_str()} on device {device.id}: {e}"
            )


def run(args, context: Context):
    global _USER_VIEW
    _USER_VIEW = bool(args["--user-view"])
    run_checks = get_run_checks(args, context)
    verify_halted_cores()
    return collect_device_health_summary(run_checks)


if __name__ == "__main__":
    run_script()
