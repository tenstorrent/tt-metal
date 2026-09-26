#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_broken_components [--user-view]

Options:
    --user-view      Draws broken cores instead of listing them. Broken cores are marked with an 'x' others are marked with a '-'.

Description:
    Verifies that cores halted by triage are still halted and prints a summary of
    broken devices and cores. It relies on run_checks to identify all broken components.
    A device is considered broken if both NoCs hang (timeout on register access).
    A core is considered broken if it cannot be halted or if it resumed execution
    even though triage did not continue it.

Owner:
    adjordjevic-TT
"""

from dataclasses import dataclass
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.device import Device

from run_checks import run as get_run_checks, RunChecks
from triage import ScriptConfig, run_script, triage_field, collection_serializer, ScriptPriority, log_warning_risc
from triage_session import get_triage_session, is_affected_by_cont_bug
from ttexalens.hardware.risc_debug import RiscDebug, RiscLocation

script_config = ScriptConfig(
    depends=["run_checks"],
    priority=ScriptPriority.LOW,
)

_USER_VIEW = False


def risc_display_name(risc_location: RiscLocation) -> str:
    neo = f" (NEO {risc_location.neo_id})" if risc_location.neo_id is not None else ""
    return f"{risc_location.risc_name}{neo}"


def group_broken_cores_by_risc_name(broken_cores: set[RiscLocation]) -> dict[str, set[OnChipCoordinate]]:
    broken_cores_by_risc_name: dict[str, set[OnChipCoordinate]] = {}
    for broken_core in broken_cores:
        risc_name = risc_display_name(broken_core)
        if risc_name not in broken_cores_by_risc_name:
            broken_cores_by_risc_name[risc_name] = set()
        broken_cores_by_risc_name[risc_name].add(broken_core.location)
    return broken_cores_by_risc_name


def draw_broken_cores(broken_cores: set[RiscLocation]) -> str:
    def location_render(location: OnChipCoordinate) -> str:
        riscs_string = ""
        for risc_debug in location.device.get_block(location).all_riscs:
            riscs_string += "x" if risc_debug.risc_location in broken_cores else "-"
        return riscs_string

    rendered = next(iter(broken_cores)).location.device.render(axis_coordinate="noc0", cell_renderer=location_render)
    return str(rendered)


def broken_core_serializer(broken_cores: set[RiscLocation] | str) -> str:
    if isinstance(broken_cores, str):
        return broken_cores
    if not _USER_VIEW:
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
    broken_cores: set[RiscLocation] | str = triage_field("Broken Cores", serializer=broken_core_serializer)


def collect_device_health_summary(run_checks: RunChecks) -> list[DeviceHealthSummary] | None:
    session = get_triage_session()
    device_health_summaries: list[DeviceHealthSummary] = []
    for device in run_checks.devices:
        if session.is_device_broken(device):
            device_health_summaries.append(
                DeviceHealthSummary(device=device, broken_cores=f"[error]Device is broken so it is skipped.[/]")
            )
        else:
            broken_cores = session.get_device_broken_cores(device)
            if broken_cores:
                device_health_summaries.append(DeviceHealthSummary(device=device, broken_cores=broken_cores))
    return device_health_summaries if len(device_health_summaries) > 0 else None


def verify_halted_cores(run_checks: RunChecks) -> None:
    """
    Verify that cores halted by triage are still halted.

    This only tells us anything where continue is suppressed: there cont() is patched to a no-op, so
    a core triage halted cannot have been continued intentionally and finding it running again means
    it broke during triage. Everywhere else triage does continue the cores it halted, and a core that
    is no longer halted is exactly what is expected.
    """
    session = get_triage_session()

    def check_core(risc_debug: RiscDebug) -> None:
        location = risc_debug.risc_location.location
        if not is_affected_by_cont_bug(location.device):
            return None
        if not session.is_halted_core(risc_debug.risc_location):
            return None
        if not risc_debug.is_halted():
            log_warning_risc(
                risc_display_name(risc_debug.risc_location),
                location,
                "Was halted by triage but is no longer halted - core was broken during triage.",
            )
            session.add_broken_core(risc_debug.risc_location)
        return None

    run_checks.run_per_core_check(check_core)


def run(args, context: Context):
    global _USER_VIEW
    _USER_VIEW = bool(args["--user-view"])
    run_checks = get_run_checks(args, context)
    verify_halted_cores(run_checks)
    return collect_device_health_summary(run_checks)


if __name__ == "__main__":
    run_script()
