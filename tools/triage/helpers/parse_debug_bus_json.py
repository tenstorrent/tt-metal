#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Parse debug bus signal groups from JSON produced by tt-triage and print selected groups
of selected block types for selected devices and locations. Each group is printed as its own table: first column
is the group name (header), second column is "Value" (header); rows are decoded
signal names and values.

Usage:
    parse_debug_bus_json.py <json_path> [--groups=<pattern>] [--devices=<devices>] [--block-types=<types>] [--locations=<locations>]

Options:
    -h --help                Show this message.
    --groups=<pattern>       Regex to match group names (comma-separated). Default: all.
    --devices=<devices>      Comma-separated device IDs (e.g. 0,1). Default: all.
    --block-types=<types>    Comma-separated block types (e.g. tensix, idle_eth). Default: all.
    --locations=<locations>  Colon(:)-separated locations: x-y(e.g. 1-1):x,y(e.g. 0,0). Default: all.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from docopt import docopt
from rich.console import Console
from rich.table import Table

from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.debug_bus_signal_store import DebugBusSignalStore
from ttexalens.hardware.noc_block import NocBlock
from ttexalens.hardware.wormhole.eth_block import WormholeEthBlock
from ttexalens.hardware.wormhole.functional_worker_block import WormholeFunctionalWorkerBlock
from ttexalens.hardware.blackhole.eth_block import BlackholeEthBlock
from ttexalens.hardware.blackhole.functional_worker_block import BlackholeFunctionalWorkerBlock

from tools.triage.utils import ERROR, INFO, WARN


def _get_selected_devices(devices_arg: str | None, device_keys: set[str]) -> list[str]:
    """Return set of device keys to include (e.g. {'Device 0', 'Device 1'})."""
    if devices_arg is None:
        return device_keys
    selected = []
    for d in devices_arg.split(","):
        d = d.strip()
        if d and f"Device {d}" in device_keys:
            selected.append(f"Device {d}")
    return selected


def _get_selected_block_types(block_types_arg: str | None, block_types: set[str]) -> list[str]:
    """Return set of block types to include (e.g. {'tensix', 'idle_eth'})."""
    if block_types_arg is None:
        return block_types
    selected = []
    for b in block_types_arg.split(","):
        b = b.strip()
        if b and b in block_types:
            selected.append(b)
    return selected


def _get_selected_locations(locations_arg: str | None, location_keys: list[str]) -> list[str]:
    """Return set of location keys to include. Keys are like 'location: 1-1 (0,0)'.
    User can reference by x-y (e.g. 1-1), x,y (e.g. 0,0), or full string. Split by ', '
    """
    if not locations_arg or not locations_arg.strip():
        return location_keys
    requested = set()
    for s in locations_arg.split(":"):
        s = s.strip()
        if s:
            if "," in s:
                requested.add(f"({s})")
            elif "-" in s:
                requested.add(f" {s} ")
    selected = []
    for key in location_keys:
        if any(spec in key for spec in requested):
            selected.append(key)
    return selected


def _get_selected_groups(groups_arg: str | None, groups: list[str]) -> list[str]:
    """Return subset of groups whose names match any of the comma-separated regex patterns."""
    if not groups_arg or not groups_arg.strip():
        return groups
    patterns = [p.strip() for p in groups_arg.split(",") if p.strip()]
    selected = []
    for group in groups:
        for pat in patterns:
            if re.search(pat, group):
                selected.append(group)
                break
    return selected


_console = Console()


def _print_signal_group_table(group_name: str, hex_value: str, debug_bus) -> None:
    """Print one table per group: column 1 = group name (header), column 2 = Value (header).
    Rows: signal names and corresponding values when decoded; otherwise single row with raw hex.
    """
    assert debug_bus is not None
    from ttexalens.debug_bus_signal_store import SignalGroupSample

    sample = SignalGroupSample(int(hex_value, 16), debug_bus.signal_groups[group_name])
    rows = [[signal_name, str(value)] for signal_name, value in sample.items()]

    table = Table(show_header=True, header_style="bold")
    table.add_column(group_name, style="dim")
    table.add_column("Value")
    for name, value in rows:
        table.add_row(name, value)
    _console.print(table)


def _print_groups_for_location(
    selected_groups: list[str],
    debug_bus: DebugBusSignalStore,
    groups_data: dict[str, str],
) -> None:
    """Print one table per group; each table has group name and Value as column headers."""
    for group in selected_groups:
        _print_signal_group_table(group, groups_data[group], debug_bus)


def create_noc_block(device_arch: str, block_type: str, location: OnChipCoordinate) -> NocBlock:
    match device_arch:
        case "wormhole_b0":
            match block_type:
                case "tensix":
                    return WormholeFunctionalWorkerBlock(location)
                case "idle_eth":
                    return WormholeEthBlock(location)
                case _:
                    raise ValueError(f"Unknown block type: {block_type}")
        case "blackhole":
            match block_type:
                case "tensix":
                    return BlackholeFunctionalWorkerBlock(location)
                case "idle_eth":
                    return BlackholeEthBlock(location)
                case _:
                    raise ValueError(f"Unknown block type: {block_type}")
        case _:
            raise ValueError(f"Unknown architecture: {device_arch}")


def main() -> None:
    args = docopt(__doc__, argv=sys.argv[1:])

    json_path = Path(args["<json_path>"])
    if not json_path.is_file():
        ERROR(f"Error: file not found: {json_path}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        ERROR(f"Error loading JSON: {e}")
        return

    context = init_ttexalens()
    mock_location = OnChipCoordinate.create("0,0", context.devices[0])

    selected_devices = _get_selected_devices(args["--devices"], data.keys())
    printed_any = False

    for device_key in sorted(selected_devices, key=lambda k: int(k.split()[-1])):
        device_data = data[device_key]
        device_arch = device_data["arch"]
        selected_block_types = _get_selected_block_types(args["--block-types"], device_data["block_types"].keys())
        if not selected_block_types:
            continue
        for block_type in selected_block_types:
            block_data = device_data["block_types"][block_type]
            try:
                noc_block = create_noc_block(device_arch, block_type, mock_location)
                debug_bus = noc_block.debug_bus
                assert debug_bus is not None
            except ValueError as e:
                ERROR(f"Error: {e}")
                continue
            selected_locations = _get_selected_locations(args["--locations"], list(block_data.keys()))
            if not selected_locations:
                continue

            for loc_key in sorted(selected_locations):
                loc_data = block_data[loc_key]
                groups_data = loc_data["debug_bus_signal_groups"]
                selected_groups = _get_selected_groups(args["--groups"], list(groups_data.keys()))
                if not selected_groups:
                    continue
                printed_any = True
                INFO(f"\n[{device_key}] {block_type} — {loc_key}")
                WARN(f"Failed RISCs: {loc_data['failed_riscs']}")
                _print_groups_for_location(selected_groups, debug_bus, groups_data)

    if not printed_any:
        WARN("No matching data to display.")


if __name__ == "__main__":
    main()
