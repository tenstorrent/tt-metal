#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_galaxy_mesh [--galaxy-shape=<RxC>]

Options:
    --galaxy-shape=<RxC>   Mesh shape, "<rows>x<cols>". Only "8x4" (default) and "4x8" are
                           supported because they correspond to the physical Galaxy tray
                           topology. Anything else falls back to 8x4 with a warning.

Description:
    Prints two tables for Galaxy hosts:
      1. tt-smi-style "Mapping of trays to devices on the galaxy" (Tray Number / Tray Bus ID /
         PCI Dev ID).
      2. A topology-aware mesh, each cell labelled `T<tray>:N<idx_in_tray> (Device <id>)` and
         showing `(idle)` or `{host_assigned_id}: {op_name}` lines for any op currently
         running there.

    The device label is suffixed with `[!]` when any op on that device is **not the majority
    op** (the op running on the most distinct devices). Tie-break is deterministic —
    alphabetical by op name.

    Depends on:
      - `running_ops_aggregation` for the per-host_assigned_id aggregation,
      - `device_locations` for per-device PCI/tray info.

Owner:
    miacim
"""

from __future__ import annotations

from operation_runtime_map import _decode_base_program_id
from device_locations import run as get_device_locations
from galaxy_topology import (
    SUPPORTED_GALAXY_SHAPES,
    device_to_cell,
)
from rich.console import Console
from rich.table import Table
from running_ops_aggregation import run as get_running_ops_aggregation
from triage import ScriptConfig, log_check, run_script
from ttexalens.context import Context


script_config = ScriptConfig(depends=["running_ops_aggregation", "device_locations"])


def _parse_galaxy_shape(shape: str | None) -> tuple[int, int] | None:
    """Parse "<rows>x<cols>" and return it only if it's in SUPPORTED_GALAXY_SHAPES."""
    if not shape:
        return None
    parts = shape.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        return None
    try:
        rows, cols = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    candidate = (rows, cols)
    if candidate not in SUPPORTED_GALAXY_SHAPES:
        return None
    return candidate


def run(args, context: Context):
    raw_shape = args["--galaxy-shape"]
    parsed_shape = _parse_galaxy_shape(raw_shape) if raw_shape else None
    if raw_shape and parsed_shape is None:
        supported = sorted(f"{r}x{c}" for r, c in SUPPORTED_GALAXY_SHAPES)
        log_check(
            False,
            f"Unsupported --galaxy-shape={raw_shape!r}; only {supported} match the physical "
            f"Galaxy tray topology. Falling back to 8x4.",
        )
    rows, cols = parsed_shape if parsed_shape is not None else (8, 4)

    bundle = get_running_ops_aggregation(args, context)
    aggregations = bundle.aggregations
    runtime_id_to_operation = bundle.runtime_id_to_operation

    locations = get_device_locations(args, context)
    tray_to_devices = locations.tray_to_devices()

    if not tray_to_devices:
        log_check(
            False,
            "Galaxy mesh suppressed: no devices map to a UBB tray (non-Galaxy host or PCI " "BDF lookup failed).",
        )
        return None

    # device_id -> set of (host_assigned_id, op_name)
    device_to_ops: dict[int, set[tuple[int, str]]] = {}
    # op_name -> set of device labels running it (used to find the majority op)
    running_ops: dict[str, set[str]] = {}
    for host_assigned_id, agg in aggregations.items():
        # Resolve the raw runtime id (raw or decoded) so we can look up the op name.
        op_info = runtime_id_to_operation.get_raw(host_assigned_id)
        if op_info is None:
            op_info = runtime_id_to_operation.get_raw(_decode_base_program_id(host_assigned_id))
        op_name = (op_info.name if op_info else "") or "N/A"
        running_ops.setdefault(op_name, set()).update(agg.device_labels)
        for dev_label in agg.device_labels:
            try:
                dev_id = int(dev_label, 0)
            except (TypeError, ValueError):
                continue
            device_to_ops.setdefault(dev_id, set()).add((host_assigned_id, op_name))

    running_ops_count: dict[str, int] = {name: len(devs) for name, devs in running_ops.items()}
    # Deterministic tie-break: highest count first, alphabetical name on tie.
    majority_op: str | None = (
        sorted(running_ops_count.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if running_ops_count else None
    )

    console = Console()

    # 1. tt-smi-style "Mapping of trays to devices" table.
    ubb_table = locations.ubb_table or {}
    tray_table = Table(title="Mapping of trays to devices on the galaxy:", title_justify="left")
    tray_table.add_column("Tray Number")
    tray_table.add_column("Tray Bus ID")
    tray_table.add_column("PCI Dev ID")
    for tray_num in sorted(tray_to_devices):
        bus_id = ubb_table.get(tray_num)
        bus_id_str = f"0x{bus_id:02x}" if bus_id is not None else "?"
        tray_table.add_row(
            f"{tray_num}",
            bus_id_str,
            ",".join(str(d) for d in tray_to_devices[tray_num]),
        )
    console.print(tray_table)

    # 2. Topology-aware Galaxy mesh.
    mesh = Table(
        title=f"Galaxy mesh ({rows}x{cols}) — current op per device",
        title_justify="left",
        show_lines=True,
    )
    for c in range(cols):
        mesh.add_column(f"col {c}", justify="left")

    grid: list[list[str]] = [["" for _ in range(cols)] for _ in range(rows)]
    for tray_num, devs in tray_to_devices.items():
        for dev_id in devs:
            try:
                r, c = device_to_cell(dev_id, tray_num, devs, (rows, cols))
            except (ValueError, KeyError):
                continue
            idx_in_tray = devs.index(dev_id)
            label = f"T{tray_num}:N{idx_in_tray} (Device {dev_id})"
            entries = device_to_ops.get(dev_id)
            if entries:
                # If any op on this device isn't the majority op, mark the device label so
                # outlier devices jump out at a glance. Greppable: `grep '\[!\]'`. ASCII-only
                # because emoji break rich.Table column-width math.
                if majority_op is not None and any((name or "N/A") != majority_op for _, name in entries):
                    label = f"{label} [!]"
                ops_lines = "\n".join(f"{hid}: {name or 'N/A'}" for hid, name in sorted(entries))
                grid[r][c] = f"{label}\n{ops_lines}"
            else:
                grid[r][c] = f"{label}\n(idle)"

    for r in range(rows):
        mesh.add_row(*grid[r])

    console.print(mesh)
    return None


if __name__ == "__main__":
    run_script()
