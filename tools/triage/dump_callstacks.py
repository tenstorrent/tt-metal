#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_callstacks [--all-cores] [--per-core]

Options:
    --all-cores        Show all cores including ones with Go Message = DONE. By default, DONE cores are filtered out.
    --per-core         Show one row per core (no aggregation). By default, rows are aggregated by canonical
                       callstack and show per-device logical-tensix grids.

Description:
    Dumps callstacks for all devices in the system and for every supported risc processor.

    By default:
    - Filters out cores with DONE status (use --all-cores to include them)
    - Aggregates rows by canonical callstack (use --per-core for detailed per-core view)
    - Shows per-device logical-tensix grids indicating which cores are running each callstack pattern:
      * 'R' = this callstack pattern is active on that core
      * '-' = core exists but this pattern is not running there

    Use --per-core to get the original one-row-per-core table with full details.
    Use -v/-vv to show more columns (affects per-core mode only).

    Color output is automatically enabled when stdout is a TTY (terminal) and can be overridden
    with TT_TRIAGE_COLOR environment variable (0=disable, 1=enable).

Owner:
    tt-vjovanovic
"""

from dataclasses import dataclass
from collections import defaultdict

from triage import ScriptConfig, log_check_risc, run_script, triage_field, collection_serializer
from callstack_provider import (
    run as get_callstack_provider,
    CallstackProvider,
    CallstacksData,
    KernelCallstackWithMessage,
    format_callstack_with_message,
)
from run_checks import run as get_run_checks
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.umd_device import TimeoutDeviceRegisterError
from ttexalens.hardware.risc_debug import CallstackEntry

script_config = ScriptConfig(
    depends=["run_checks", "callstack_provider"],
)


@dataclass
class AggregatedCallstackRow:
    """Aggregated view of callstacks grouped by canonical callstack."""

    # Kernel name (should be unique per canonical callstack)
    kernel_name: str | None = triage_field("Kernel Name")

    # Full formatted callstack (frames and/or error text)
    callstack: str = triage_field("Callstack")

    # Per-device logical-tensix grids showing which cores have this callstack pattern
    # Each cell shows 5 RISC status chars: brisc, trisc0, trisc1, trisc2, ncrisc
    device_maps: str = triage_field("Devices (R----=brisc, -R---=trisc0, --R--=trisc1, ---R-=trisc2, ----R=ncrisc)")


def _canonical_frame_key(frame: CallstackEntry) -> tuple[str, str, int]:
    """Extract canonical key from a callstack frame (function, file, line)."""
    fn = frame.function_name or ""
    file = frame.file or ""
    line = frame.line if frame.line is not None else -1
    return (fn, file, line)


def _canonical_callstack_key(kcwm: KernelCallstackWithMessage) -> tuple:
    """
    Generate a canonical key for grouping callstacks.

    - If there are frames: tuple of (fn,file,line) for each frame.
    - If no frames but message is set: ("__ERROR__", message).
    - If no frames and no message: ("__EMPTY__",).
    """
    if kcwm.callstack:
        return tuple(_canonical_frame_key(f) for f in kcwm.callstack)
    if kcwm.message:
        return ("__ERROR__", kcwm.message)
    return ("__EMPTY__",)


def _build_device_grid_metadata(per_core_results) -> dict:
    """
    Build per-device tensix grid metadata from per-core results.

    Returns a dict mapping device_id to:
      {
        'device': Device object,
        'tensix_logical': set of (lx, ly) tuples for all tensix cores,
        'grid_dims': (max_lx, max_ly)
      }
    """
    devices_seen = {}

    # Collect unique devices from per_core_results
    for check in per_core_results:
        device = check.device_description.device
        dev_id = device.id

        if dev_id not in devices_seen:
            # Get all tensix locations for this device
            tensix_locs = device.get_block_locations("functional_workers")

            # Convert to logical coords and track max values
            logical_coords = set()
            max_lx = -1
            max_ly = -1

            for loc in tensix_locs:
                coords = loc.to("logical")
                # Handle nested tuple structure if needed
                if isinstance(coords[0], tuple):
                    lx, ly = coords[0]
                else:
                    lx, ly = coords[0], coords[1]

                # Ensure we have integers
                lx = int(lx)
                ly = int(ly)
                logical_coords.add((lx, ly))

                # Track max values as we go
                if lx > max_lx:
                    max_lx = lx
                if ly > max_ly:
                    max_ly = ly

            if logical_coords:
                devices_seen[dev_id] = {
                    "device": device,
                    "tensix_logical": logical_coords,
                    "grid_dims": (max_lx, max_ly),
                }

    return devices_seen


def _format_risc_status(active_riscs: set[str]) -> str:
    """
    Format RISC status as 5-character string: brisc, trisc0, trisc1, trisc2, ncrisc.

    Returns:
        5-character string with 'R' for active RISCs and '-' for inactive ones
    """
    risc_order = ["brisc", "trisc0", "trisc1", "trisc2", "ncrisc"]
    return "".join("R" if risc in active_riscs else "-" for risc in risc_order)


def _format_device_grid(device_id, grid_dims, tensix_locs, active_core_riscs):
    """
    Format a device grid showing which tensix cores have a specific callstack pattern.

    Args:
        device_id: Device ID
        grid_dims: (max_lx, max_ly) tuple
        tensix_locs: set of all (lx, ly) tensix coordinates for this device
        active_core_riscs: dict mapping (lx, ly) to set of active risc names

    Returns:
        Multi-line string with grid visualization
    """
    max_lx, max_ly = grid_dims

    # Build header
    lines = []
    lines.append(f"=== Device {device_id} ===")

    # Header row with X coordinates
    header = "    "
    for x in range(max_lx + 1):
        header += f"{x:02d}     "
    lines.append(header)

    # Build each row
    for y in range(max_ly + 1):
        row = f"{y:02d}  "
        for x in range(max_lx + 1):
            if (x, y) in active_core_riscs:
                # Show which RISCs are active on this core
                cell = _format_risc_status(active_core_riscs[(x, y)])
            elif (x, y) in tensix_locs:
                # Core exists but no RISCs with this pattern
                cell = "-----"
            else:
                # Not a tensix core at all
                cell = "     "

            row += cell
            if x < max_lx:
                row += "  "  # Two spaces between columns
        lines.append(row)

    return "\n".join(lines)


def _build_device_maps_string(dev_logical_keys, device_metadata):
    """
    Build a multi-line string with per-device grids showing active cores.

    Args:
        dev_logical_keys: list of (dev_id, lx, ly, risc_name) tuples
        device_metadata: dict from device_id to metadata (from _build_device_grid_metadata)

    Returns:
        Multi-line string with one grid per device
    """
    # Group by device and core, tracking which RISCs are active
    per_device_core_riscs = defaultdict(lambda: defaultdict(set))
    for dev_id, lx, ly, risc_name in dev_logical_keys:
        per_device_core_riscs[dev_id][(lx, ly)].add(risc_name)

    # Build grids for each device
    grids = []
    for dev_id in sorted(per_device_core_riscs.keys()):
        if dev_id not in device_metadata:
            continue

        meta = device_metadata[dev_id]
        grid_str = _format_device_grid(dev_id, meta["grid_dims"], meta["tensix_logical"], per_device_core_riscs[dev_id])
        grids.append(grid_str)
        grids.append("")  # Empty line after each device

    return "\n".join(grids)


def _aggregate_callstacks_v3(per_core_results) -> list[AggregatedCallstackRow]:
    """
    Aggregate per-core callstack results into groups by canonical callstack.

    Returns a list of AggregatedCallstackRow objects, sorted by descending count.
    """
    if not per_core_results:
        return []

    # Build device grid metadata
    device_metadata = _build_device_grid_metadata(per_core_results)

    groups: dict[tuple, dict] = {}

    for check in per_core_results:
        data: CallstacksData = check.result
        if data is None:
            continue

        kcwm = data.kernel_callstack_with_message
        key = _canonical_callstack_key(kcwm)

        if key not in groups:
            # Extract kernel name from dispatcher core data
            kernel_name = getattr(data.dispatcher_core_data, "kernel_name", None)
            groups[key] = {
                "kcwm": kcwm,
                "kernel_name": kernel_name,
                "dev_logical_keys": [],  # (dev_id, lx, ly, risc_name)
            }

        g = groups[key]

        dev_id = check.device_description.device.id
        coords = check.location.to("logical")
        # Handle nested tuple structure if needed
        if isinstance(coords[0], tuple):
            lx, ly = coords[0]
        else:
            lx, ly = coords[0], coords[1]

        # Ensure we have integers
        lx = int(lx)
        ly = int(ly)
        g["dev_logical_keys"].append((dev_id, lx, ly, check.risc_name))

    rows: list[AggregatedCallstackRow] = []
    for g in groups.values():
        kcwm = g["kcwm"]
        formatted_callstack = format_callstack_with_message(kcwm)

        # Build device maps
        device_maps = _build_device_maps_string(g["dev_logical_keys"], device_metadata)

        rows.append(
            AggregatedCallstackRow(
                kernel_name=g["kernel_name"],
                callstack=formatted_callstack,
                device_maps=device_maps,
            )
        )

    # Sort by descending count (number of risc instances), then callstack
    rows.sort(key=lambda r: (-len(r.device_maps), r.callstack))
    return rows


def dump_callstacks(
    location: OnChipCoordinate,
    risc_name: str,
    callstack_provider: CallstackProvider,
    show_all_cores: bool = False,
) -> CallstacksData | None:
    try:
        # Skip DONE cores unless --all-cores is specified
        if not show_all_cores:
            dispatcher_core_data = callstack_provider.dispatcher_data.get_cached_core_data(location, risc_name)
            if dispatcher_core_data.go_message == "DONE":
                return None
        return callstack_provider.get_callstacks(location, risc_name)
    except TimeoutDeviceRegisterError:
        raise
    except Exception as e:
        log_check_risc(
            risc_name,
            location,
            False,
            f"[warning]Failed to dump callstacks: {e}[/]",
        )
        return None


def run(args, context: Context):
    show_all_cores: bool = args["--all-cores"]
    per_core: bool = args["--per-core"]
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth"]
    run_checks = get_run_checks(args, context)
    callstack_provider = get_callstack_provider(args, context)
    per_core_results = run_checks.run_per_core_check(
        lambda location, risc_name: dump_callstacks(
            location,
            risc_name,
            callstack_provider,
            show_all_cores,
        ),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )

    # If --per-core is specified or no results, return per-core view
    if per_core or not per_core_results:
        return per_core_results

    # Default: return aggregated view
    return _aggregate_callstacks_v3(per_core_results)


if __name__ == "__main__":
    run_script()
