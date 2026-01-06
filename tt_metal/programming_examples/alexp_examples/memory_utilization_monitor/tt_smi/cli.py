#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for TT-SMI."""

import click
from rich.console import Console

from .core import get_devices, cleanup_dead_processes
from .ui.dashboard import Dashboard
from .ui.graphs import GraphWindow


@click.command()
@click.option("-w", "--watch", is_flag=True, help="Watch mode (continuous updates)")
@click.option("-r", "--refresh", default=500, type=int, help="Refresh interval in milliseconds")
@click.option("-g", "--graph", is_flag=True, help="Show telemetry graphs (watch mode only)")
@click.option("--shm-only", is_flag=True, help="Only read SHM (no device access)")
@click.option("--cleanup/--no-cleanup", default=True, help="Enable/disable dead process cleanup")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--reset", is_flag=True, help="Reset all Tenstorrent devices and exit")
@click.option("--reset-m3", is_flag=True, help="Reset with M3 board-level reset (deeper, slower)")
def main(
    watch: bool,
    refresh: int,
    graph: bool,
    shm_only: bool,
    cleanup: bool,
    output_json: bool,
    reset: bool,
    reset_m3: bool,
):
    """TT-SMI: Tenstorrent System Management Interface

    Beautiful, interactive monitoring for Tenstorrent devices.

    Examples:

        tt-smi-ui                    # Single snapshot

        tt-smi-ui -w                 # Watch mode (table view)

        tt-smi-ui -w -g              # Watch mode with telemetry graphs

        tt-smi-ui -w -r 500          # Watch mode, 500ms refresh

        tt-smi-ui --shm-only         # SHM-only (non-invasive)

        tt-smi-ui --json             # JSON output

        tt-smi-ui --reset            # Reset all devices
    """
    console = Console()

    try:
        # Cleanup dead processes if enabled
        if cleanup:
            cleaned = cleanup_dead_processes()
            if cleaned > 0 and not watch:
                console.print(f"[green]✓ Cleaned up {cleaned} dead process(es)[/green]")

        # Get devices
        devices = get_devices(shm_only=shm_only)

        if not devices:
            console.print("[red]No Tenstorrent devices found![/red]")
            return 1

        # Update telemetry and memory for all devices (unless in shm_only mode)
        if not shm_only:
            from .core import update_telemetry_parallel, update_memory

            update_telemetry_parallel(devices, timeout=1.0)
            # Also update memory stats (from SHM)
            for dev in devices:
                try:
                    update_memory(dev)
                except Exception:
                    pass

        # JSON output
        if output_json:
            import json

            data = {
                "devices": [
                    {
                        "id": dev.display_id,
                        "arch": dev.arch_name,
                        "is_remote": dev.is_remote,
                        "telemetry": {
                            "temperature": dev.temperature,
                            "power": dev.power,
                            "aiclk_mhz": dev.aiclk_mhz,
                            "status": dev.telemetry_status,
                        },
                        "memory": {
                            "dram": {"used": dev.used_dram, "total": dev.total_dram},
                            "l1": {"used": dev.used_l1, "total": dev.total_l1},
                            "l1_small": dev.used_l1_small,
                            "trace": dev.used_trace,
                            "cb": dev.used_cb,
                            "kernel": dev.used_kernel,
                        },
                        "processes": dev.processes,
                    }
                    for dev in devices
                ]
            }
            console.print_json(data=data)
            return 0

        # Dashboard rendering
        dashboard = Dashboard(console)

        if watch:
            # Watch mode with parallel telemetry updates and memory refresh
            from .core import update_telemetry_parallel, update_memory

            # Create graph window if requested
            graph_window = GraphWindow(console, history_size=100) if graph else None

            dashboard.watch(
                lambda: get_devices(shm_only=shm_only),
                refresh_ms=refresh,
                update_telemetry_parallel_func=None if shm_only else update_telemetry_parallel,
                update_memory_func=update_memory,
                graph_window=graph_window,
            )
        else:
            # Single snapshot
            if graph:
                console.print("[yellow]⚠ Graph mode only available in watch mode (-w)[/yellow]")
            dashboard.print_snapshot(devices)

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


if __name__ == "__main__":
    exit(main())
