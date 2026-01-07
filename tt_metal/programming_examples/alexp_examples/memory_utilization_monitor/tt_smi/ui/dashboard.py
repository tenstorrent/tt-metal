# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Rich-based dashboard for TT-SMI."""

from datetime import datetime
from typing import List
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box
import time

from ..core import Device, format_bytes


class Dashboard:
    """Live dashboard using Rich library."""

    def __init__(self, console: Console = None):
        self.console = console or Console()

    def render_header(self) -> Panel:
        """Render header panel."""
        now = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        header_text = Text()
        header_text.append("tt-smi", style="bold cyan")
        header_text.append(" - Tenstorrent System Management Interface", style="cyan")
        header_text.append(f"\n{now}", style="white")
        return Panel(header_text, box=box.DOUBLE, border_style="cyan")

    def render_device_table(self, devices: List[Device]) -> Table:
        """Render main device table."""
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold")

        table.add_column("ID", style="cyan", width=12)
        table.add_column("Arch", width=12)
        table.add_column("Temp", width=5)
        table.add_column("Power", width=5)
        table.add_column("AICLK", width=10)
        table.add_column("DRAM Usage", width=18)
        table.add_column("L1 Usage", width=18)
        table.add_column("Status", width=6)

        for dev in devices:
            # Temperature
            if dev.temperature >= 0:
                temp_str = f"{int(dev.temperature)}°C"
                temp_style = "red" if dev.temperature > 80 else "yellow" if dev.temperature > 70 else "green"
            else:
                temp_str = "N/A"
                temp_style = "dim"

            # Power
            if dev.power >= 0:
                power_str = f"{int(dev.power)}W"
                power_style = "yellow"
            else:
                power_str = "N/A"
                power_style = "dim"

            # AICLK
            if dev.aiclk_mhz > 0:
                aiclk_str = f"{dev.aiclk_mhz} MHz"
                aiclk_style = "white"
            else:
                aiclk_str = "N/A"
                aiclk_style = "dim"

            # DRAM usage
            if dev.has_shm:
                dram_total = dev.used_dram + dev.used_trace
                dram_str = f"{format_bytes(dram_total)} / {format_bytes(dev.total_dram)}"
                dram_pct = (dram_total / dev.total_dram * 100) if dev.total_dram > 0 else 0
                dram_style = (
                    "red" if dram_pct > 90 else "yellow" if dram_pct > 70 else "green" if dram_total > 0 else "white"
                )
            else:
                dram_str = f"0B / {format_bytes(dev.total_dram)}"
                dram_style = "dim"

            # L1 usage
            if dev.has_shm:
                l1_total = dev.used_l1 + dev.used_l1_small + dev.used_cb
                l1_str = f"{format_bytes(l1_total)} / {format_bytes(dev.total_l1)}"
                l1_pct = (l1_total / dev.total_l1 * 100) if dev.total_l1 > 0 else 0
                l1_style = "red" if l1_pct > 90 else "yellow" if l1_pct > 70 else "green" if l1_total > 0 else "white"
            else:
                l1_str = f"0B / {format_bytes(dev.total_l1)}"
                l1_style = "dim"

            # Status
            status_str = dev.telemetry_status
            status_style = "green" if status_str == "OK" else "yellow"

            table.add_row(
                dev.display_id,
                dev.arch_name,
                Text(temp_str, style=temp_style),
                Text(power_str, style=power_style),
                Text(aiclk_str, style=aiclk_style),
                Text(dram_str, style=dram_style),
                Text(l1_str, style=l1_style),
                Text(status_str, style=status_style),
            )

        return table

    def render_process_table(self, devices: List[Device]) -> Table:
        """Render per-process memory usage table (shows only devices with active processes)."""
        # Filter to show only devices with processes to reduce clutter on large systems
        devices_with_processes = [dev for dev in devices if dev.processes]

        if not devices_with_processes:
            return None

        table = Table(
            title=f"Per-Process Memory Usage ({len(devices_with_processes)} of {len(devices)} devices active)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )

        table.add_column("Dev", style="cyan", width=12)
        table.add_column("PID", width=8)
        table.add_column("Process", width=16)
        table.add_column("DRAM", width=12)
        table.add_column("L1", width=10)
        table.add_column("L1 Small", width=10)
        table.add_column("Trace", width=10)
        table.add_column("CB", width=10)

        has_processes = False
        for dev in devices_with_processes:
            for proc in dev.processes:
                has_processes = True
                proc_name = proc["name"]
                if len(proc_name) > 14:
                    proc_name = proc_name[:11] + "..."

                table.add_row(
                    dev.display_id,
                    str(proc["pid"]),
                    proc_name,
                    Text(format_bytes(proc["dram"]), style="green" if proc["dram"] > 0 else "dim"),
                    Text(format_bytes(proc["l1"]), style="green" if proc["l1"] > 0 else "dim"),
                    Text(format_bytes(proc["l1_small"]), style="green" if proc["l1_small"] > 0 else "dim"),
                    Text(format_bytes(proc["trace"]), style="green" if proc["trace"] > 0 else "dim"),
                    Text(format_bytes(proc["cb"]), style="green" if proc["cb"] > 0 else "dim"),
                )

        if not has_processes:
            return None

        return table

    def render_snapshot(self, devices: List[Device]) -> Layout:
        """Render complete snapshot."""
        layout = Layout()

        layout.split_column(
            Layout(self.render_header(), size=4),
            Layout(self.render_device_table(devices), name="devices"),
            Layout(name="processes", size=0),
        )

        proc_table = self.render_process_table(devices)
        if proc_table:
            layout["processes"].update(proc_table)
            # Dynamic size based on terminal height and actual process count
            # Count devices with processes
            devices_with_procs = sum(1 for dev in devices if dev.processes)
            max_size = self.console.height // 2  # Use up to half the terminal for process table
            estimated_size = devices_with_procs * 3 + 5  # Header + rows
            layout["processes"].size = min(estimated_size, max_size, 50)  # Cap at 50 to leave room

        return layout

    def print_snapshot(self, devices: List[Device]):
        """Print a single snapshot (non-watch mode)."""
        self.console.print(self.render_header())
        self.console.print(self.render_device_table(devices))

        proc_table = self.render_process_table(devices)
        if proc_table:
            self.console.print("\n")
            self.console.print(proc_table)

    def watch(
        self,
        get_devices_func,
        refresh_ms: int = 1000,
        update_telemetry_parallel_func=None,
        update_memory_func=None,
        graph_window=None,
    ):
        """Live watch mode with auto-refresh.

        Args:
            get_devices_func: Function to get device list
            refresh_ms: Refresh interval in milliseconds
            update_telemetry_parallel_func: Function to update telemetry in parallel
            update_memory_func: Function to update memory stats
            graph_window: GraphWindow instance for telemetry graphs (or None for table view)
        """
        # Import cleanup function once for efficiency
        from ..core import cleanup_dead_processes

        try:
            devices = get_devices_func()
            if update_telemetry_parallel_func:
                update_telemetry_parallel_func(devices, timeout=1.0)
            if update_memory_func:
                for dev in devices:
                    try:
                        update_memory_func(dev)
                    except Exception:
                        pass

            # Calculate optimal screen refresh rate based on data refresh interval
            # Cap at 10 FPS (100ms) for smooth updates without excessive CPU
            screen_refresh_rate = min(10, max(2, 1000 / refresh_ms))

            # If graph mode, show graphs instead of table
            if graph_window:
                # Initial population
                for dev in devices:
                    graph_window.update_device(dev)

                with Live(
                    graph_window.render_all_devices(devices),
                    refresh_per_second=screen_refresh_rate,
                    console=self.console,
                    screen=True,
                ) as live:
                    while True:
                        time.sleep(refresh_ms / 1000.0)

                        # Clean up dead processes on every iteration (critical for memory accuracy)
                        try:
                            cleanup_dead_processes()
                        except Exception:
                            pass

                        devices = get_devices_func()

                        # Update telemetry each iteration (parallel)
                        if update_telemetry_parallel_func:
                            update_telemetry_parallel_func(devices, timeout=1.0)

                        # Update memory stats
                        if update_memory_func:
                            for dev in devices:
                                try:
                                    update_memory_func(dev)
                                except Exception:
                                    pass

                        # Update graph history
                        for dev in devices:
                            graph_window.update_device(dev)

                        live.update(graph_window.render_all_devices(devices))
            else:
                # Normal table view
                with Live(
                    self.render_snapshot(devices),
                    refresh_per_second=screen_refresh_rate,
                    console=self.console,
                    screen=True,
                ) as live:
                    while True:
                        time.sleep(refresh_ms / 1000.0)

                        # Clean up dead processes on every iteration (critical for memory accuracy)
                        try:
                            cleanup_dead_processes()
                        except Exception:
                            pass

                        devices = get_devices_func()

                        # Update telemetry each iteration (parallel)
                        if update_telemetry_parallel_func:
                            update_telemetry_parallel_func(devices, timeout=1.0)

                        # Update memory stats
                        if update_memory_func:
                            for dev in devices:
                                try:
                                    update_memory_func(dev)
                                except Exception:
                                    pass

                        live.update(self.render_snapshot(devices))
        except KeyboardInterrupt:
            pass
