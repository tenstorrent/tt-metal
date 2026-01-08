# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Real-time telemetry graphs for TT devices (nvtop-style).
Shows per-device line graphs with combined metrics like nvtop.
"""

import time
from collections import deque
from typing import Dict, List, Deque
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box


class TelemetryHistory:
    """Stores historical telemetry data for a single device."""

    def __init__(self, max_points: int = 60):
        """
        Args:
            max_points: Maximum number of data points (default 60)
        """
        self.max_points = max_points
        self.timestamps: Deque[float] = deque(maxlen=max_points)
        self.temperature: Deque[float] = deque(maxlen=max_points)
        self.power: Deque[float] = deque(maxlen=max_points)
        self.voltage: Deque[float] = deque(maxlen=max_points)
        self.current: Deque[float] = deque(maxlen=max_points)
        self.aiclk: Deque[int] = deque(maxlen=max_points)
        self.memory_used: Deque[float] = deque(maxlen=max_points)  # Percentage
        self.dram_pct: Deque[float] = deque(maxlen=max_points)  # DRAM percentage
        self.l1_pct: Deque[float] = deque(maxlen=max_points)  # L1 percentage

    def add_sample(self, device) -> None:
        """Add a telemetry sample from a device."""
        now = time.time()
        self.timestamps.append(now)

        # Temperature
        if hasattr(device, "temperature") and device.temperature >= 0:
            self.temperature.append(device.temperature)
        else:
            self.temperature.append(0.0)

        # Power
        if hasattr(device, "power") and device.power >= 0:
            self.power.append(device.power)
        else:
            self.power.append(0.0)

        # Voltage (convert mV to V)
        if hasattr(device, "voltage_mv") and device.voltage_mv > 0:
            self.voltage.append(device.voltage_mv / 1000.0)
        else:
            self.voltage.append(0.0)

        # Current (convert mA to A)
        if hasattr(device, "current_ma") and device.current_ma > 0:
            self.current.append(device.current_ma / 1000.0)
        else:
            self.current.append(0.0)

        # AICLK
        if hasattr(device, "aiclk_mhz") and device.aiclk_mhz > 0:
            self.aiclk.append(device.aiclk_mhz)
        else:
            self.aiclk.append(0)

        # Memory usage percentage (overall)
        if hasattr(device, "total_dram") and device.total_dram > 0:
            mem_used = device.used_dram + device.used_trace
            mem_pct = (mem_used / device.total_dram) * 100.0
            self.memory_used.append(mem_pct)
        else:
            self.memory_used.append(0.0)

        # DRAM usage percentage
        if hasattr(device, "total_dram") and device.total_dram > 0:
            dram_used = device.used_dram + device.used_trace
            dram_pct = (dram_used / device.total_dram) * 100.0
            self.dram_pct.append(dram_pct)
        else:
            self.dram_pct.append(0.0)

        # L1 usage percentage
        if hasattr(device, "total_l1") and device.total_l1 > 0:
            l1_used = device.used_l1 + device.used_l1_small + device.used_cb
            l1_pct = (l1_used / device.total_l1) * 100.0
            self.l1_pct.append(l1_pct)
        else:
            self.l1_pct.append(0.0)


def render_combined_graph(metrics: List[tuple], width: int = 60, height: int = 12) -> Text:
    """
    Render multiple metrics on same graph with colors (nvtop-style).

    Args:
        metrics: List of (values, max_val, label, color, current_val) tuples
        width: Full graph width in characters
        height: Graph height in lines

    Returns:
        Rich Text object with colored lines and legend overlaid
    """
    # Use full width for graph
    graph_width = width

    if not metrics or all(not m[0] for m in metrics):
        result = Text()
        for i in range(height):
            result.append(" " * width + "\n")
        return result

    # Create canvas with color tracking
    canvas = [[(" ", "white") for _ in range(graph_width)] for _ in range(height)]

    # Plot each metric with its color
    for values, max_val, label, color, current_val in metrics:
        if not values:
            continue

        # Determine scale - use full 0-100 range if max_val specified
        # Don't skip metrics that are all zeros - they should show at bottom
        if max_val is None:
            # Auto-scale with 10% headroom
            non_zero = [v for v in values if v > 0]
            if non_zero:
                max_val = max(non_zero) * 1.1
            else:
                max_val = 100.0  # Default scale if all zeros
        # For fixed max (like 100.0), use it as-is to fill the full chart height
        if max_val == 0:
            max_val = 1.0

        # Sample data to fit width - grow from left to right
        if len(values) > graph_width:
            # More data than width - take the most recent points
            sampled = list(values)[-graph_width:]
        else:
            # Less data than width - use what we have, it will grow from left naturally
            sampled = list(values)

        # Plot line with box-drawing characters
        prev_y = None
        for x, val in enumerate(sampled):
            # Always plot values (including 0, which goes to bottom)
            y_norm = val / max_val
            y = int((1.0 - y_norm) * (height - 1))
            y = max(0, min(height - 1, y))

            if prev_y is not None and x > 0:
                if prev_y == y:
                    # Same level - horizontal line (just overwrite)
                    canvas[y][x] = ("─", color)
                elif prev_y < y:
                    # Going down - draw with nice corners
                    canvas[prev_y][x - 1] = ("┐", color)
                    for y_fill in range(prev_y + 1, y):
                        canvas[y_fill][x - 1] = ("│", color)
                    canvas[y][x - 1] = ("└", color)
                    canvas[y][x] = ("─", color)
                else:  # prev_y > y
                    # Going up - draw with nice corners
                    canvas[prev_y][x - 1] = ("┘", color)
                    for y_fill in range(y + 1, prev_y):
                        canvas[y_fill][x - 1] = ("│", color)
                    canvas[y][x - 1] = ("┌", color)
                    canvas[y][x] = ("─", color)
            else:
                # First point
                canvas[y][x] = ("─", color)

            prev_y = y

    # Convert canvas to Rich Text with colors and add Y-axis labels
    result = Text()

    # Add legend at top (overlaid on graph, nvtop-style)
    for idx, (values, max_val, label, color, current_val) in enumerate(metrics):
        if idx > 0:
            result.append("  ", style="dim")
        result.append(f"{label}: ", style="dim")
        result.append(f"{current_val:.1f}", style=color + " bold")
    result.append("\n")

    # Determine Y-axis labels (0-100 scale for all metrics)
    y_labels = []
    if metrics:
        _, max_val, label, _, _ = metrics[0]
        # Create Y-axis labels for 0, 50, 100 (fixed scale)
        y_labels = [
            (0, "100"),  # Top (100)
            (height // 2, "50"),  # Middle (50)
            (height - 1, "0"),  # Bottom (0)
        ]

    # Add graph lines with Y-axis labels
    for row_idx, row in enumerate(canvas):
        # Add Y-axis label if this row has one
        y_label = ""
        for label_row, label_text in y_labels:
            if row_idx == label_row:
                y_label = label_text.rjust(4) + "│"
                break
        if not y_label:
            y_label = "    │"  # Just the axis line

        result.append(y_label, style="dim")

        # Add graph data
        for char, color in row:
            result.append(char, style=color)
        result.append("\n")

    return result


class GraphWindow:
    """
    Interactive graph window showing telemetry history (nvtop-style).
    """

    def __init__(self, console: Console, history_size: int = 100):
        self.history: Dict[str, TelemetryHistory] = {}
        self.console = console
        self.history_size = history_size

    def _calculate_optimal_layout(self, num_devices: int, terminal_width: int, terminal_height: int) -> tuple:
        """
        Calculate optimal grid layout with preference for wide layouts (more columns, fewer rows).

        For monitoring dashboards, wider layouts (4×8 instead of 8×4) are preferred because:
        - More devices visible side-by-side
        - Better use of wide monitors
        - Easier horizontal scanning

        Returns:
            (rows, cols) tuple for optimal grid layout
        """
        # Simple strategy: Prefer wide layouts
        # For 32 devices: try 8 cols first, then 6, then 4

        if num_devices <= 2:
            return (1, num_devices)
        elif num_devices <= 4:
            return (2, 2)
        elif num_devices <= 8:
            return (2, 4)  # 2 rows × 4 columns (wide)
        elif num_devices <= 16:
            return (2, 8)  # 2 rows × 8 columns (very wide)
        elif num_devices <= 32:
            return (4, 8)  # 4 rows × 8 columns (WIDE for Galaxy systems)
        else:
            # For >32 devices, use 8 columns and calculate rows
            cols = 8
            rows = (num_devices + cols - 1) // cols
            return (rows, cols)

    def update_device(self, device) -> None:
        """Update telemetry history for a device."""
        device_id = device.display_id if hasattr(device, "display_id") else str(device.chip_id)

        if device_id not in self.history:
            self.history[device_id] = TelemetryHistory(max_points=self.history_size)

        self.history[device_id].add_sample(device)

    def render_device_card(self, device_id: str, device, chart_height: int = 30, chart_width: int = 60) -> Panel:
        """
        Render a single device card with combined graphs (nvtop-style).

        Args:
            device_id: Device identifier
            device: Device object with telemetry
            chart_height: Height of chart in rows
            chart_width: Width of chart in columns (dynamic based on terminal size)
        """
        if device_id not in self.history:
            return Panel(f"No data for device {device_id}", title=device_id, border_style="cyan")

        hist = self.history[device_id]
        if len(hist.timestamps) < 2:
            return Panel(f"Collecting data...", title=f"[cyan bold]{device_id}[/]", border_style="cyan")

        layout = Table.grid(padding=0)
        layout.add_column(ratio=1)

        # Current values
        temp_current = hist.temperature[-1] if hist.temperature else 0
        power_current = hist.power[-1] if hist.power else 0
        aiclk_current = hist.aiclk[-1] if hist.aiclk else 0
        mem_pct = hist.memory_used[-1] if hist.memory_used else 0

        # Calculate DRAM and L1 stats for header
        dram_used = device.used_dram + device.used_trace if hasattr(device, "used_dram") else 0
        dram_total = device.total_dram if hasattr(device, "total_dram") else 0
        dram_pct = (dram_used / dram_total * 100.0) if dram_total > 0 else 0.0
        dram_bar_len = int((dram_pct / 100.0) * 20)  # Compact bar for header

        l1_used = device.used_l1 + device.used_l1_small + device.used_cb if hasattr(device, "used_l1") else 0
        l1_total = device.total_l1 if hasattr(device, "total_l1") else 0
        l1_pct = (l1_used / l1_total * 100.0) if l1_total > 0 else 0.0
        l1_bar_len = int((l1_pct / 100.0) * 20)  # Compact bar for header

        def fmt_sz(b):
            u = ["B", "KB", "MB", "GB", "TB"]
            i = 0
            v = float(b)
            while v >= 1024 and i < len(u) - 1:
                v /= 1024.0
                i += 1
            return f"{v:.1f}{u[i]}"

        # Compact header with all stats
        header = Text()
        header.append(f"{device.arch_name if hasattr(device, 'arch_name') else 'Unknown'}", style="cyan bold")
        header.append("  ", style="dim")
        header.append(
            f"Temp: {temp_current:.0f}°C",
            style="red" if temp_current > 80 else "yellow" if temp_current > 70 else "green",
        )
        header.append("  ", style="dim")
        header.append(f"Power: {power_current:.0f}W", style="yellow")
        header.append("  ", style="dim")
        header.append(f"Clock: {aiclk_current}MHz", style="blue")

        layout.add_row(header)

        # Memory bars in header section (nvtop-style with bars) - split into two lines for readability
        # DRAM bar line
        dram_bar = Text()
        dram_bar.append("DRAM [", style="white")
        dram_bar.append("|" * dram_bar_len, style="green")
        dram_bar.append(" " * (20 - dram_bar_len), style="dim")
        dram_bar.append(f"] {fmt_sz(dram_used)}/{fmt_sz(dram_total)}", style="white")
        layout.add_row(dram_bar)

        # L1 bar line
        l1_bar = Text()
        l1_bar.append("L1   [", style="white")
        l1_bar.append("|" * l1_bar_len, style="green")
        l1_bar.append(" " * (20 - l1_bar_len), style="dim")
        l1_bar.append(f"] {fmt_sz(l1_used)}/{fmt_sz(l1_total)}", style="white")
        layout.add_row(l1_bar)

        layout.add_row("")

        # Combined graph with all metrics (nvtop-style)
        # Each metric: (values, max_val, label, color, current_val)
        # All metrics use fixed 0-100 scale for consistent display across full chart height
        dram_current = hist.dram_pct[-1] if hist.dram_pct else 0.0
        l1_current = hist.l1_pct[-1] if hist.l1_pct else 0.0

        metrics = [
            (list(hist.temperature), 100.0, "Temp °C", "red", temp_current),
            (list(hist.power), 100.0, "Power W", "yellow", power_current),
            (list(hist.dram_pct), 100.0, "DRAM %", "cyan", dram_current),
            (list(hist.l1_pct), 100.0, "L1 %", "blue", l1_current),
        ]

        # Use dynamic width and height based on terminal size/number of devices
        combined_graph = render_combined_graph(metrics, width=chart_width, height=chart_height)

        graph_panel = Panel(combined_graph, title=f"[cyan]Telemetry & Memory History[/]", border_style="cyan")
        layout.add_row(graph_panel)

        return Panel(layout, title=f"[cyan bold]{device_id}[/]", border_style="cyan", padding=(0, 1))

    def render_all_devices(self, devices: List) -> Layout:
        """Render graphs for all devices in a grid layout (matrix style)."""
        root_layout = Layout()

        if len(devices) == 0:
            return Layout(Panel("No devices tracked yet", title="Telemetry Graphs"))

        # Get terminal size for dynamic height and width calculation
        terminal_height = self.console.height
        terminal_width = self.console.width
        num_devices = len(devices)

        # Calculate chart height based on terminal size and number of devices
        # Account for: header (3), device headers (~6 per device), panel borders (~2 per device), legend (~1)
        header_overhead = 3
        per_device_overhead = 10  # Device header, bars, borders, padding

        # Intelligently determine optimal rows and columns based on terminal size
        # This maximizes chart visibility while ensuring readability
        rows, cols = self._calculate_optimal_layout(num_devices, terminal_width, terminal_height)

        # Calculate available height per device
        available_height = terminal_height - header_overhead
        height_per_device = available_height // rows
        chart_height = max(15, height_per_device - per_device_overhead)  # Min 15 rows for readability

        # Calculate available width per device
        # Account for: panel borders (4 per device), Y-axis labels (5 chars), padding (4 chars)
        per_device_width_overhead = 13
        available_width = terminal_width // cols
        chart_width = max(40, available_width - per_device_width_overhead)  # Min 40 chars for readability

        # Header
        root_layout.split_column(Layout(name="header", size=3), Layout(name="devices"))

        root_layout["header"].update(
            Panel(
                f"[cyan bold]TT-SMI Telemetry ({len(devices)} devices)[/]  Press Ctrl+C to return to table view",
                style="cyan",
            )
        )

        # Create device cards with dynamic height and width
        device_panels = []
        for dev in devices:
            device_id = dev.display_id if hasattr(dev, "display_id") else str(dev.chip_id)
            device_panels.append(self.render_device_card(device_id, dev, chart_height, chart_width))

        # Apply the calculated layout dynamically (rows × cols)
        num_devices = len(device_panels)

        if num_devices == 1:
            root_layout["devices"].update(device_panels[0])
        elif num_devices == 2:
            # 1×2 layout
            root_layout["devices"].split_row(Layout(device_panels[0]), Layout(device_panels[1]))
        else:
            # Use calculated rows/cols for optimal layout
            # For 32 devices: rows=4, cols=8 → 4 rows × 8 columns (WIDE)

            # Create column layouts
            col_names = [f"col{i}" for i in range(1, cols + 1)]
            root_layout["devices"].split_row(*[Layout(name=name) for name in col_names])

            # Distribute devices across columns (fills row-by-row)
            for col_idx, col_name in enumerate(col_names):
                # Each column gets devices at positions: col_idx, col_idx+cols, col_idx+2*cols, ...
                col_devices = [device_panels[i] for i in range(col_idx, num_devices, cols)]
                if col_devices:
                    root_layout[col_name].split_column(*[Layout(d) for d in col_devices])

        return root_layout
