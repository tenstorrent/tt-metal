#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# ttnvtop_py: Python/Textual day-to-day live monitor. Reads the same
# /dev/shm files the C++ ttnvtop reads (per-chip util SHM + program
# registry) and renders them in a reactive TUI with sortable tables,
# mouse support, keyboard scrubbing, and proper layout. Lighter on the
# eyes than the hand-rolled ANSI viewer, easier to extend.
#
# Usage:
#   pip install textual rich      # one-time
#   python tt_metal/tools/ttnvtop/scripts/ttnvtop_py.py
#
# Requires the C++ ttnvtop-collector to be running (it publishes the SHM
# files). For program names, also requires the workload to have been
# launched with TTNVTOP_REGISTER_PROGRAMS=1.
#
# Layout:
#   ┌─ Top bar: status + counts ─────────────────────────────┐
#   ├─ Chips: heatmap grid per chip ───┬── Programs table ───┤
#   ├─ Timeline (sparkline gantt) ────────────────────────────┤
#   ├─ History (sortable, every program seen) ───────────────┤
#   └─ Footer: hotkeys ──────────────────────────────────────┘
#
# Keys (more discoverable in-app):
#   q          quit
#   /          filter program names (focus active table's filter)
#   s          cycle sort column on focused table
#   r          reset history (start fresh)

from __future__ import annotations

import argparse
import collections
import glob
import os
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Static


# ─── On-disk schemas (mirror common/shm_schema.hpp + program_registry.hpp) ──

# Per-chip util SHM header. See shm_schema.hpp.
HEADER_FMT = "<4sHHQIIQQIIII4I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
PER_CORE_FMT = "<6B10H2x3I"
PER_CORE_SIZE = struct.calcsize(PER_CORE_FMT)

# Program registry. See program_registry.hpp (v3 schema, 128-byte entry).
REG_HEADER_SIZE = 48
REG_ENTRY_SIZE = 128
REG_CAPACITY = 16384


def _read_chip(path: str):
    """Return (asic_id, aiclk_mhz, last_update_us, [(noc_x,noc_y,lx,ly,is_remote,kid,f‰,s‰,d‰), ...])"""
    with open(path, "rb") as f:
        data = f.read()
    hdr = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    # hdr layout (mirrors UtilShmHeader, common/shm_schema.hpp:42):
    #   0 magic, 1 version, 2 struct_size, 3 asic_id, 4 arch_id, 5 signal_sources,
    #   6 epoch_us, 7 last_update_us, 8 num_cores, 9 host_assigned_id, 10 collector_pid,
    #   11 aiclk_mhz, 12 dram_rd_mbps, 13 dram_wr_mbps, 14 dram_peak_mbps, 15 reserved[0]
    # D6: this comment used to list epoch_us and last_update_us in the wrong order and
    # the read below took hdr[6] (epoch_us, fixed at collector start) for last_update_us,
    # which would have made every staleness/age figure read as a constant "seconds since
    # the collector started". Same shape as the shm_probe.py:27 mislabelling.
    asic_id = hdr[3]
    last_update_us = hdr[7]
    n_cores = hdr[8]
    aiclk_mhz = hdr[11]
    rows = []
    for i in range(n_cores):
        r = struct.unpack(
            PER_CORE_FMT,
            data[HEADER_SIZE + i * PER_CORE_SIZE : HEADER_SIZE + (i + 1) * PER_CORE_SIZE],
        )
        # 6 u8 + 10 u16 + 3 u32; see record.py for the field meanings.
        rows.append((r[0], r[1], r[2], r[3], r[4], r[17], r[8], r[6], r[7]))
    return asic_id, aiclk_mhz, last_update_us, rows


def _read_registry(path: str) -> Dict[int, dict]:
    """Read v3 registry; aggregate entries by runtime_id.

    Returns {rid: {name, dispatch_count, peak_cycles_total}}.
    Each registrar fetch_add(write_cursor) call writes one slot, so dispatch_count
    is the count of slots seen for that rid (modulo 16k circular wrap).
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return {}
    if len(data) < REG_HEADER_SIZE + REG_ENTRY_SIZE:
        return {}
    if data[0:4] != b"TPRG":
        return {}
    version = struct.unpack_from("<H", data, 4)[0]
    if version != 3:
        return {}
    entry_size = struct.unpack_from("<H", data, 6)[0]
    if entry_size != 128:
        return {}
    cursor = struct.unpack_from("<I", data, 24)[0]
    n = min(cursor, REG_CAPACITY)
    out: Dict[int, dict] = {}
    for i in range(n):
        off = REG_HEADER_SIZE + i * REG_ENTRY_SIZE
        rid = struct.unpack_from("<I", data, off)[0]
        name_bytes = data[off + 16 : off + 16 + 96]
        name = name_bytes.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        cycles_total = struct.unpack_from("<Q", data, off + 120)[0]
        prev = out.get(rid)
        if prev is None:
            out[rid] = {
                "name": name,
                "dispatch_count": 1,
                "peak_cycles_total": cycles_total,
            }
        else:
            prev["dispatch_count"] += 1
            if not prev["name"] and name:
                prev["name"] = name
            if cycles_total > prev["peak_cycles_total"]:
                prev["peak_cycles_total"] = cycles_total
    return out


def resolve_name(prog_id: int, names: Dict[int, dict]) -> str:
    """Match the C++ viewer's lookup heuristic: try raw, then encoded forms."""
    if prog_id in names:
        return names[prog_id].get("name", "")
    for dev in range(8):
        encoded = (prog_id << 10) | dev
        if encoded in names:
            return names[encoded].get("name", "")
    return ""


# ─── Live state ─────────────────────────────────────────────────────────────


@dataclass
class HistoryEntry:
    name: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    dispatch_count: int = 0
    peak_cycles_total: int = 0


@dataclass
class ChipState:
    asic_id: int = 0
    aiclk_mhz: int = 0
    last_update_us: int = 0
    cores: List[Tuple] = field(default_factory=list)
    timeline: deque = field(default_factory=lambda: deque(maxlen=80))


# ─── Helpers ────────────────────────────────────────────────────────────────


def fcolor(pct: float) -> str:
    if pct == 0:
        return "grey50"
    if pct < 33:
        return "green"
    if pct < 66:
        return "yellow"
    return "red"


def progcolor(prog_id: int) -> str:
    """Stable per-program color (HSL hash)."""
    if prog_id == 0:
        return "grey50"
    palette = [
        "cyan",
        "magenta",
        "blue",
        "orange3",
        "purple",
        "spring_green3",
        "pink1",
        "violet",
    ]
    return palette[prog_id % len(palette)]


def fmt_time_secs(secs: float) -> str:
    if secs < 0:
        return "—"
    m = int(secs) // 60
    s = secs - m * 60
    return f"{m:02d}:{s:06.3f}"


# ─── Widgets ────────────────────────────────────────────────────────────────


class ChipGridWidget(Static):
    """Per-chip core grid heatmap. One Static per chip."""

    chip_idx: int = 0

    def update_grid(self, chip: ChipState, names: Dict[int, dict]) -> None:
        if not chip.cores:
            self.update("waiting for chip data…")
            return
        # Find grid extent (max NOC x/y).
        max_x = max(c[0] for c in chip.cores)
        max_y = max(c[1] for c in chip.cores)
        W = max_x + 1
        H = max_y + 1
        # Build per-(x,y) lookup.
        grid: Dict[Tuple[int, int], Tuple] = {}
        for c in chip.cores:
            grid[(c[0], c[1])] = c

        text = Text()
        # Title.
        text.append(
            f"chip {self.chip_idx}  asic 0x{chip.asic_id:x}  {chip.aiclk_mhz} MHz  " f"{len(chip.cores)} cores\n",
            style="bold",
        )
        # Header row (NOC x labels).
        text.append("    ")
        for x in range(W):
            text.append(f"{x:>4}", style="dim")
        text.append("\n")
        for y in range(H):
            text.append(f"{y:>2}  ", style="dim")
            for x in range(W):
                c = grid.get((x, y))
                if c is None:
                    text.append("    ", style="grey15")
                    continue
                _x, _y, _lx, _ly, _rem, kid, fp, sp, dp = c
                f_pct = fp / 10.0
                prog = (kid >> 10) & 0x1FFFFF if kid else 0
                # Cell: 4 chars wide. Color by F%, stripe by program.
                if prog == 0:
                    text.append(" ___", style="grey39")
                else:
                    pcol = progcolor(prog)
                    fc = fcolor(f_pct)
                    # Show prog id (last 3 digits) in program color, then F% bar in fcolor
                    text.append(f"{prog%1000:>3}", style=f"{pcol}")
                    text.append("█", style=fc)
            text.append("\n")
        self.update(text)


class ProgramsWidget(DataTable):
    """Active programs table, refreshed each frame."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.cursor_type = "row"
        self.zebra_stripes = True

    def on_mount(self) -> None:
        self.add_columns("ID", "Name", "Cores", "F%", "S%", "D%")

    def update_rows(
        self,
        chips: Dict[int, ChipState],
        names: Dict[int, dict],
    ) -> None:
        # Aggregate programs across all chips.
        agg: Dict[int, dict] = {}
        for ci, ch in chips.items():
            for c in ch.cores:
                _x, _y, _lx, _ly, _rem, kid, fp, sp, dp = c
                if kid == 0 or dp == 0:
                    continue
                prog = (kid >> 10) & 0x1FFFFF
                a = agg.setdefault(prog, {"cores": 0, "sum_f": 0, "sum_s": 0, "sum_d": 0, "chips": set()})
                a["cores"] += 1
                a["sum_f"] += fp
                a["sum_s"] += sp
                a["sum_d"] += dp
                a["chips"].add(ci)

        rows_sorted = sorted(agg.items(), key=lambda kv: (-kv[1]["cores"], kv[0]))

        # Snapshot cursor position to restore after update.
        cursor_row = self.cursor_row if self.row_count else 0
        self.clear()
        for prog, a in rows_sorted[:50]:
            name = resolve_name(prog, names) or ""
            f_avg = a["sum_f"] / a["cores"] / 10.0
            s_avg = a["sum_s"] / a["cores"] / 10.0
            d_avg = a["sum_d"] / a["cores"] / 10.0
            id_text = Text(f"#{prog}", style=progcolor(prog))
            self.add_row(
                id_text,
                name[:48] if name else Text("—", style="dim"),
                str(a["cores"]),
                Text(f"{f_avg:.1f}", style=fcolor(f_avg)),
                Text(f"{s_avg:.1f}", style=fcolor(s_avg)),
                Text(f"{d_avg:.1f}", style=fcolor(d_avg)),
            )
        if self.row_count and cursor_row < self.row_count:
            try:
                self.move_cursor(row=min(cursor_row, self.row_count - 1))
            except Exception:
                pass


class TimelineWidget(Static):
    """Per-chip Gantt timeline of the last N frames' program sets."""

    def update_timeline(self, chips: Dict[int, ChipState], names: Dict[int, dict]) -> None:
        if not chips:
            self.update("")
            return
        text = Text()
        for ci in sorted(chips):
            ch = chips[ci]
            if not ch.timeline:
                continue
            # Top programs by frame presence.
            counts: Dict[int, int] = collections.Counter()
            for fset in ch.timeline:
                for p in fset:
                    counts[p] += 1
            if not counts:
                continue
            top = counts.most_common(8)
            text.append(f"\nchip {ci} timeline ({len(ch.timeline) * 100} ms history)\n", style="bold")
            for prog, _cnt in top:
                name = resolve_name(prog, names) or "?"
                idstr = f"#{prog}"
                pcol = progcolor(prog)
                text.append(f"  {idstr:<6}", style=pcol)
                text.append(f" {name[:36]:<36}  [")
                # Pad on left so the rightmost cell is "now".
                pad = 80 - len(ch.timeline)
                if pad > 0:
                    text.append(" " * pad, style="dim")
                for fset in ch.timeline:
                    if prog in fset:
                        text.append("█", style=pcol)
                    else:
                        text.append("·", style="grey30")
                text.append("]\n")
        self.update(text)


class HistoryWidget(DataTable):
    """Every program seen since the viewer started."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.cursor_type = "row"
        self.zebra_stripes = True

    def on_mount(self) -> None:
        self.add_columns("ID", "Name", "First", "Last", "Disp", "Cycles")

    def update_rows(self, history: Dict[int, HistoryEntry], session_start: float) -> None:
        rows_sorted = sorted(history.items(), key=lambda kv: -kv[1].last_seen)
        cursor_row = self.cursor_row if self.row_count else 0
        self.clear()
        now = time.monotonic() - session_start
        for rid, h in rows_sorted[:200]:
            id_text = Text(f"#{rid}", style=progcolor(rid))
            recency = now - h.last_seen
            recency_style = "white" if recency < 0.5 else ("dim" if recency < 5 else "grey50")
            self.add_row(
                id_text,
                Text(h.name[:48], style=recency_style) if h.name else Text("—", style="dim"),
                fmt_time_secs(h.first_seen),
                fmt_time_secs(h.last_seen),
                str(h.dispatch_count),
                f"{h.peak_cycles_total:,}" if h.peak_cycles_total else Text("—", style="dim"),
            )
        if self.row_count and cursor_row < self.row_count:
            try:
                self.move_cursor(row=min(cursor_row, self.row_count - 1))
            except Exception:
                pass


# ─── App ────────────────────────────────────────────────────────────────────


class TtnvtopApp(App):
    CSS = """
    Screen { background: $background; }
    Header { background: $primary-darken-2; }
    .pane-title { color: $accent; padding: 0 1; }
    #chips-row { height: 45%; min-height: 16; border: solid $primary-darken-3; }
    #programs-pane { width: 40%; border: solid $primary-darken-3; }
    #chips-pane    { width: 60%; border: solid $primary-darken-3; padding: 1; overflow: auto; }
    #timeline      { height: 22%; min-height: 8; border: solid $primary-darken-3; padding: 0 1; overflow: auto; }
    #history-pane  { height: 1fr; border: solid $primary-darken-3; }
    DataTable      { height: 100%; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reset_history", "Reset history"),
        Binding("space", "toggle_pause", "Pause/resume"),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    refresh_hz: float = reactive(10.0)
    chips: Dict[int, ChipState] = {}
    history: Dict[int, HistoryEntry] = {}
    names: Dict[int, dict] = {}
    session_start: float = 0.0
    paused: bool = False

    def __init__(self, shm_glob: str, registry_path: str, hz: float):
        super().__init__()
        self.shm_glob = shm_glob
        self.registry_path = registry_path
        self.refresh_hz = hz
        self.session_start = time.monotonic()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="chips-row"):
            yield Static("waiting…", id="chips-pane")
            yield ProgramsWidget(id="programs-pane")
        yield TimelineWidget(id="timeline")
        yield HistoryWidget(id="history-pane")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(1.0 / self.refresh_hz, self.refresh_data)
        self.title = "ttnvtop_py"
        self.sub_title = "live"

    def action_reset_history(self) -> None:
        self.history.clear()
        self.session_start = time.monotonic()

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        self.sub_title = "PAUSED" if self.paused else "live"

    def refresh_data(self) -> None:
        if self.paused:
            return  # freeze the view; data path stops collecting too
        # Refresh registry.
        try:
            self.names = _read_registry(self.registry_path)
        except Exception:
            self.names = {}

        # Refresh per-chip SHM.
        paths = sorted(glob.glob(self.shm_glob))
        seen_chip_idxs: Set[int] = set()
        for ci, path in enumerate(paths):
            try:
                asic, aiclk, last_us, cores = _read_chip(path)
            except Exception:
                continue
            ch = self.chips.setdefault(ci, ChipState())
            ch.asic_id = asic
            ch.aiclk_mhz = aiclk
            ch.last_update_us = last_us
            ch.cores = cores
            # Per-frame program set for the timeline.
            fset: Set[int] = set()
            for c in cores:
                kid = c[5]
                dp = c[8]
                if kid and dp:
                    fset.add((kid >> 10) & 0x1FFFFF)
            ch.timeline.append(fset)
            seen_chip_idxs.add(ci)
            # Update history.
            now_s = time.monotonic() - self.session_start
            for prog in fset:
                h = self.history.setdefault(prog, HistoryEntry())
                if h.dispatch_count == 0:
                    h.first_seen = now_s
                    if prog in self.names:
                        h.name = self.names[prog].get("name", "")
                    else:
                        h.name = resolve_name(prog, self.names)
                h.last_seen = now_s
                h.dispatch_count += 1
                # Refresh peak cycles_total.
                if prog in self.names:
                    n = self.names[prog]
                    if n["peak_cycles_total"] > h.peak_cycles_total:
                        h.peak_cycles_total = n["peak_cycles_total"]
                else:
                    # Try encoded form for cycles.
                    for dev in range(8):
                        enc = (prog << 10) | dev
                        if enc in self.names:
                            ct = self.names[enc]["peak_cycles_total"]
                            if ct > h.peak_cycles_total:
                                h.peak_cycles_total = ct
                            break

        # Drop chips that vanished (workload finished + collector unmounted).
        for stale in list(self.chips.keys()):
            if stale not in seen_chip_idxs:
                del self.chips[stale]

        # Push to widgets.
        chips_pane = self.query_one("#chips-pane", Static)
        if not self.chips:
            chips_pane.update(
                Text(
                    "no /dev/shm/tt_device_*_util — start ttnvtop-collector",
                    style="yellow",
                )
            )
        else:
            t = Text()
            # Cell layout (8 chars wide per core × 2 rows per y):
            #   row 1:  ' #495   '   ← prog id (or '  ___  ' if idle)
            #   row 2:  '  87%   '   ← F% (colored by saturation)
            # Plus 2 dim header columns for x labels and y labels. Empty
            # NOC slots (e.g. dispatch/eth cores) render as 8 spaces so
            # the grid keeps its rectangular shape.
            CELL_W = 8
            for ci in sorted(self.chips):
                ch = self.chips[ci]
                if not ch.cores:
                    continue
                t.append(
                    f"\nchip {ci}  asic 0x{ch.asic_id:x}  {ch.aiclk_mhz} MHz  {len(ch.cores)} cores\n",
                    style="bold",
                )
                max_x = max(c[0] for c in ch.cores)
                max_y = max(c[1] for c in ch.cores)
                W = max_x + 1
                H = max_y + 1
                grid: Dict[Tuple[int, int], Tuple] = {}
                for c in ch.cores:
                    grid[(c[0], c[1])] = c
                # x header
                t.append("     ")
                for x in range(W):
                    t.append(f"x={x:<2d}".center(CELL_W), style="dim")
                t.append("\n")
                for y in range(H):
                    # Row 1: prog id labels
                    t.append(f"y={y:<2d}".rjust(5), style="dim")
                    for x in range(W):
                        c = grid.get((x, y))
                        if c is None:
                            t.append(" " * CELL_W)
                            continue
                        kid = c[5]
                        prog = (kid >> 10) & 0x1FFFFF if kid else 0
                        if prog == 0:
                            t.append("  ___   ", style="grey39")
                        else:
                            label = f"#{prog}"
                            t.append(label.center(CELL_W), style=progcolor(prog))
                    t.append("\n")
                    # Row 2: F% number
                    t.append("     ")
                    for x in range(W):
                        c = grid.get((x, y))
                        if c is None:
                            t.append(" " * CELL_W)
                            continue
                        kid = c[5]
                        fp = c[6]
                        prog = (kid >> 10) & 0x1FFFFF if kid else 0
                        f_pct = fp / 10.0
                        if prog == 0:
                            t.append("        ", style="grey39")
                        else:
                            label = f"{f_pct:.0f}%"
                            t.append(label.center(CELL_W), style=fcolor(f_pct))
                    t.append("\n")
                    # Spacer between rows
                    t.append("\n")
            chips_pane.update(t)

        self.query_one(ProgramsWidget).update_rows(self.chips, self.names)
        self.query_one(TimelineWidget).update_timeline(self.chips, self.names)
        self.query_one(HistoryWidget).update_rows(self.history, self.session_start)


def main():
    ap = argparse.ArgumentParser(description="ttnvtop_py — Python/Textual TUI viewer")
    ap.add_argument("--shm-glob", default="/dev/shm/tt_device_*_util")
    ap.add_argument("--registry", default="/dev/shm/tt_program_registry")
    ap.add_argument("--hz", type=float, default=10.0, help="UI refresh rate (Hz)")
    args = ap.parse_args()

    if not glob.glob(args.shm_glob):
        print(
            f"warning: no SHM files match {args.shm_glob} — start ttnvtop-collector first.\n"
            f"the UI will keep waiting and pick them up when they appear.",
        )

    app = TtnvtopApp(args.shm_glob, args.registry, args.hz)
    app.run()


if __name__ == "__main__":
    main()
