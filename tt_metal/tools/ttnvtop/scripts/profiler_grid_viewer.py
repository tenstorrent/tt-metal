#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# profiler_grid_viewer: render tt-metal device-profiler output as an
# interactive chip-cluster timeline. Takes profile_log_device.csv (and
# optionally tt_program_registry.bin), emits a self-contained HTML file
# with a 2D core grid that animates as you scrub a timeline.
#
# This visualizes the SAME data Tracy and ttnn-visualizer consume, but
# laid out by chip topology instead of by thread — so you can see "at
# t=12.345s, the whole 8x8 grid looked like this on chip 0; this one
# core diverged from the rest". Hover any cell for full details.
#
# Usage:
#   python profiler_grid_viewer.py \
#       --profiler runs/<ts>/profile_log_device.csv \
#       --registry runs/<ts>/tt_program_registry.bin \
#       --out runs/<ts>/grid.html
#
# Open the resulting HTML in any browser. No server, no deps beyond
# the Python stdlib + a modern browser.

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


KERNEL_ZONES = {
    "BRISC-KERNEL",
    "NCRISC-KERNEL",
    "TRISC-KERNEL",
    "ERISC-KERNEL",
}


def parse_registry(path: str) -> Dict[int, str]:
    """v3 registry → {runtime_id: name}."""
    out: Dict[int, str] = {}
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return out
    if len(data) < 48 or data[0:4] != b"TPRG":
        return out
    if struct.unpack_from("<H", data, 4)[0] != 3:
        return out
    if struct.unpack_from("<H", data, 6)[0] != 128:
        return out
    cursor = struct.unpack_from("<I", data, 24)[0]
    for i in range(min(cursor, 16384)):
        off = 48 + i * 128
        rid = struct.unpack_from("<I", data, off)[0]
        name_b = data[off + 16 : off + 16 + 96]
        name = name_b.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        if rid not in out and name:
            out[rid] = name
    return out


def parse_profiler(path: str, max_events: int = 200_000) -> Tuple[List[dict], Dict, int]:
    """Pair ZONE_START/END rows into zones.

    Returns (zones, meta, kept_count).
      zones = [{chip, x, y, risc, kid, t0, t1}, ...]  (cycles, raw)
      meta  = {min_cycles, max_cycles, chips: set, riscs: set}
    """
    zones: List[dict] = []
    open_zones: Dict[Tuple, int] = {}  # (chip, x, y, risc, zone, kid) -> t_start
    chips: set = set()
    riscs: set = set()
    min_cyc = None
    max_cyc = 0
    rows_total = 0
    rows_kept = 0

    with open(path, "r") as f:
        first = f.readline()
        if not first.startswith("ARCH:"):
            f.seek(0)
        reader = csv.reader(f)
        header = next(reader)
        idx = {col.strip(): i for i, col in enumerate(header)}
        REQ = [
            "PCIe slot",
            "core_x",
            "core_y",
            "RISC processor type",
            "time[cycles since reset]",
            "run host ID",
            "zone name",
            "type",
        ]
        for r in REQ:
            if r not in idx:
                print(f"ERROR: profiler CSV missing column: {r!r}", file=sys.stderr)
                sys.exit(2)

        for row in reader:
            rows_total += 1
            if not row or len(row) <= idx["type"]:
                continue
            zone = row[idx["zone name"]].strip()
            if zone not in KERNEL_ZONES:
                continue
            try:
                raw_id = int(row[idx["run host ID"]] or 0)
                cyc = int(row[idx["time[cycles since reset]"]])
            except ValueError:
                continue
            if raw_id == 0:
                continue
            kid = (raw_id >> 10) & 0x1FFFFF
            if kid == 0:
                continue
            chip = int(row[idx["PCIe slot"]])
            x = int(row[idx["core_x"]])
            y = int(row[idx["core_y"]])
            risc = row[idx["RISC processor type"]].strip()
            ztype = row[idx["type"]].strip()
            chips.add(chip)
            riscs.add(risc)
            if min_cyc is None or cyc < min_cyc:
                min_cyc = cyc
            if cyc > max_cyc:
                max_cyc = cyc
            key = (chip, x, y, risc, zone, kid)
            if ztype == "ZONE_START":
                open_zones[key] = cyc
            elif ztype == "ZONE_END":
                t0 = open_zones.pop(key, None)
                if t0 is None or cyc < t0:
                    continue
                zones.append(
                    {
                        "chip": chip,
                        "x": x,
                        "y": y,
                        "risc": risc,
                        "kid": kid,
                        "t0": t0,
                        "t1": cyc,
                    }
                )
                rows_kept += 1
                if len(zones) >= max_events:
                    break

    if min_cyc is None:
        min_cyc = 0
    print(
        f"  profiler: {rows_total:,} rows scanned, {rows_kept:,} zones kept "
        f"(chips={sorted(chips)}, riscs={sorted(riscs)})",
        file=sys.stderr,
    )

    # Normalize cycles to start at 0 for display.
    for z in zones:
        z["t0"] -= min_cyc
        z["t1"] -= min_cyc

    meta = {
        "min_cycles": 0,
        "max_cycles": max_cyc - min_cyc,
        "chips": sorted(chips),
        "riscs": sorted(riscs),
        "rows_kept": rows_kept,
    }
    return zones, meta, rows_kept


HTML_TEMPLATE = """<!doctype html>
<!--
  profiler_grid_viewer — generated __TIMESTAMP__
  source: __SOURCE_PATH__
  zones: __ZONE_COUNT__  cycles: __MAX_CYCLES__  aiclk: __AICLK_MHZ__ MHz
-->
<html lang="en">
<head>
<meta charset="utf-8">
<title>profiler grid viewer</title>
<style>
:root {
  --bg:#0d1117; --panel:#161b22; --border:#30363d;
  --fg:#c9d1d9; --fg-dim:#8b949e; --accent:#58a6ff; --idle:#21262d;
}
* { box-sizing: border-box; }
body { margin:0; font-family: ui-monospace, monospace; font-size:13px; background:var(--bg); color:var(--fg); }
header { padding:10px 16px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
header h1 { margin:0; font-size:15px; }
.meta { color:var(--fg-dim); font-size:12px; }
button, select { background:var(--panel); color:var(--fg); border:1px solid var(--border); padding:4px 10px; border-radius:4px; cursor:pointer; font:inherit; }
button:hover, select:hover { border-color:var(--accent); }
button.playing { color:#3fb950; border-color:#3fb950; }
#scrubber-row { display:flex; align-items:center; gap:8px; padding:8px 16px; border-bottom:1px solid var(--border); }
#scrubber { flex:1; }
#t-display { width:130px; font-variant-numeric:tabular-nums; text-align:right; }
main { padding:12px; display:grid; grid-template-columns: 2fr 1fr; gap:12px; height:calc(100vh - 100px); }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:6px; padding:10px; overflow:auto; }
.panel h2 { margin:0 0 8px 0; font-size:11px; color:var(--fg-dim); text-transform:uppercase; letter-spacing:0.5px; }
#grids { grid-column:1; }
#detail { grid-column:2; }
.chip-grid { margin-bottom:14px; }
.chip-grid .title { font-weight:bold; margin-bottom:4px; }
.chip-grid .title .meta { margin-left:8px; }
table.grid-table { border-collapse:separate; border-spacing:2px; }
table.grid-table th { color:var(--fg-dim); font-size:10px; padding:0 4px; text-align:center; font-weight:normal; }
.cell { width:54px; height:36px; border-radius:3px; cursor:pointer; user-select:none;
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  font-size:10px; line-height:1.1; position:relative; }
.cell.idle { background:var(--idle); color:var(--fg-dim); }
.cell.empty { background:#0d1117; cursor:default; }
.cell.selected { outline:2px solid var(--accent); outline-offset:-1px; z-index:5; }
.cell .pid { font-weight:bold; font-size:10px; }
.cell .pct { font-size:9px; opacity:0.85; }
.cell:hover { outline:2px solid var(--accent); outline-offset:-1px; z-index:5; }
.tooltip { position:fixed; pointer-events:none; background:#000a; color:#fff;
  padding:6px 10px; border-radius:4px; border:1px solid var(--border); font-size:11px;
  line-height:1.4; box-shadow:0 2px 8px rgba(0,0,0,0.5); z-index:1000; display:none; max-width:420px; }
.lane-row { display:flex; align-items:center; margin-bottom:1px; flex-wrap:nowrap; height:18px; }
.lane-label { width:96px; flex-shrink:0; font-size:10px; color:var(--fg-dim); padding-right:6px; text-align:right; }
.lane-bar { flex:1; height:14px; position:relative; background:var(--idle); border-radius:2px; }
.lane-zone { position:absolute; height:14px; top:0; }
.lane-zone:hover { outline:1px solid #fff; }
.now-marker { position:absolute; top:-4px; bottom:-4px; width:1px; background:var(--accent); pointer-events:none; }
table.data { width:100%; border-collapse:collapse; font-size:11px; }
table.data th, table.data td { padding:3px 6px; border-bottom:1px solid var(--border); text-align:left; }
table.data th { color:var(--fg-dim); font-weight:normal; position:sticky; top:0; background:var(--panel); }
.num { text-align:right; font-variant-numeric:tabular-nums; }
</style>
</head>
<body>
<header>
  <h1>profiler grid viewer</h1>
  <span class="meta">zones: __ZONE_COUNT__</span>
  <span class="meta">aiclk: __AICLK_MHZ__ MHz</span>
  <span class="meta">total: __DURATION_MS__ ms</span>
  <span class="meta">source: __SOURCE_PATH_HTML__</span>
  <select id="risc-select" style="margin-left:auto" title="Which RISC's running zone determines cell color"></select>
</header>
<div id="scrubber-row">
  <button id="play-btn">▶</button>
  <button id="step-back">−</button>
  <button id="step-fwd">+</button>
  <input id="scrubber" type="range" min="0" max="100000" value="0">
  <span id="t-display">0.000 ms</span>
</div>
<main>
  <section id="grids" class="panel">
    <h2>chip cluster (cell color = program at scrubber time)</h2>
    <div id="grids-content"></div>
  </section>
  <section id="detail" class="panel">
    <h2>per-core lanes</h2>
    <div id="detail-content"></div>
  </section>
</main>
<div id="tooltip" class="tooltip"></div>
<script>
"use strict";

const ZONES = __ZONES_JSON__;
const META = __META_JSON__;
const NAMES = __NAMES_JSON__;
const AICLK_MHZ = __AICLK_MHZ__;

const tooltip = document.getElementById("tooltip");
const scrubber = document.getElementById("scrubber");
const tDisplay = document.getElementById("t-display");
const playBtn = document.getElementById("play-btn");
const riscSelect = document.getElementById("risc-select");

const state = {
  curCycle: 0,
  playing: false,
  playTimer: null,
  selectedRisc: META.riscs.includes("TRISC_1") ? "TRISC_1" : (META.riscs[0] || ""),
  selectedCell: null,  // {chip, x, y}
};

// Index zones for fast lookup at any cycle: per (chip,x,y,risc) → sorted zones
const idx = new Map();
const cores = new Map();  // chip -> set of "x,y"
for (const z of ZONES) {
  const k = `${z.chip}|${z.x}|${z.y}|${z.risc}`;
  if (!idx.has(k)) idx.set(k, []);
  idx.get(k).push(z);
  if (!cores.has(z.chip)) cores.set(z.chip, new Set());
  cores.get(z.chip).add(`${z.x},${z.y}`);
}
for (const arr of idx.values()) arr.sort((a, b) => a.t0 - b.t0);

scrubber.min = 0;
scrubber.max = META.max_cycles;
scrubber.step = Math.max(1, Math.floor(META.max_cycles / 100000));
scrubber.value = 0;

for (const r of META.riscs) {
  const o = document.createElement("option");
  o.value = r; o.textContent = r;
  if (r === state.selectedRisc) o.selected = true;
  riscSelect.appendChild(o);
}
riscSelect.addEventListener("change", () => {
  state.selectedRisc = riscSelect.value;
  render();
});

function progColor(pid) {
  if (!pid) return "#666";
  const h = ((pid * 2654435761) >>> 0) % 360;
  return `hsl(${h},70%,60%)`;
}
function fmtCycles(c) {
  const us = c / AICLK_MHZ;
  if (us < 1000) return us.toFixed(1) + " µs";
  if (us < 1e6) return (us / 1000).toFixed(3) + " ms";
  return (us / 1e6).toFixed(3) + " s";
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[ch]));
}

// Binary search: find zone active at cycle c on (chip,x,y,risc).
function zoneAt(chip, x, y, risc, c) {
  const k = `${chip}|${x}|${y}|${risc}`;
  const arr = idx.get(k);
  if (!arr) return null;
  // linear scan back from upper bound (good enough for typical workloads)
  let lo = 0, hi = arr.length - 1;
  let best = null;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid].t0 <= c) { lo = mid + 1; best = arr[mid]; }
    else hi = mid - 1;
  }
  if (best && best.t0 <= c && best.t1 >= c) return best;
  return null;
}

scrubber.addEventListener("input", () => {
  state.curCycle = parseInt(scrubber.value, 10);
  updateTime();
  render();
});

playBtn.addEventListener("click", togglePlay);
document.getElementById("step-back").addEventListener("click", () => step(-1));
document.getElementById("step-fwd").addEventListener("click", () => step(+1));
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "ArrowRight") { step(+1); e.preventDefault(); }
  else if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
  else if (e.key === "Home") { state.curCycle = 0; scrubber.value = 0; updateTime(); render(); }
  else if (e.key === "End") { state.curCycle = META.max_cycles; scrubber.value = META.max_cycles; updateTime(); render(); }
});

function togglePlay() {
  state.playing = !state.playing;
  playBtn.textContent = state.playing ? "❚❚" : "▶";
  playBtn.classList.toggle("playing", state.playing);
  if (state.playing) {
    state.playTimer = setInterval(() => {
      const inc = Math.max(1, Math.floor(META.max_cycles / 1000));  // 1000 frames cover the run at ~30 Hz
      state.curCycle += inc;
      if (state.curCycle >= META.max_cycles) { state.curCycle = META.max_cycles; togglePlay(); return; }
      scrubber.value = state.curCycle;
      updateTime(); render();
    }, 33);
  } else {
    clearInterval(state.playTimer);
    state.playTimer = null;
  }
}
function step(d) {
  const inc = Math.max(1, Math.floor(META.max_cycles / 10000)) * d;
  state.curCycle = Math.max(0, Math.min(META.max_cycles, state.curCycle + inc));
  scrubber.value = state.curCycle;
  updateTime(); render();
}

function updateTime() {
  const ms = state.curCycle / AICLK_MHZ / 1000;
  tDisplay.textContent = ms.toFixed(3) + " ms";
}

function render() {
  renderGrids();
  renderDetail();
}

function renderGrids() {
  const root = document.getElementById("grids-content");
  let html = "";
  for (const chip of [...cores.keys()].sort((a,b) => a-b)) {
    const set = cores.get(chip);
    let maxX = 0, maxY = 0;
    for (const k of set) { const [x,y] = k.split(",").map(Number); if (x>maxX) maxX=x; if (y>maxY) maxY=y; }
    html += `<div class="chip-grid">
      <div class="title">chip ${chip} <span class="meta">${set.size} cores · risc=${state.selectedRisc}</span></div>
      <table class="grid-table"><thead><tr><th></th>`;
    for (let x = 0; x <= maxX; x++) html += `<th>x=${x}</th>`;
    html += `</tr></thead><tbody>`;
    for (let y = 0; y <= maxY; y++) {
      html += `<tr><th>y=${y}</th>`;
      for (let x = 0; x <= maxX; x++) {
        if (!set.has(`${x},${y}`)) {
          html += `<td><div class="cell empty"></div></td>`;
          continue;
        }
        const z = zoneAt(chip, x, y, state.selectedRisc, state.curCycle);
        const sel = state.selectedCell && state.selectedCell.chip === chip
          && state.selectedCell.x === x && state.selectedCell.y === y;
        if (!z) {
          html += `<td><div class="cell idle ${sel?'selected':''}" data-chip="${chip}" data-x="${x}" data-y="${y}">
            <span class="pid">—</span></div></td>`;
          continue;
        }
        const pc = progColor(z.kid);
        html += `<td><div class="cell ${sel?'selected':''}" data-chip="${chip}" data-x="${x}" data-y="${y}"
          style="background:${pc}40; border-left:3px solid ${pc};">
          <span class="pid" style="color:${pc}">#${z.kid}</span>
          <span class="pct">${fmtCycles(z.t1 - z.t0)}</span>
        </div></td>`;
      }
      html += `</tr>`;
    }
    html += `</tbody></table></div>`;
  }
  root.innerHTML = html;
  root.querySelectorAll(".cell:not(.empty)").forEach(el => {
    el.addEventListener("click", () => {
      state.selectedCell = { chip: +el.dataset.chip, x: +el.dataset.x, y: +el.dataset.y };
      render();
    });
    el.addEventListener("mouseenter", (e) => showTip(e, el));
    el.addEventListener("mousemove", posTip);
    el.addEventListener("mouseleave", () => tooltip.style.display = "none");
  });
}

function showTip(ev, el) {
  const chip = +el.dataset.chip, x = +el.dataset.x, y = +el.dataset.y;
  const z = zoneAt(chip, x, y, state.selectedRisc, state.curCycle);
  if (!z) {
    tooltip.innerHTML = `<b>chip ${chip} · core (${x},${y}) · ${state.selectedRisc}</b><br><i>idle at this moment</i>`;
  } else {
    const name = NAMES[z.kid] ? escapeHtml(NAMES[z.kid]) : "<i style='color:#888'>(unnamed)</i>";
    tooltip.innerHTML = `
      <b>chip ${chip} · core (${x},${y}) · ${state.selectedRisc}</b><br>
      program: <span style="color:${progColor(z.kid)}">#${z.kid}</span> ${name}<br>
      zone: ${escapeHtml(z.risc)}-KERNEL<br>
      duration: ${fmtCycles(z.t1 - z.t0)} (cycles ${z.t0.toLocaleString()}—${z.t1.toLocaleString()})
    `;
  }
  tooltip.style.display = "block";
  posTip(ev);
}
function posTip(ev) {
  tooltip.style.left = (ev.clientX + 14) + "px";
  tooltip.style.top = (ev.clientY + 14) + "px";
}

function renderDetail() {
  const root = document.getElementById("detail-content");
  if (!state.selectedCell) {
    root.innerHTML = "<i style='color:var(--fg-dim)'>click a cell to see its full per-RISC timeline.</i>";
    return;
  }
  const sc = state.selectedCell;
  let html = `<div style="font-weight:bold; margin-bottom:8px;">chip ${sc.chip} · core (${sc.x},${sc.y})</div>`;
  // For each RISC, draw the lane.
  const W = 360;  // px width of bar — we'll scale to that.
  const totalC = META.max_cycles || 1;
  const nowFrac = state.curCycle / totalC;
  for (const risc of META.riscs) {
    const k = `${sc.chip}|${sc.x}|${sc.y}|${risc}`;
    const arr = idx.get(k) || [];
    let zoneHtml = "";
    for (const z of arr) {
      const left = (z.t0 / totalC) * 100;
      const w = Math.max(0.2, ((z.t1 - z.t0) / totalC) * 100);
      const pc = progColor(z.kid);
      zoneHtml += `<div class="lane-zone" style="left:${left}%; width:${w}%; background:${pc};" title="#${z.kid} ${fmtCycles(z.t1-z.t0)}"></div>`;
    }
    html += `<div class="lane-row">
      <span class="lane-label">${risc}</span>
      <div class="lane-bar">${zoneHtml}<div class="now-marker" style="left:${nowFrac*100}%"></div></div>
    </div>`;
  }
  // Recent zone list
  html += `<div style="margin-top:14px; font-weight:bold;">recent zones (most recent first)</div>`;
  html += `<table class="data"><thead><tr><th>risc</th><th>id</th><th>name</th><th class="num">start</th><th class="num">dur</th></tr></thead><tbody>`;
  const all = [];
  for (const risc of META.riscs) {
    const k = `${sc.chip}|${sc.x}|${sc.y}|${risc}`;
    const arr = idx.get(k) || [];
    for (const z of arr) all.push({...z, risc});
  }
  all.sort((a,b) => b.t0 - a.t0);
  for (const z of all.slice(0, 80)) {
    const name = NAMES[z.kid] ? escapeHtml(NAMES[z.kid]).slice(0, 50) : "—";
    html += `<tr>
      <td>${z.risc}</td>
      <td><span style="color:${progColor(z.kid)}">#${z.kid}</span></td>
      <td>${name}</td>
      <td class="num">${fmtCycles(z.t0)}</td>
      <td class="num">${fmtCycles(z.t1 - z.t0)}</td>
    </tr>`;
  }
  if (all.length > 80) html += `<tr><td colspan="5" style="color:var(--fg-dim)">(${all.length-80} more)</td></tr>`;
  html += `</tbody></table>`;
  root.innerHTML = html;
}

updateTime();
render();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(
        description="Render tt-metal device-profiler output as an interactive chip-cluster timeline."
    )
    ap.add_argument("--profiler", required=True, help="path to profile_log_device.csv")
    ap.add_argument("--registry", default=None, help="path to tt_program_registry.bin (optional, for names)")
    ap.add_argument("--out", required=True, help="output HTML path")
    ap.add_argument(
        "--max-events",
        type=int,
        default=200_000,
        help="cap on parsed zones (truncates if profiler is huge; default 200000)",
    )
    ap.add_argument("--aiclk-mhz", type=int, default=1000, help="AICLK MHz for cycle→time conversion")
    args = ap.parse_args()

    print(f"reading profiler: {args.profiler}", file=sys.stderr)
    zones, meta, n = parse_profiler(args.profiler, max_events=args.max_events)
    if not zones:
        print("ERROR: zero zones parsed — was TT_METAL_DEVICE_PROFILER=1 set on the workload?", file=sys.stderr)
        sys.exit(2)

    names: Dict[int, str] = {}
    if args.registry:
        print(f"reading registry: {args.registry}", file=sys.stderr)
        names = parse_registry(args.registry)
        print(f"  registry: {len(names):,} programs", file=sys.stderr)

    duration_ms = meta["max_cycles"] / args.aiclk_mhz / 1000.0
    # Use explicit placeholder substitution rather than %-formatting; the
    # HTML/CSS/JS template is full of literal '%' characters (percentage
    # widths, CSS percentages, escape sequences) and escaping every one
    # of them is brittle.
    repl = {
        "__TIMESTAMP__": time.strftime("%Y-%m-%d %H:%M:%S"),
        "__SOURCE_PATH__": args.profiler,
        "__SOURCE_PATH_HTML__": os.path.basename(args.profiler),
        "__ZONE_COUNT__": str(n),
        "__MAX_CYCLES__": str(meta["max_cycles"]),
        "__DURATION_MS__": f"{duration_ms:.3f}",
        "__AICLK_MHZ__": str(args.aiclk_mhz),
        "__ZONES_JSON__": json.dumps(zones, separators=(",", ":")),
        "__META_JSON__": json.dumps(meta, separators=(",", ":")),
        "__NAMES_JSON__": json.dumps({str(k): v for k, v in names.items()}, separators=(",", ":")),
    }
    page = HTML_TEMPLATE
    for k, v in repl.items():
        page = page.replace(k, v)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nwrote {out_path}  ({size_mb:.2f} MB)", file=sys.stderr)
    print(f"open with: xdg-open {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
