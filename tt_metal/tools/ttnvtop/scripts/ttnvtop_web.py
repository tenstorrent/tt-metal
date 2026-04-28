#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# ttnvtop_web: live browser dashboard. Reads the same /dev/shm/tt_device_*_util
# + /dev/shm/tt_program_registry that the C++ ttnvtop reads, serves them
# via Server-Sent Events to a single self-contained HTML page. Open the
# URL it prints; the page updates at 10 Hz.
#
# No external dependencies — uses only Python stdlib (http.server, json,
# struct). The HTML/CSS/JS is embedded as a string below; opening the
# served URL gives you a polished core grid with hover tooltips, sortable
# tables, and a Gantt-style timeline.
#
# Usage:
#   python tt_metal/tools/ttnvtop/scripts/ttnvtop_web.py             # 0.0.0.0:8080
#   python tt_metal/tools/ttnvtop/scripts/ttnvtop_web.py --port 9000
#
# Architecture:
#   ┌─ collector (C++, separate process) ─┐
#   │   reads UMD, writes SHM             │
#   └─────────────┬───────────────────────┘
#                 │ /dev/shm/tt_device_*_util
#                 │ /dev/shm/tt_program_registry
#                 ▼
#   ┌─ ttnvtop_web (this) ─────────────────────────┐
#   │   reads SHM at 10 Hz                         │
#   │   serves HTML on /                           │
#   │   serves JSON SSE stream on /events          │
#   └─────────────┬─────────────────────────────────┘
#                 │ HTTP + SSE
#                 ▼
#   ┌─ browser ───────────────────────────────────┐
#   │   single HTML page; renders grid live       │
#   └─────────────────────────────────────────────┘

from __future__ import annotations

import argparse
import glob
import http.server
import json
import os
import socketserver
import struct
import threading
import time
from typing import Dict


# ─── SHM schemas (mirror common/shm_schema.hpp + program_registry.hpp) ─────

HEADER_FMT = "<4sHHQIIQQIIII4I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
PER_CORE_FMT = "<6B10H2x3I"
PER_CORE_SIZE = struct.calcsize(PER_CORE_FMT)
REG_HEADER_SIZE = 48
REG_ENTRY_SIZE = 128
REG_CAPACITY = 16384


def _read_chip(path: str):
    with open(path, "rb") as f:
        data = f.read()
    hdr = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    asic_id = hdr[3]
    n_cores = hdr[8]
    aiclk_mhz = hdr[11]
    cores = []
    for i in range(n_cores):
        r = struct.unpack(
            PER_CORE_FMT,
            data[HEADER_SIZE + i * PER_CORE_SIZE : HEADER_SIZE + (i + 1) * PER_CORE_SIZE],
        )
        cores.append(
            {
                "x": r[0],
                "y": r[1],
                "lx": r[2],
                "ly": r[3],
                "remote": int(r[4]),
                "kid": r[17],
                "f": r[8] / 10.0,
                "s": r[6] / 10.0,
                "d": r[7] / 10.0,
            }
        )
    return {"asic": asic_id, "aiclk_mhz": aiclk_mhz, "cores": cores}


def _read_registry(path: str) -> Dict[int, dict]:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return {}
    if len(data) < REG_HEADER_SIZE + REG_ENTRY_SIZE or data[0:4] != b"TPRG":
        return {}
    if struct.unpack_from("<H", data, 4)[0] != 3:
        return {}
    if struct.unpack_from("<H", data, 6)[0] != 128:
        return {}
    cursor = struct.unpack_from("<I", data, 24)[0]
    out: Dict[int, dict] = {}
    for i in range(min(cursor, REG_CAPACITY)):
        off = REG_HEADER_SIZE + i * REG_ENTRY_SIZE
        rid = struct.unpack_from("<I", data, off)[0]
        name_b = data[off + 16 : off + 16 + 96]
        name = name_b.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        ct = struct.unpack_from("<Q", data, off + 120)[0]
        prev = out.get(rid)
        if prev is None:
            out[rid] = {"name": name, "dispatch_count": 1, "cycles_total": ct}
        else:
            prev["dispatch_count"] += 1
            if not prev["name"] and name:
                prev["name"] = name
            if ct > prev["cycles_total"]:
                prev["cycles_total"] = ct
    return out


def resolve_name(prog_id: int, names: Dict[int, dict]) -> str:
    if prog_id in names:
        return names[prog_id]["name"]
    for dev in range(8):
        enc = (prog_id << 10) | dev
        if enc in names:
            return names[enc]["name"]
    return ""


def gather_frame(shm_glob: str, registry_path: str, history: Dict[int, dict], session_start: float) -> dict:
    """Read everything once, return a JSON-serializable frame snapshot."""
    names = _read_registry(registry_path)
    paths = sorted(glob.glob(shm_glob))
    chips = []
    now = time.monotonic() - session_start
    for ci, path in enumerate(paths):
        try:
            chip = _read_chip(path)
        except Exception:
            continue
        # decorate cores with prog id + name for the frontend
        for c in chip["cores"]:
            kid = c["kid"]
            prog = (kid >> 10) & 0x1FFFFF if kid else 0
            c["prog"] = prog
            c["name"] = resolve_name(prog, names) if prog else ""
            del c["kid"]
        chips.append({"idx": ci, **chip})
        # Update history
        for c in chip["cores"]:
            prog = c["prog"]
            if not prog or c["d"] == 0:
                continue
            h = history.setdefault(
                prog,
                {
                    "name": c["name"],
                    "first_s": now,
                    "last_s": now,
                    "frames": 0,
                    "cycles_total": 0,
                },
            )
            h["last_s"] = now
            h["frames"] += 1
            if not h["name"] and c["name"]:
                h["name"] = c["name"]
            # Refresh cycles_total from registry
            if prog in names:
                ct = names[prog]["cycles_total"]
            else:
                ct = 0
                for dev in range(8):
                    enc = (prog << 10) | dev
                    if enc in names:
                        ct = names[enc]["cycles_total"]
                        break
            if ct > h["cycles_total"]:
                h["cycles_total"] = ct

    history_list = sorted(
        ({"prog": p, **h} for p, h in history.items()),
        key=lambda x: -x["last_s"],
    )
    return {
        "t": now,
        "chips": chips,
        "history": history_list[:200],
        "history_total": len(history),
    }


# ─── HTML page (single-string template) ────────────────────────────────────


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ttnvtop_web</title>
<style>
:root {
  --bg:#0d1117; --panel:#161b22; --border:#30363d;
  --fg:#c9d1d9; --fg-dim:#8b949e; --accent:#58a6ff;
  --idle:#21262d;
}
* { box-sizing: border-box; }
body { margin:0; font-family: ui-monospace, monospace; font-size:13px; background:var(--bg); color:var(--fg); }
header { padding:10px 16px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
header h1 { margin:0; font-size:15px; }
.status { color:var(--fg-dim); font-size:12px; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#3fb950; margin-right:5px; vertical-align:middle; }
.dot.disconnected { background:#f85149; }
main { padding:12px; display:grid; grid-template-columns: 1fr 380px; grid-template-rows: auto auto 1fr; gap:12px; height:calc(100vh - 50px); }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:6px; padding:10px; overflow:auto; }
.panel h2 { margin:0 0 8px 0; font-size:11px; color:var(--fg-dim); text-transform:uppercase; letter-spacing:0.5px; }
#chips { grid-column:1; grid-row:1/3; min-height:0; }
#programs { grid-column:2; grid-row:1; min-height:200px; }
#history { grid-column:2; grid-row:2/4; min-height:200px; }
#timeline { grid-column:1; grid-row:3; min-height:120px; }
.chip-grid { margin-bottom:16px; }
.chip-grid .title { font-weight:bold; margin-bottom:4px; }
.chip-grid .title .meta { color:var(--fg-dim); font-weight:normal; margin-left:8px; }
.grid-table { border-collapse:separate; border-spacing:2px; }
.grid-table .x-label, .grid-table .y-label { color:var(--fg-dim); font-size:10px; text-align:center; padding:0 4px; }
.cell {
  width:60px; height:42px; border-radius:3px;
  position:relative; cursor:pointer; user-select:none;
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  font-size:11px; line-height:1.1;
}
.cell.idle { background:var(--idle); color:var(--fg-dim); }
.cell.empty { background:#0d1117; cursor:default; }
.cell .pid { font-weight:bold; font-size:11px; }
.cell .pct { font-size:10px; opacity:0.85; }
.cell:hover { outline:2px solid var(--accent); outline-offset:-1px; z-index:5; }
.tooltip {
  position:fixed; pointer-events:none;
  background:#000a; color:#fff; padding:6px 10px; border-radius:4px;
  border:1px solid var(--border); font-size:11px; line-height:1.4;
  box-shadow:0 2px 8px rgba(0,0,0,0.5); z-index:1000;
  display:none; max-width:380px;
}
table.data { width:100%; border-collapse:collapse; font-size:12px; }
table.data th, table.data td { padding:3px 6px; border-bottom:1px solid var(--border); text-align:left; }
table.data th { color:var(--fg-dim); font-weight:normal; cursor:pointer; user-select:none; position:sticky; top:0; background:var(--panel); }
table.data th:hover { color:var(--accent); }
table.data tr:hover td { background:rgba(88,166,255,0.06); }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.tl-row { display:flex; align-items:center; margin-bottom:1px; }
.tl-label { width:200px; flex-shrink:0; font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding:0 4px; }
.tl-cells { display:flex; gap:1px; }
.tl-cell { width:6px; height:14px; background:var(--idle); }
.tl-cell.on { background:var(--prog-color); }
</style>
</head>
<body>
<header>
  <h1>ttnvtop_web</h1>
  <span class="status"><span id="conn-dot" class="dot"></span><span id="conn-status">connecting…</span></span>
  <span class="status">t=<span id="t-display">—</span></span>
  <span class="status">history: <span id="history-count">0</span></span>
  <button id="pause-btn" style="margin-left:auto; background:var(--panel); color:var(--fg); border:1px solid var(--border); padding:4px 10px; border-radius:4px; cursor:pointer;">⏸ pause</button>
</header>
<main>
  <section id="chips" class="panel">
    <h2>core grid</h2>
    <div id="chips-content">waiting for chip data…</div>
  </section>
  <section id="programs" class="panel">
    <h2>programs running now</h2>
    <div id="programs-content"><i style="color:var(--fg-dim)">—</i></div>
  </section>
  <section id="timeline" class="panel">
    <h2>timeline (last ~8 s)</h2>
    <div id="timeline-content"><i style="color:var(--fg-dim)">—</i></div>
  </section>
  <section id="history" class="panel">
    <h2>history (every program seen)</h2>
    <div id="history-content"><i style="color:var(--fg-dim)">—</i></div>
  </section>
</main>
<div id="tooltip" class="tooltip"></div>
<script>
"use strict";

const tooltip = document.getElementById("tooltip");
const state = {
  paused: false,
  // Per-chip ring buffer of program-id sets for the timeline.
  timeline: new Map(), // chipIdx -> [Set, Set, ...]
  TIMELINE_MAX: 80,
  sortHistory: { col: "last_s", desc: true },
};

document.getElementById("pause-btn").addEventListener("click", () => {
  state.paused = !state.paused;
  document.getElementById("pause-btn").textContent = state.paused ? "▶ resume" : "⏸ pause";
});

function pct2color(pct) {
  if (pct === 0) return "#21262d";
  if (pct < 33) return "#3fb950";
  if (pct < 66) return "#d29922";
  return "#f85149";
}
function progColor(pid) {
  if (!pid) return "#666";
  const h = ((pid * 2654435761) >>> 0) % 360;
  return `hsl(${h},70%,60%)`;
}
function fmtSecs(s) {
  if (s < 0) return "—";
  const m = Math.floor(s / 60);
  const ss = (s - m*60).toFixed(3);
  return String(m).padStart(2,"0") + ":" + String(ss).padStart(6,"0");
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[ch]));
}

// SSE
let es = null;
function connect() {
  es = new EventSource("/events");
  es.addEventListener("frame", (ev) => {
    if (state.paused) return;
    const f = JSON.parse(ev.data);
    render(f);
    document.getElementById("conn-dot").classList.remove("disconnected");
    document.getElementById("conn-status").textContent = "connected";
  });
  es.addEventListener("error", () => {
    document.getElementById("conn-dot").classList.add("disconnected");
    document.getElementById("conn-status").textContent = "disconnected — retrying";
    setTimeout(connect, 2000);
  });
}
connect();

function render(f) {
  document.getElementById("t-display").textContent = fmtSecs(f.t);
  document.getElementById("history-count").textContent = f.history_total;
  renderChips(f.chips);
  renderPrograms(f.chips);
  renderTimeline(f.chips);
  renderHistory(f.history);
}

function renderChips(chips) {
  const root = document.getElementById("chips-content");
  if (!chips || chips.length === 0) {
    root.innerHTML = "<i style='color:var(--fg-dim)'>no chips found — start ttnvtop-collector</i>";
    return;
  }
  let html = "";
  for (const ch of chips) {
    let maxX = 0, maxY = 0;
    for (const c of ch.cores) { if (c.x > maxX) maxX = c.x; if (c.y > maxY) maxY = c.y; }
    const W = maxX + 1, H = maxY + 1;
    const grid = {};
    for (const c of ch.cores) grid[c.x + "," + c.y] = c;
    html += `<div class="chip-grid">
      <div class="title">chip ${ch.idx} <span class="meta">asic 0x${ch.asic.toString(16)} · ${ch.aiclk_mhz} MHz · ${ch.cores.length} cores</span></div>
      <table class="grid-table"><thead><tr><th></th>`;
    for (let x = 0; x < W; x++) html += `<th class="x-label">x=${x}</th>`;
    html += `</tr></thead><tbody>`;
    for (let y = 0; y < H; y++) {
      html += `<tr><th class="y-label">y=${y}</th>`;
      for (let x = 0; x < W; x++) {
        const c = grid[x + "," + y];
        if (!c) {
          html += `<td><div class="cell empty"></div></td>`;
          continue;
        }
        if (c.prog === 0) {
          html += `<td><div class="cell idle" data-chip="${ch.idx}" data-x="${x}" data-y="${y}">
            <span class="pid">—</span><span class="pct">idle</span>
          </div></td>`;
          continue;
        }
        const fc = pct2color(c.f);
        const pc = progColor(c.prog);
        html += `<td><div class="cell" data-chip="${ch.idx}" data-x="${x}" data-y="${y}"
          style="background:${fc}40; border-left:3px solid ${pc};">
          <span class="pid" style="color:${pc}">#${c.prog}</span>
          <span class="pct" style="color:${fc}">${c.f.toFixed(0)}%</span>
        </div></td>`;
      }
      html += `</tr>`;
    }
    html += `</tbody></table></div>`;
  }
  root.innerHTML = html;
  // Attach hover for tooltips (event delegation).
  root.querySelectorAll(".cell:not(.empty)").forEach(el => {
    el.addEventListener("mouseenter", (e) => showTooltip(e, chips, el));
    el.addEventListener("mousemove", positionTooltip);
    el.addEventListener("mouseleave", hideTooltip);
  });
}

function showTooltip(ev, chips, el) {
  const ci = +el.dataset.chip, x = +el.dataset.x, y = +el.dataset.y;
  const ch = chips.find(c => c.idx === ci);
  const c = ch ? ch.cores.find(o => o.x === x && o.y === y) : null;
  if (!c) return;
  const name = c.name ? escapeHtml(c.name) : "<i style='color:#888'>(unnamed)</i>";
  tooltip.innerHTML = `
    <b>chip ${ci} · core (${x},${y})</b>${c.remote ? " · remote" : ""}<br>
    program: <span style="color:${progColor(c.prog)}">#${c.prog || "—"}</span> ${name}<br>
    F=${c.f.toFixed(1)}% · S=${c.s.toFixed(1)}% · D=${c.d.toFixed(1)}%
  `;
  tooltip.style.display = "block";
  positionTooltip(ev);
}
function positionTooltip(ev) {
  tooltip.style.left = (ev.clientX + 14) + "px";
  tooltip.style.top = (ev.clientY + 14) + "px";
}
function hideTooltip() { tooltip.style.display = "none"; }

function renderPrograms(chips) {
  const agg = new Map();
  for (const ch of chips) for (const c of ch.cores) {
    if (!c.prog || c.d === 0) continue;
    let a = agg.get(c.prog);
    if (!a) { a = { prog: c.prog, name: c.name, cores: 0, sumF: 0, sumS: 0, sumD: 0, chips: new Set() }; agg.set(c.prog, a); }
    a.cores++; a.sumF += c.f; a.sumS += c.s; a.sumD += c.d;
    a.chips.add(ch.idx);
    if (c.name && !a.name) a.name = c.name;
  }
  const rows = [...agg.values()].sort((a,b) => b.cores - a.cores);
  if (rows.length === 0) {
    document.getElementById("programs-content").innerHTML = "<i style='color:var(--fg-dim)'>no programs running</i>";
    return;
  }
  let html = `<table class="data"><thead><tr><th>id</th><th>name</th><th class="num">cores</th><th class="num">F%</th><th class="num">S%</th><th class="num">D%</th></tr></thead><tbody>`;
  for (const r of rows.slice(0, 30)) {
    html += `<tr>
      <td><span style="color:${progColor(r.prog)}">#${r.prog}</span></td>
      <td>${escapeHtml(r.name) || "<i style='color:#888'>—</i>"}</td>
      <td class="num">${r.cores}</td>
      <td class="num" style="color:${pct2color(r.sumF/r.cores)}">${(r.sumF/r.cores).toFixed(1)}</td>
      <td class="num" style="color:${pct2color(r.sumS/r.cores)}">${(r.sumS/r.cores).toFixed(1)}</td>
      <td class="num" style="color:${pct2color(r.sumD/r.cores)}">${(r.sumD/r.cores).toFixed(1)}</td>
    </tr>`;
  }
  if (rows.length > 30) html += `<tr><td colspan="6" style="color:var(--fg-dim)">(${rows.length-30} more)</td></tr>`;
  html += `</tbody></table>`;
  document.getElementById("programs-content").innerHTML = html;
}

function renderTimeline(chips) {
  for (const ch of chips) {
    const set = new Set();
    for (const c of ch.cores) if (c.prog && c.d > 0) set.add(c.prog);
    if (!state.timeline.has(ch.idx)) state.timeline.set(ch.idx, []);
    const buf = state.timeline.get(ch.idx);
    buf.push(set);
    while (buf.length > state.TIMELINE_MAX) buf.shift();
  }
  // Render: rows = top programs by presence frequency
  let html = "";
  for (const [ci, buf] of state.timeline) {
    if (!buf.length) continue;
    const counts = new Map();
    for (const s of buf) for (const p of s) counts.set(p, (counts.get(p)||0) + 1);
    const top = [...counts.entries()].sort((a,b) => b[1] - a[1]).slice(0, 10);
    if (!top.length) continue;
    html += `<div style="font-weight:bold; margin:8px 0 4px 0;">chip ${ci}</div>`;
    for (const [pid, _] of top) {
      const pc = progColor(pid);
      html += `<div class="tl-row" style="--prog-color:${pc}">
        <span class="tl-label" style="color:${pc}">#${pid}</span>
        <span class="tl-cells">`;
      const pad = state.TIMELINE_MAX - buf.length;
      for (let i = 0; i < pad; i++) html += `<span class="tl-cell"></span>`;
      for (const s of buf) html += `<span class="tl-cell${s.has(pid)?" on":""}"></span>`;
      html += `</span></div>`;
    }
  }
  document.getElementById("timeline-content").innerHTML = html || "<i style='color:var(--fg-dim)'>—</i>";
}

function renderHistory(history) {
  const cur = state.sortHistory;
  const rows = [...history];
  rows.sort((a,b) => {
    const av = a[cur.col], bv = b[cur.col];
    const cmp = (typeof av === "string") ? av.localeCompare(bv) : av - bv;
    return cur.desc ? -cmp : cmp;
  });
  let html = `<table class="data"><thead><tr>
    <th data-sort="prog">id</th>
    <th data-sort="name">name</th>
    <th class="num" data-sort="first_s">first</th>
    <th class="num" data-sort="last_s">last</th>
    <th class="num" data-sort="frames">frames</th>
    <th class="num" data-sort="cycles_total">cycles</th>
  </tr></thead><tbody>`;
  for (const h of rows.slice(0, 200)) {
    html += `<tr>
      <td><span style="color:${progColor(h.prog)}">#${h.prog}</span></td>
      <td>${escapeHtml(h.name) || "<i style='color:#888'>—</i>"}</td>
      <td class="num">${fmtSecs(h.first_s)}</td>
      <td class="num">${fmtSecs(h.last_s)}</td>
      <td class="num">${h.frames}</td>
      <td class="num">${h.cycles_total ? h.cycles_total.toLocaleString() : "<i style='color:#888'>—</i>"}</td>
    </tr>`;
  }
  html += `</tbody></table>`;
  document.getElementById("history-content").innerHTML = html;
  document.querySelectorAll("#history-content th[data-sort]").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.sort;
      if (state.sortHistory.col === col) state.sortHistory.desc = !state.sortHistory.desc;
      else { state.sortHistory.col = col; state.sortHistory.desc = true; }
    });
  });
}
</script>
</body>
</html>
"""


# ─── HTTP / SSE server ─────────────────────────────────────────────────────


class FrameProducer:
    """Reads SHM at refresh_hz and pushes frames to all connected SSE clients."""

    def __init__(self, shm_glob, registry_path, refresh_hz):
        self.shm_glob = shm_glob
        self.registry_path = registry_path
        self.refresh_period = 1.0 / refresh_hz
        self.clients: list = []  # list of (queue: list, lock, evt)
        self.history: Dict[int, dict] = {}
        self.session_start = time.monotonic()
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self.stop_evt.set()

    def add_client(self):
        q = []
        evt = threading.Event()
        with self.lock:
            self.clients.append((q, evt))
        return q, evt

    def remove_client(self, q, evt):
        with self.lock:
            self.clients = [(qq, ee) for qq, ee in self.clients if qq is not q]

    def _loop(self):
        while not self.stop_evt.is_set():
            try:
                frame = gather_frame(self.shm_glob, self.registry_path, self.history, self.session_start)
            except Exception as e:
                frame = {"t": 0, "chips": [], "history": [], "history_total": 0, "error": str(e)}
            payload = json.dumps(frame, separators=(",", ":"))
            with self.lock:
                for q, evt in self.clients:
                    q.append(payload)
                    if len(q) > 8:  # cap backlog per slow client
                        del q[0:-4]
                    evt.set()
            self.stop_evt.wait(self.refresh_period)


_PRODUCER: FrameProducer = None  # set in main


class TtnvtopHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Quiet by default — uncomment for debugging.
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            q, evt = _PRODUCER.add_client()
            try:
                while True:
                    evt.wait(timeout=15)
                    evt.clear()
                    while q:
                        payload = q.pop(0)
                        msg = f"event: frame\ndata: {payload}\n\n"
                        try:
                            self.wfile.write(msg.encode("utf-8"))
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                    # Heartbeat to keep connection alive.
                    try:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
            finally:
                _PRODUCER.remove_client(q, evt)
            return
        self.send_response(404)
        self.end_headers()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global _PRODUCER
    ap = argparse.ArgumentParser(description="ttnvtop_web — live HTML dashboard")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--shm-glob", default="/dev/shm/tt_device_*_util")
    ap.add_argument("--registry", default="/dev/shm/tt_program_registry")
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()

    if not glob.glob(args.shm_glob):
        print(f"warning: no SHM files match {args.shm_glob} — start ttnvtop-collector first.")

    _PRODUCER = FrameProducer(args.shm_glob, args.registry, args.hz)
    _PRODUCER.start()

    server = ThreadingHTTPServer((args.host, args.port), TtnvtopHandler)
    bind_host = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"ttnvtop_web: serving http://{bind_host}:{args.port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down…")
    finally:
        _PRODUCER.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
