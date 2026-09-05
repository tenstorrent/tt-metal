# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The dashboard page (HTML/CSS/JS) served by optimize_dashboard.

Kept apart from the collector so the data code stays readable. Everything rendered here comes from
/api/state — the JS never assumes a model's stage set; labels like TTFT/TPOT are metric names the
collector computes from stage VALUES (see _serving_metrics), each card subtitles the stage it was
derived from.
"""

from __future__ import annotations

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Optimize — live</title>
<style>
  :root {
    --bg: #0a1020; --panel: #0f1930; --panel2: #0c1426; --line: #1e2c47;
    --txt: #e6edf7; --dim: #8494ad; --blue: #3b82f6; --green: #34d399;
    --red: #f87171; --amber: #fbbf24; --chip: #182642;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--txt);
         font: 14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  header { display: flex; align-items: baseline; gap: 14px; padding: 18px 26px 14px; flex-wrap: wrap; }
  header h1 { font-size: 19px; margin: 0; font-weight: 650; }
  header .sub { color: var(--dim); font-size: 13px; }
  header .right { margin-left: auto; display: flex; align-items: center; gap: 14px; }
  header .updated { color: var(--dim); font-size: 12.5px; }
  .badge { font-size: 12px; font-weight: 600; padding: 4px 14px; border-radius: 20px; }
  .badge.live { background: rgba(52,211,153,.14); color: var(--green); border: 1px solid rgba(52,211,153,.45); }
  .badge.live::before { content: "●"; margin-right: 6px; animation: pulse 1.6s infinite; }
  .badge.idle { background: rgba(132,148,173,.12); color: var(--dim); border: 1px solid var(--line); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }
  .wrap { padding: 4px 26px 26px; max-width: 1520px; margin: 0 auto; }

  #cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 14px; margin: 10px 0 18px; }
  .card { background: linear-gradient(165deg, #12233f, #0d1930); border: 1px solid #1d3252;
          border-radius: 12px; padding: 16px 18px; }
  .card .k { font-size: 12px; color: #7f9ec9; letter-spacing: .4px; }
  .card .v { font-size: 26px; font-weight: 700; margin-top: 6px; font-variant-numeric: tabular-nums; }
  .card .v small { font-size: 14px; font-weight: 500; color: #9db4d4; }
  .card .d { font-size: 12px; margin-top: 3px; color: #7f9ec9; }
  .up { color: var(--green); } .dn { color: var(--red); }

  .grid { display: grid; grid-template-columns: 5fr 7fr; gap: 16px; margin-bottom: 16px; }
  @media (max-width: 1050px) { .grid { grid-template-columns: 1fr; } }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; }
  .panel > h2 { font-size: 14px; margin: 0; padding: 14px 18px 10px; font-weight: 650; }
  .panel .body { padding: 4px 18px 16px; }
  .caption { color: var(--dim); font-size: 12px; margin: 2px 0 10px; }

  .mrow { margin: 13px 0; }
  .mrow .lab { display: flex; justify-content: space-between; font-size: 13.5px; margin-bottom: 5px; }
  .mrow .lab .r { color: var(--dim); font-variant-numeric: tabular-nums; }
  .bar { height: 6px; background: var(--chip); border-radius: 3px; position: relative; }
  .bar > i { position: absolute; left: 0; top: 0; bottom: 0; background: var(--blue); border-radius: 3px; }
  .bar > i.win { background: var(--green); }
  .bar > b { position: absolute; top: -2px; bottom: -2px; width: 2px; background: var(--amber); }

  .opp { border: 1px solid var(--line); border-left: 3px solid var(--blue); border-radius: 8px;
         padding: 12px 14px; margin: 10px 0; background: var(--panel2);
         display: flex; gap: 12px; align-items: flex-start; }
  .opp.amber { border-left-color: var(--amber); }
  .opp.green { border-left-color: var(--green); }
  .opp.red { border-left-color: var(--red); }
  .opp .main { flex: 1; min-width: 0; }
  .opp .name { font-weight: 650; font-size: 13.5px; }
  .opp .why { color: var(--dim); font-size: 12.5px; margin-top: 3px; }
  .opp .side { text-align: right; white-space: nowrap; }
  .opp .est { color: var(--green); font-size: 12.5px; font-weight: 600; }
  .opp.hot { border-color: var(--amber); border-left-color: var(--amber);
             box-shadow: 0 0 0 1px rgba(251,191,36,.25); }
  .chip { display: inline-block; font-size: 11px; font-weight: 650; padding: 3px 10px; border-radius: 12px;
          background: var(--chip); color: var(--dim); }
  .chip.kept { background: rgba(52,211,153,.14); color: var(--green); }
  .chip.reverted, .chip.wedged { background: rgba(248,113,113,.14); color: var(--red); }
  .chip.no-gain { background: rgba(251,191,36,.13); color: var(--amber); }
  .chip.open { background: rgba(59,130,246,.14); color: var(--blue); }
  .chip.lever { background: var(--chip); color: #8fb4e8; font-weight: 600; }
  .chip.applying { background: rgba(251,191,36,.13); color: var(--amber); }
  .btn { border: none; border-radius: 7px; padding: 6px 14px; font-size: 12.5px; font-weight: 650;
         cursor: pointer; margin-left: 6px; }
  .btn.apply { background: var(--blue); color: #fff; }
  .btn.revert { background: transparent; color: var(--red); border: 1px solid rgba(248,113,113,.5); }
  .btn:disabled { opacity: .45; cursor: default; }

  .feed { max-height: 190px; overflow-y: auto; margin-top: 12px; border-top: 1px solid var(--line);
          padding-top: 8px; font-size: 12px; }
  .feed .ev { display: flex; gap: 8px; padding: 2px 0; color: var(--dim); }
  .feed .ev .t { color: #54678a; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .feed .ev .s { color: #8fb4e8; white-space: nowrap; }

  table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line);
           font-variant-numeric: tabular-nums; vertical-align: top; }
  th { color: var(--dim); font-weight: 600; font-size: 11.5px; }
  .mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }
  td .why { color: var(--dim); font-size: 11.5px; margin-top: 2px; }

  #tabs { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; }
  #tabbar { display: flex; gap: 6px; padding: 10px 14px 0; border-bottom: 1px solid var(--line); flex-wrap: wrap; }
  #tabbar button { background: transparent; border: none; color: var(--dim); font-size: 13px;
                   padding: 8px 14px; cursor: pointer; border-radius: 8px 8px 0 0; }
  #tabbar button.on { color: var(--txt); background: var(--chip); font-weight: 650; }
  #tabbody { padding: 16px 18px; }
  .empty { color: var(--dim); font-size: 13px; padding: 20px; text-align: center; }
  .stack { display: flex; height: 26px; border-radius: 6px; overflow: hidden; margin: 8px 0 14px; }
  .stack > div { height: 100%; }
  .legend { font-size: 12px; color: var(--dim); display: flex; gap: 14px; flex-wrap: wrap; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }
  .segbar { display: flex; gap: 6px; flex-wrap: wrap; margin: 0 0 12px; }
  .segbar button { background: var(--chip); border: 1px solid var(--line); color: var(--dim);
                   font-size: 12.5px; padding: 5px 13px; cursor: pointer; border-radius: 999px; }
  .segbar button.on { color: var(--txt); border-color: var(--blue); font-weight: 650; }
</style>
</head>
<body>
<header>
  <h1>Optimize — <span id="model">…</span></h1>
  <span class="sub" id="runinfo"></span>
  <div class="right">
    <span class="updated" id="updated"></span>
    <span id="livebadge" class="badge idle">CONNECTING</span>
  </div>
</header>
<div class="wrap">
  <div id="cards"></div>
  <div class="grid">
    <div class="panel"><h2>Performance Metrics</h2><div class="body" id="perf"></div></div>
    <div class="panel"><h2>Recommendations</h2><div class="body" id="opps"></div></div>
  </div>
  <div class="panel" style="margin-bottom:16px"><h2>Optimization History</h2><div class="body" id="histbody"></div></div>
  <div id="tabs">
    <div id="tabbar"></div>
    <div id="tabbody"></div>
  </div>
</div>
<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmtMs = (v) => (v === null || v === undefined) ? "—" : (v >= 100 ? v.toFixed(1) : v.toFixed(2)) + " ms";
const fmtMs2 = (v) => (v === null || v === undefined) ? "—" : v.toFixed(2) + " ms";
const fmtPct = (v) => (v === null || v === undefined) ? "" : v.toFixed(1) + "%";
const PALETTE = ["#3b82f6","#34d399","#fbbf24","#f778ba","#76e3ea","#e3b341","#ff9bce","#56d4dd"];

function deltaTxt(cur, base, dir) {
  if (cur == null || base == null || !base) return "";
  const d = (cur - base) / base * 100;
  const better = dir === "max" ? d > 0 : d < 0;
  return `<span class="${better ? "up" : "dn"}">${d > 0 ? "+" : ""}${d.toFixed(1)}%</span> vs baseline`;
}

/* Headline cards: TTFT / TPOT / ITL / E2EL / Throughput when the run declared a per-token unit;
   otherwise the run's own metric + stages. Card subtitles name the stage the number came from. */
function cardSpec(S) {
  const sv = S.serving || {};
  const cards = [];
  if (sv.first_token || sv.per_token) {
    const ft = sv.first_token, pt = sv.per_token, e2 = sv.e2e_latency, th = sv.throughput || {};
    if (ft) cards.push({k: "TTFT", v: ft.ms, unit: "ms", sub: ft.stage, d: deltaTxt(ft.ms, ft.baseline_ms, "min")});
    if (pt) {
      cards.push({k: "TPOT", v: pt.ms, unit: "ms", sub: pt.stage, d: deltaTxt(pt.ms, pt.baseline_ms, "min")});
      cards.push({k: "ITL", v: pt.ms, unit: "ms", sub: pt.stage, d: deltaTxt(pt.ms, pt.baseline_ms, "min")});
    }
    if (e2) cards.push({k: "E2EL", v: e2.ms, unit: "ms", sub: "all stages", d: deltaTxt(e2.ms, e2.baseline_ms, "min")});
    if (th.per_s != null || th.baseline != null)
      cards.push({k: "Throughput", v: th.per_s, unit: " " + (th.unit || "tok/s"), sub: "", d: deltaTxt(th.per_s, th.baseline, "max")});
    return cards;
  }
  const m = S.metric || {};
  cards.push({k: (m.name || "metric"), v: m.current, unit: " " + (m.unit || "ms"), sub: "current",
              d: deltaTxt(m.current, m.baseline, m.direction || "min")});
  cards.push({k: "baseline", v: m.baseline, unit: " " + (m.unit || "ms"), sub: "",
              d: m.target != null ? "target " + fmtMs(m.target) : ""});
  (S.stages || []).slice(0, 3).forEach(s =>
    cards.push({k: s.name, v: s.ms, unit: " ms", sub: s.path || "", d: deltaTxt(s.ms, s.baseline_ms, "min")}));
  return cards;
}

function renderCards(S) {
  $("cards").innerHTML = cardSpec(S).map(c => {
    const v = c.v == null ? "—" : (c.unit === "ms" ? Number(c.v).toFixed(1) : Number(c.v).toFixed(2));
    return `<div class="card"><div class="k">${esc(c.k)}</div>
      <div class="v">${v}<small>${esc(c.unit || "")}</small></div>
      <div class="d">${c.d || ""}${c.sub ? (c.d ? " · " : "") + esc(c.sub) : ""}</div></div>`;
  }).join("");
}

function barRow(label, cur, base, scale, extra) {
  const w = cur != null && scale ? Math.min(100, cur / scale * 100) : 0;
  const win = cur != null && base != null && cur < base;
  const mark = base != null && scale ? Math.min(100, base / scale * 100) : null;
  return `<div class="mrow"><div class="lab"><span>${esc(label)}</span>
    <span class="r">${fmtMs2(cur)}${extra ? " · " + esc(extra) : ""}</span></div>
    <div class="bar"><i class="${win ? "win" : ""}" style="width:${w}%"></i>${mark != null ? `<b style="left:${mark}%"></b>` : ""}</div></div>`;
}

function renderPerf(S) {
  const sv = S.serving || {};
  let rows = [];
  if (sv.first_token || sv.per_token) {
    const vals = [sv.first_token, sv.per_token, sv.e2e_latency].filter(Boolean).map(x => Math.max(x.ms || 0, x.baseline_ms || 0));
    const th = sv.throughput || {};
    const scale = Math.max(...vals, 1e-9);
    if (sv.first_token) rows.push(barRow("TTFT", sv.first_token.ms, sv.first_token.baseline_ms, scale));
    if (sv.per_token) {
      rows.push(barRow("TPOT", sv.per_token.ms, sv.per_token.baseline_ms, scale));
      rows.push(barRow("ITL", sv.per_token.ms, sv.per_token.baseline_ms, scale));
    }
    if (sv.e2e_latency) rows.push(barRow("E2E Latency", sv.e2e_latency.ms, sv.e2e_latency.baseline_ms, scale));
    if (th.per_s != null) {
      const tscale = Math.max(th.per_s, th.baseline || 0);
      rows.push(`<div class="mrow"><div class="lab"><span>Throughput</span>
        <span class="r">${th.per_s.toFixed(2)} ${esc(th.unit || "tok/s")}</span></div>
        <div class="bar"><i class="${th.baseline != null && th.per_s > th.baseline ? "win" : ""}" style="width:${Math.min(100, th.per_s / tscale * 100)}%"></i></div></div>`);
    }
  } else {
    const m = S.metric || {};
    const scale = Math.max(m.baseline || 0, m.current || 0, m.target || 0,
                           ...(S.stages || []).flatMap(s => [s.ms || 0, s.baseline_ms || 0]), 1e-9);
    if (m.current != null) rows.push(barRow("overall · " + (m.name || ""), m.current, m.baseline, scale));
    (S.stages || []).forEach(s => rows.push(barRow(s.name, s.ms, s.baseline_ms, scale, s.path)));
  }
  $("perf").innerHTML = rows.join("") ||
    `<div class="empty">no measurements yet — the baseline is still being measured</div>`;
}

function oppCard(o) {
  const tags = o.tags || {};
  const facts = [fmtMs2(o.device_ms) + " across " + (o.count ?? "?") + " calls — " + fmtPct(o.pct) + " of device time"];
  const tagline = ["bound", "memory", "fidelity", "grid"].map(k => tags[k]).filter(Boolean).join(" · ");
  const tried = (o.tried_rungs || []).map(r => `<span class="chip lever">${esc(r)}</span>`).join(" ");
  const accent = o.status === "cleared" ? "green" : (o.status === "touched" ? "" : "amber");
  const side = o.status === "cleared"
    ? `<span class="chip kept">✓ applied</span>`
    : `<span class="chip ${o.status === "touched" ? "no-gain" : "open"}">${o.status === "touched" ? "in progress" : "open"}</span>`;
  return `<div class="opp ${accent}"><div class="main">
    <div class="name">${esc(o.id)}</div>
    <div class="why">${esc(facts[0])}</div>
    ${tagline ? `<div class="why">${esc(tagline)}</div>` : ""}
    <div class="why">${tried ? "tried: " + tried : "no lever tried yet"}</div></div>
    <div class="side">${side}</div></div>`;
}

function hitlCard(p) {
  const t = p.tried || {}, r = p.result || {}, n = p.next || {};
  return `<div class="opp hot"><div class="main">
    <div class="name">Decision pending — ${esc(t.lever || "?")} on ${esc(t.op || "?")}</div>
    <div class="why">${esc(r.win ? "WIN" : "no win")} (${fmtMs2(r.before_ms)} → ${fmtMs2(r.after_ms)})${t.why ? " — " + esc(t.why) : ""}</div>
    ${n.target ? `<div class="why">next: ${esc(n.target)}${n.why ? " — " + esc(n.why) : ""}</div>` : ""}</div>
    <div class="side"><button class="btn apply" onclick="decide('commit', this)">Commit</button>
    <button class="btn revert" onclick="decide('revert', this)">Revert</button></div></div>`;
}

async function decide(action, btn) {
  btn.disabled = true;
  try {
    const r = await fetch("/api/hitl-decision", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({action})});
    const j = await r.json();
    if (!j.ok) btn.disabled = false;
  } catch (e) { btn.disabled = false; }
}

function renderOpps(S) {
  let html = `<div class="caption">Model Analysis</div>`;
  if (S.headroom) {
    html += `<div class="opp"><div class="main"><div class="name">Roofline headroom</div>
      <div class="why">modeled floor ${fmtMs2(S.headroom.floor_ms)} vs current ${fmtMs2(S.headroom.current_ms)}</div></div>
      <div class="side"><span class="est">Est. improvement: -${S.headroom.pct.toFixed(0)}%</span></div></div>`;
  }
  if (S.hitl_proposal) html += hitlCard(S.hitl_proposal);
  const at = S.attempts || [];
  const live = S.run.live && at.length ? at[at.length - 1] : null;
  if (live) {
    html += `<div class="opp amber"><div class="main"><div class="name">${esc(live.op)}</div>
      <div class="why">latest lever: <b>${esc(live.lever)}</b>${live.note ? " — " + esc(live.note) : ""}</div></div>
      <div class="side"><span class="chip applying">applying…</span></div></div>`;
  }
  html += (S.opportunities || []).slice(0, 4).map(oppCard).join("");
  if (!(S.opportunities || []).length && !S.hitl_proposal)
    html += `<div class="empty">no profiled opportunities yet — they appear once the baseline profile lands</div>`;
  const ev = (S.events || []).slice(0, 30).map(e =>
    `<div class="ev"><span class="t">${esc((e.ts || "").slice(11, 19))}</span>
     <span class="s">${esc(e.stage || "")} ${esc(e.event || "")}</span><span>${esc(e.detail || "")}</span></div>`).join("");
  html += `<div class="feed">${ev || '<div class="empty">no events yet</div>'}</div>`;
  $("opps").innerHTML = html;
}

/* ---- optimization-history chart: every logged experiment in the order the run recorded it.
   y is the full-pipeline verdict that attempt produced (the number the commit gate banks on), so a
   point's height is the result the run actually measured, and the step line is the engine's OWN
   best-so-far (fullpipe_best_ms, recorded per attempt) rather than a "best" this file invents.

   The axis is a RATE so that progress reads UPWARD, which is only legitimate when the run itself
   declared its pipeline time is per token (the full-pipeline baseline's "unit" field — the same thing
   that gates S.throughput): a verdict of ms-per-token IS 1000/ms tokens per second. A run that never
   declared that unit keeps its raw ms and therefore reads downward, because turning latency into a
   rate for it would be a fabricated number. ---- */
/* Monotone cubic (Fritsch–Carlson) path through [x,y] points. A plain Catmull-Rom/Bezier smoothing
   overshoots at direction changes, which on a best-so-far line would render a dip the run never
   measured; clamping the tangents keeps every segment inside its own two endpoints, so the curve
   only ever states what was recorded. */
function smoothPath(p) {
  const n = p.length;
  if (n < 2) return "";
  if (n === 2) return `M${p[0][0].toFixed(1)},${p[0][1].toFixed(1)} L${p[1][0].toFixed(1)},${p[1][1].toFixed(1)}`;
  const dx = [], slope = [];
  for (let i = 0; i < n - 1; i++) {
    dx.push(p[i + 1][0] - p[i][0]);
    slope.push(dx[i] ? (p[i + 1][1] - p[i][1]) / dx[i] : 0);
  }
  const t = new Array(n);
  t[0] = slope[0];
  t[n - 1] = slope[n - 2];
  for (let i = 1; i < n - 1; i++) {
    t[i] = (slope[i - 1] * slope[i] <= 0) ? 0 : (slope[i - 1] + slope[i]) / 2;
  }
  for (let i = 0; i < n - 1; i++) {
    if (slope[i] === 0) { t[i] = 0; t[i + 1] = 0; continue; }
    const a = t[i] / slope[i], b = t[i + 1] / slope[i], s = a * a + b * b;
    if (s > 9) {
      const tau = 3 / Math.sqrt(s);
      t[i] = tau * a * slope[i];
      t[i + 1] = tau * b * slope[i];
    }
  }
  let d = `M${p[0][0].toFixed(1)},${p[0][1].toFixed(1)}`;
  for (let i = 0; i < n - 1; i++) {
    const h = dx[i] / 3;
    d += ` C${(p[i][0] + h).toFixed(1)},${(p[i][1] + t[i] * h).toFixed(1)}` +
         ` ${(p[i + 1][0] - h).toFixed(1)},${(p[i + 1][1] - t[i + 1] * h).toFixed(1)}` +
         ` ${p[i + 1][0].toFixed(1)},${p[i + 1][1].toFixed(1)}`;
  }
  return d;
}

const HIST_COLOR = {kept: "#34d399", reverted: "#f87171", "no-gain": "#fbbf24", wedged: "#f59e0b"};
const HIST_LEGEND = {kept: "kept", reverted: "discarded", "no-gain": "no gain", wedged: "crash / no measurement"};
const histVal = (a) => (a.after_ms != null ? a.after_ms : a.measured_ms);

/* The stacks the run's OWN attempt records carry timings for (record_kernel_attempt's stages_json).
   Dominant first, then slowest first — the same "where does the time go" ordering the rest of the page
   uses. Nothing here assumes a particular stage exists. */
function historyStages(S) {
  const seen = new Map();
  (S.attempts || []).forEach(a => (a.stages || []).forEach(st => {
    if (st && st.name && !seen.has(st.name)) {
      seen.set(st.name, {name: st.name, ms: st.ms || 0, dom: !!st.dominant});
    }
  }));
  return [...seen.values()].sort((x, y) => (y.dom - x.dom) || (y.ms - x.ms)).map(o => o.name);
}

/* How to read ONE stack out of the attempt log.

   The per-token stack is special: its number is the one the commit gate BANKS, so its series is the
   recorded verdict and its best line is the engine's own fullpipe_best_ms. It is also the only stack
   that converts to a rate (ms per token -> tokens per second), so it is the only one that reads upward.

   Every other stack is a one-shot latency with no gate behind it, so there is no banked best to show:
   its best line is the running minimum of what was actually measured, called "best recorded" to keep
   the two apart, and its axis stays in ms where lower is better. */
function historySeries(S, stage) {
  const tok = ((S.serving || {}).per_token || {}).stage;
  if (!stage || stage === tok) {
    return {
      stage: tok || null,
      value: histVal,
      best: a => a.before_ms,
      bestLabel: "best so far",
      rate: (S.throughput || {}).unit || null,
    };
  }
  return {
    stage: stage,
    value: a => {
      const st = (a.stages || []).find(s => s && s.name === stage);
      return st && st.ms != null ? st.ms : null;
    },
    best: null,
    bestLabel: "best recorded",
    rate: null,
  };
}

function historyChart(S, stage) {
  const at = S.attempts || [];
  const W = 1000, H = 340, PL = 62, PR = 16, PT = 16, PB = 50;

  const ser = historySeries(S, stage);
  const rate = ser.rate;
  // A non-positive reading has no rate (1000/0 is not a throughput), so it is plotted as "no value"
  // rather than as an infinity that would collapse the whole axis.
  const yOf = ms => (ms == null ? null : (rate ? (ms > 0 ? 1000.0 / ms : null) : ms));
  const yLabel = rate
    ? rate + (ser.stage ? " · " + ser.stage : "") + " (higher is better)"
    : (ser.stage ? ser.stage + " ms (lower is better)" : "full-pipeline ms (lower is better)");
  const fmtY = v => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));

  const measured = at.filter(a => yOf(ser.value(a)) != null);
  if (measured.length < 2) return "";

  // The goal line is drawn ONLY when the run's optimized metric is the SAME QUANTITY as the plotted
  // verdicts — metric.current agreeing with the newest verdict is that proof. A run optimizing
  // whole-model device time while these verdicts are per-token pipeline time must NOT share an axis,
  // and silently drawing that target would put the goal off by the ratio between the two.
  const m = S.metric || {};
  const last = ser.value(measured[measured.length - 1]);
  const sameQuantity = m.current != null && last && Math.abs(m.current - last) / last < 0.25;

  const goal = (sameQuantity && m.target != null) ? yOf(m.target) : null;
  // The best line: the engine's banked best where one exists, otherwise the running minimum of the
  // readings themselves (a plain description of the data, not a second opinion on what counts as best).
  let runMin = null;
  const bestAt = at.map(a => {
    if (ser.best) return yOf(ser.best(a));
    const v = ser.value(a);
    if (v != null && (runMin == null || v < runMin)) runMin = v;
    return yOf(runMin);
  });
  const bests = bestAt.filter(v => v != null);
  const all = measured.map(a => yOf(ser.value(a))).concat(bests).concat(goal != null ? [goal] : []);
  let yMin = Math.min(...all), yMax = Math.max(...all);
  const pad = (yMax - yMin || Math.abs(yMax) || 1) * 0.12;
  yMin -= pad; yMax += pad;
  const n = at.length;
  const X = i => PL + (n <= 1 ? 0 : i / (n - 1)) * (W - PL - PR);
  const Y = v => H - PB - (v - yMin) / (yMax - yMin) * (H - PT - PB);

  let g = "";
  for (let k = 0; k <= 5; k++) {
    const v = yMin + (yMax - yMin) * k / 5;
    g += `<line x1="${PL}" y1="${Y(v)}" x2="${W - PR}" y2="${Y(v)}" stroke="#1e2c47" stroke-width="1"/>
      <text x="${PL - 8}" y="${Y(v) + 4}" fill="#8494ad" font-size="11" text-anchor="end">${fmtY(v)}</text>`;
  }
  if (goal != null) {
    g += `<line x1="${PL}" y1="${Y(goal)}" x2="${W - PR}" y2="${Y(goal)}" stroke="#f87171" stroke-width="1.5" stroke-dasharray="6 4"/>
      <text x="${PL + 6}" y="${Y(goal) - 6}" fill="#f87171" font-size="11">goal ${fmtY(goal)} ${esc(rate || m.unit || "")}</text>`;
  }

  // The best line, one point per attempt, smoothed. Smoothing here is MONOTONE (see smoothPath): a
  // best only ever improves, and an ordinary spline would bulge past its own endpoints and draw the
  // best briefly getting worse — a curve claiming something the run never measured.
  const line = [];
  bestAt.forEach((b, i) => {
    if (b != null) line.push([X(i), Y(b)]);
  });
  if (line.length > 1) {
    g += `<path d="${smoothPath(line)}" fill="none" stroke="#3b82f6" stroke-width="2" stroke-opacity="0.9"
      stroke-linecap="round"/>`;
  }

  at.forEach((a, i) => {
    const ms = ser.value(a), v = yOf(ms);
    const c = HIST_COLOR[a.status] || "#8494ad";
    const reading = v == null ? "no measurement"
      : (rate ? `${fmtY(v)} ${rate} (${ms.toFixed(2)} ms)` : `${ms.toFixed(2)} ms`);
    const tip = esc(`run ${i + 1} · ${a.lever} · ${a.op} — ${reading}` +
      ` (${HIST_LEGEND[a.status] || a.status})` +
      (a.commit ? ` · commit ${String(a.commit).slice(0, 7)}` : "") +
      (a.note ? "\n" + a.note : ""));
    if (v == null) {
      // No verdict recorded (the measurement crashed): marked ON the axis, never at a made-up height.
      const y = H - PB;
      g += `<path d="M${X(i) - 4},${y - 4} L${X(i) + 4},${y + 4} M${X(i) + 4},${y - 4} L${X(i) - 4},${y + 4}"
        stroke="${c}" stroke-width="1.8"><title>${tip}</title></path>`;
    } else {
      g += `<circle cx="${X(i)}" cy="${Y(v)}" r="4.2" fill="${c}" fill-opacity="0.9" stroke="#0a1020" stroke-width="0.8">
        <title>${tip}</title></circle>`;
    }
  });

  const counts = {};
  at.forEach(a => { counts[a.status] = (counts[a.status] || 0) + 1; });
  let legend = Object.keys(counts).map(k =>
    `<span><i style="background:${HIST_COLOR[k] || "#8494ad"}"></i>${esc(HIST_LEGEND[k] || k)} (${counts[k]})</span>`).join("");
  if (line.length > 1) legend += `<span><i style="background:#3b82f6"></i>${esc(ser.bestLabel)}</span>`;
  if (goal != null) legend += `<span><i style="background:#f87171"></i>goal (${esc(m.name || "target")})</span>`;

  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">${g}
    <text x="${PL}" y="${H - PB + 18}" fill="#8494ad" font-size="11">run 1</text>
    <text x="${W - PR}" y="${H - PB + 18}" fill="#8494ad" font-size="11" text-anchor="end">run ${n}</text>
    <text x="14" y="${H / 2}" fill="#8494ad" font-size="12" text-anchor="middle" transform="rotate(-90 14 ${H / 2})">${esc(yLabel)}</text>
    </svg><div class="legend" style="margin:6px 0 14px">${legend}</div>`;
}

/* One curve per stack, each expressed as speedup against its OWN first recorded reading.

   This is the only way the stacks share an axis honestly. Their measured units genuinely differ — the
   per-token stack is a rate, the one-shot stacks are latencies, and the run records no work count to
   turn those into rates (prompt_tokens sits next to a batch size with no per-request split, and the
   encode stack has no count at all). A ratio against a stack's own baseline needs no such denominator:
   it is unit-free, so every stack reads upward on the same scale and stays directly comparable. */
function historyStackSeries(S) {
  const at = S.attempts || [];
  return historyStages(S).map(name => {
    const pts = [];
    at.forEach((a, i) => {
      const st = (a.stages || []).find(s => s && s.name === name);
      if (st && st.ms != null && st.ms > 0) pts.push({i: i, ms: st.ms, a: a});
    });
    return {name: name, pts: pts};
  }).filter(s => s.pts.length > 1);
}

function speedupChart(S) {
  const series = historyStackSeries(S);
  if (!series.length) return "";
  const n = (S.attempts || []).length;
  const W = 1000, H = 340, PL = 62, PR = 16, PT = 16, PB = 50;
  const X = i => PL + (n <= 1 ? 0 : i / (n - 1)) * (W - PL - PR);

  const base = s => s.pts[0].ms;
  const vals = series.flatMap(s => s.pts.map(p => base(s) / p.ms)).concat([1.0]);
  let yMin = Math.min(...vals), yMax = Math.max(...vals);
  const pad = (yMax - yMin || 1) * 0.12;
  yMin -= pad; yMax += pad;
  const Y = v => H - PB - (v - yMin) / (yMax - yMin) * (H - PT - PB);

  let g = "";
  for (let k = 0; k <= 5; k++) {
    const v = yMin + (yMax - yMin) * k / 5;
    g += `<line x1="${PL}" y1="${Y(v)}" x2="${W - PR}" y2="${Y(v)}" stroke="#1e2c47" stroke-width="1"/>
      <text x="${PL - 8}" y="${Y(v) + 4}" fill="#8494ad" font-size="11" text-anchor="end">${v.toFixed(2)}×</text>`;
  }
  if (yMin < 1.0 && yMax > 1.0) {
    g += `<line x1="${PL}" y1="${Y(1)}" x2="${W - PR}" y2="${Y(1)}" stroke="#8494ad" stroke-width="1.2" stroke-dasharray="6 4"/>
      <text x="${PL + 6}" y="${Y(1) - 6}" fill="#8494ad" font-size="11">1.00× — where each stack started</text>`;
  }

  series.forEach((s, si) => {
    const c = PALETTE[si % PALETTE.length];
    const b = base(s);
    g += `<path d="${smoothPath(s.pts.map(p => [X(p.i), Y(b / p.ms)]))}" fill="none" stroke="${c}"
      stroke-width="2" stroke-opacity="0.9" stroke-linecap="round"/>`;
    s.pts.forEach(p => {
      const sp = b / p.ms;
      const tip = esc(`run ${p.i + 1} · ${s.name} ${sp.toFixed(2)}× — ${p.ms.toFixed(2)} ms` +
        ` (from ${b.toFixed(2)} ms)\n${p.a.lever} · ${p.a.op} — ${HIST_LEGEND[p.a.status] || p.a.status}` +
        (p.a.note ? "\n" + p.a.note : ""));
      g += `<circle cx="${X(p.i)}" cy="${Y(sp)}" r="3.4" fill="${c}" fill-opacity="0.95"
        stroke="#0a1020" stroke-width="0.8"><title>${tip}</title></circle>`;
    });
  });

  const legend = series.map((s, si) => {
    const b = base(s), lastMs = s.pts[s.pts.length - 1].ms;
    return `<span><i style="background:${PALETTE[si % PALETTE.length]}"></i>${esc(s.name)} —
      ${(b / lastMs).toFixed(2)}× (${b.toFixed(0)} → ${lastMs.toFixed(0)} ms)</span>`;
  }).join("");

  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">${g}
    <text x="${PL}" y="${H - PB + 18}" fill="#8494ad" font-size="11">run 1</text>
    <text x="${W - PR}" y="${H - PB + 18}" fill="#8494ad" font-size="11" text-anchor="end">run ${n}</text>
    <text x="14" y="${H / 2}" fill="#8494ad" font-size="12" text-anchor="middle" transform="rotate(-90 14 ${H / 2})">speedup vs first reading (higher is better)</text>
    </svg><div class="legend" style="margin:6px 0 14px">${legend}</div>`;
}

let curHistStage = null;    // null = the run's per-token stack, the number the commit gate banks
let curHistSpeedup = false; // the all-stacks comparison, kept apart so no stage name can collide

function renderHistory(S) {
  const at = S.attempts || [];
  if (!at.length) {
    $("histbody").innerHTML = `<div class="empty">no levers applied yet — every attempt the run records lands here (same kernel log RUN_REPORT.md renders from)</div>`;
    return;
  }
  const stages = historyStages(S);
  const tok = ((S.serving || {}).per_token || {}).stage;
  // A remembered stage that this run does not report falls back to the default rather than charting
  // nothing, so the selection survives a poll but never outlives the data.
  if (curHistStage && !stages.includes(curHistStage)) curHistStage = null;
  const shown = curHistStage || (stages.includes(tok) ? tok : stages[0]) || null;
  const canCompare = historyStackSeries(S).length > 1;
  if (curHistSpeedup && !canCompare) curHistSpeedup = false;

  let bar = "";
  if (stages.length > 1 || canCompare) {
    const chips = stages.map(s =>
      `<button data-stage="${esc(s)}" class="${!curHistSpeedup && s === shown ? "on" : ""}">${esc(s)}${s === tok ? " · per-token" : ""}</button>`);
    if (canCompare) {
      chips.push(`<button data-mode="speedup" class="${curHistSpeedup ? "on" : ""}">all stacks · speedup</button>`);
    }
    bar = `<div class="segbar">${chips.join("")}</div>`;
  }

  const chart = curHistSpeedup ? speedupChart(S) : historyChart(S, shown);
  if (!chart) {
    $("histbody").innerHTML = bar + `<div class="empty">no charted trend for
      ${esc(shown || "this run")} yet — ${at.length} attempt(s) recorded, and a trend needs two measured
      readings of this stack.</div>`;
    return;
  }
  const caption = curHistSpeedup
    ? `Each stack against its OWN first recorded reading, so the stacks are comparable despite being
       measured in different units. Hover any point for the reading it came from.`
    : `Every logged experiment in order. Hover any point for the lever, the op, the reading and why it
       was tried.`;
  $("histbody").innerHTML = bar + `<div class="why" style="margin:-2px 0 8px">${caption}</div>` + chart;
}

const TABS = ["Roofline", "Compute vs Memory", "Latency Breakdown", "Power Analysis", "Scaling"];
let curTab = TABS[0];
let LAST = null;

/* ---- tiny inline-SVG chart helpers (no external libs; the box may be offline) ---- */
function logTicks(min, max) {
  const t = [];
  for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) t.push(Math.pow(10, e));
  return t;
}

/* Decade labels for a log axis. Fixed-decimal formatting rounds every decade below 0.1 to the same
   "0.0", which on these charts spanned five decades and made the whole low end unreadable; an
   exponent keeps each decade distinct without widening the axis. */
function logTickLabel(t) {
  if (t >= 1) return String(t);
  const e = Math.round(Math.log10(t));
  return e >= -2 ? t.toFixed(-e) : "1e" + e;
}

function logScatter({points, xGet, yGet, xLabel, yLabel, roof, height = 380}) {
  if (!points.length) return "";
  const W = 1000, H = height, PL = 64, PR = 18, PT = 18, PB = 46;
  let xs = points.map(xGet), ys = points.map(yGet);
  if (roof) { xs = xs.concat([roof.x0, roof.x1]); ys = ys.concat([roof.yAt(roof.x0), roof.yAt(roof.x1)]); }
  const xMin = Math.min(...xs) / 1.6, xMax = Math.max(...xs) * 1.6;
  const yMin = Math.min(...ys) / 1.6, yMax = Math.max(...ys) * 1.6;
  const X = v => PL + (Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin)) * (W - PL - PR);
  const Y = v => H - PB - (Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin)) * (H - PT - PB);
  let g = "";
  logTicks(xMin, xMax).forEach(t => {
    g += `<line x1="${X(t)}" y1="${PT}" x2="${X(t)}" y2="${H - PB}" stroke="#1e2c47" stroke-width="1"/>
      <text x="${X(t)}" y="${H - PB + 16}" fill="#8494ad" font-size="11" text-anchor="middle">${logTickLabel(t)}</text>`;
  });
  logTicks(yMin, yMax).forEach(t => {
    g += `<line x1="${PL}" y1="${Y(t)}" x2="${W - PR}" y2="${Y(t)}" stroke="#1e2c47" stroke-width="1"/>
      <text x="${PL - 8}" y="${Y(t) + 4}" fill="#8494ad" font-size="11" text-anchor="end">${logTickLabel(t)}</text>`;
  });
  if (roof) {
    g += `<polyline points="${X(roof.x0)},${Y(roof.yAt(roof.x0))} ${X(roof.knee)},${Y(roof.yAt(roof.knee))} ${X(roof.x1)},${Y(roof.yAt(roof.x1))}"
      fill="none" stroke="#f87171" stroke-width="2" stroke-dasharray="7 4"/>
      <text x="${X(roof.knee) + 8}" y="${Y(roof.yAt(roof.knee)) - 8}" fill="#f87171" font-size="11">roof</text>`;
  }
  const buckets = [...new Set(points.map(p => p.bucket))];
  const color = b => PALETTE[buckets.indexOf(b) % PALETTE.length];
  points.forEach(p => {
    g += `<circle cx="${X(xGet(p))}" cy="${Y(yGet(p))}" r="5.5" fill="${color(p.bucket)}" fill-opacity="0.85"
      stroke="#0a1020" stroke-width="1"><title>${esc(p.op || p.bucket)} — ${esc(p.bucket)}</title></circle>`;
  });
  const legend = buckets.map((b) =>
    `<span><i style="background:${color(b)}"></i>${esc(b)}</span>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">${g}
    <text x="${(W + PL) / 2}" y="${H - 6}" fill="#8494ad" font-size="12" text-anchor="middle">${esc(xLabel)}</text>
    <text x="14" y="${H / 2}" fill="#8494ad" font-size="12" text-anchor="middle" transform="rotate(-90 14 ${H / 2})">${esc(yLabel)}</text>
    </svg><div class="legend" style="margin:6px 0 4px">${legend}</div>`;
}

function rooflineChart(rf) {
  const pts = rf.points || [];
  if (!pts.length) return "";
  let roof = null;
  if (rf.bw_gbps && rf.peak_tflops) {
    const bwT = rf.bw_gbps / 1000.0;  // GB/s * FLOP/byte -> TFLOP/s
    const xs = pts.map(p => p.intensity);
    const x0 = Math.min(...xs) / 1.6, x1 = Math.max(...xs) * 1.6;
    const knee = rf.peak_tflops / bwT;
    roof = {x0, x1, knee, yAt: x => Math.min(rf.peak_tflops, bwT * x)};
  }
  return logScatter({points: pts, xGet: p => p.intensity, yGet: p => p.tflops,
    xLabel: "arithmetic intensity (FLOP / byte)", yLabel: "achieved TFLOP/s", roof});
}

function groupedBars(stages) {
  const W = 1000, H = 300, PL = 60, PB = 40, PT = 16;
  const maxV = Math.max(...stages.flatMap(s => [s.ms || 0, s.baseline_ms || 0]), 1e-9);
  const bw = (W - PL - 20) / stages.length;
  let g = "";
  stages.forEach((s, i) => {
    const x0 = PL + i * bw + bw * 0.15, bw2 = bw * 0.3;
    const hB = (s.baseline_ms || 0) / maxV * (H - PT - PB), hC = (s.ms || 0) / maxV * (H - PT - PB);
    g += `<rect x="${x0}" y="${H - PB - hB}" width="${bw2}" height="${hB}" fill="#54678a"/>
          <rect x="${x0 + bw2 + 4}" y="${H - PB - hC}" width="${bw2}" height="${hC}" fill="${(s.baseline_ms && s.ms < s.baseline_ms) ? "#34d399" : "#3b82f6"}"/>
          <text x="${x0 + bw2}" y="${H - PB + 16}" fill="#8494ad" font-size="12" text-anchor="middle">${esc(s.name)}</text>
          <text x="${x0 + bw2 / 2}" y="${H - PB - hB - 5}" fill="#8494ad" font-size="10.5" text-anchor="middle">${s.baseline_ms != null ? s.baseline_ms.toFixed(1) : ""}</text>
          <text x="${x0 + bw2 * 1.5 + 4}" y="${H - PB - hC - 5}" fill="#e6edf7" font-size="10.5" text-anchor="middle">${s.ms != null ? s.ms.toFixed(1) : ""}</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">${g}</svg>
    <div class="legend"><span><i style="background:#54678a"></i>baseline</span><span><i style="background:#3b82f6"></i>current</span><span><i style="background:#34d399"></i>current (faster)</span></div>`;
}

function seriesChart(thermal) {
  const series = Object.entries(thermal).filter(([, v]) => Array.isArray(v) && v.length && v.every(x => typeof x === "number"));
  if (!series.length) return "";
  const W = 1000, H = 280, PL = 60, PB = 36, PT = 16;
  const all = series.flatMap(([, v]) => v);
  const yMin = Math.min(...all), yMax = Math.max(...all), pad = (yMax - yMin || 1) * 0.15;
  const nMax = Math.max(...series.map(([, v]) => v.length));
  const X = i => PL + (nMax <= 1 ? 0 : i / (nMax - 1)) * (W - PL - 20);
  const Y = v => H - PB - (v - yMin + pad) / (yMax - yMin + 2 * pad) * (H - PT - PB);
  let g = "";
  // Gridded value ticks: without them this chart showed the SHAPE of the samples and none of their
  // magnitudes, which for a thermal profile is the whole point (how hot, not merely "rising"). No unit
  // is printed because the profile does not declare one.
  for (let k = 0; k <= 4; k++) {
    const v = yMin - pad + (yMax - yMin + 2 * pad) * k / 4;
    g += `<line x1="${PL}" y1="${Y(v)}" x2="${W - 20}" y2="${Y(v)}" stroke="#1e2c47" stroke-width="1"/>
      <text x="${PL - 8}" y="${Y(v) + 4}" fill="#8494ad" font-size="11" text-anchor="end">${v.toFixed(1)}</text>`;
  }
  series.forEach(([k, v], si) => {
    const c = PALETTE[si % PALETTE.length];
    g += `<polyline points="${v.map((x, i) => X(i) + "," + Y(x)).join(" ")}" fill="none" stroke="${c}" stroke-width="2"/>`;
    v.forEach((x, i) => {
      g += `<circle cx="${X(i)}" cy="${Y(x)}" r="2.6" fill="${c}"><title>${esc(k)} #${i + 1} — ${x}</title></circle>`;
    });
  });
  const legend = series.map(([k], si) => {
    const v = series[si][1];
    const lo = Math.min(...v), hi = Math.max(...v);
    return `<span><i style="background:${PALETTE[si % PALETTE.length]}"></i>${esc(k)} —
      ${v.length} samples, ${lo.toFixed(1)}–${hi.toFixed(1)}</span>`;
  }).join("");
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">${g}
    <text x="${(W + PL) / 2}" y="${H - 6}" fill="#8494ad" font-size="12" text-anchor="middle">sample</text></svg>
    <div class="legend" style="margin:6px 0 4px">${legend}</div>`;
}

function renderTab(S) {
  const el = $("tabbody");
  if (curTab === "Roofline") {
    const ops = S.opportunities || [];
    const m = S.metric || {};
    const h = S.headroom;
    let head = `<div class="legend" style="margin-bottom:10px">current ${fmtMs(m.current)} · baseline ${fmtMs(m.baseline)} · target ${fmtMs(m.target)}` +
      (h ? ` · modeled floor ${fmtMs2(h.floor_ms)} (−${h.pct.toFixed(0)}% headroom)` : "") + `</div>`;
    let chart = "";
    if (S.roofline) {
      chart = rooflineChart(S.roofline) ||
        `<div class="empty">no contraction ops in the profile to place on the roofline</div>`;
      if (!S.roofline.bw_gbps || !S.roofline.peak_tflops)
        chart += `<div class="caption">roof line needs the run's bandwidth + peak anchors; points alone are shown</div>`;
    }
    if (!ops.length) { el.innerHTML = head + chart + `<div class="empty">no profile buckets yet</div>`; return; }
    const rows = ops.map(o => `<tr><td>${esc(o.id)}</td><td>${fmtMs2(o.device_ms)}</td><td>${fmtPct(o.pct)}</td>
      <td>${esc((o.tags || {}).bound || "")}</td><td>${esc((o.tags || {}).memory || "")}</td>
      <td>${esc((o.tags || {}).fidelity || "")}</td><td>${esc((o.tags || {}).grid || "")}</td></tr>`).join("");
    el.innerHTML = head + chart +
      `<table><tr><th>bucket</th><th>device ms</th><th>% total</th><th>bound</th><th>memory</th><th>fidelity</th><th>grid</th></tr>${rows}</table>`;
  } else if (curTab === "Compute vs Memory") {
    const tops = (S.opportunities || []).flatMap(o => (o.top_ops || []).map(t => ({...t, bucket: o.id})));
    if (!tops.length) { el.innerHTML = `<div class="empty">no per-op profile yet</div>`; return; }
    const pts = tops.filter(t => t.bytes && t.device_ms);
    const chart = pts.length
      ? logScatter({points: pts, xGet: p => p.bytes / 1e9, yGet: p => p.device_ms,
                    xLabel: "bytes read (GB)", yLabel: "device time (ms)"})
      : `<div class="empty">ops carry no byte counts yet</div>`;
    const rows = tops.slice(0, 25).map(t => `<tr><td class="mono">${esc(t.op_code || "")}</td><td>${esc(t.bucket)}</td>
      <td>${t.device_ms != null ? Number(t.device_ms).toFixed(3) : "—"}</td>
      <td>${t.bytes != null ? (t.bytes / 1e9).toFixed(2) + " GB" : "—"}</td>
      <td>${t.cores ?? "—"}</td><td>${esc(t.fidelity || "")}</td></tr>`).join("");
    el.innerHTML = chart +
      `<table><tr><th>op</th><th>bucket</th><th>ms</th><th>bytes read</th><th>cores</th><th>fidelity</th></tr>${rows}</table>`;
  } else if (curTab === "Latency Breakdown") {
    const st = (S.stages || []).filter(s => s.ms != null);
    if (!st.length) { el.innerHTML = `<div class="empty">no per-stage timing captured yet</div>`; return; }
    const tot = st.reduce((a, s) => a + s.ms, 0) || 1;
    const stack = st.map((s, i) =>
      `<div style="width:${(s.ms / tot * 100).toFixed(2)}%;background:${PALETTE[i % PALETTE.length]}" title="${esc(s.name)} ${fmtMs(s.ms)}"></div>`).join("");
    const legend = st.map((s, i) =>
      `<span><i style="background:${PALETTE[i % PALETTE.length]}"></i>${esc(s.name)} — ${fmtMs2(s.ms)} (${(s.ms / tot * 100).toFixed(1)}%)</span>`).join("");
    const rows = st.map(s => `<tr><td>${esc(s.name)}</td><td>${fmtMs2(s.ms)}</td>
      <td>${fmtMs2(s.baseline_ms)}</td><td>${esc(s.path || "")}</td>
      <td>${s.bytes != null ? (s.bytes / 1e9).toFixed(2) + " GB" : "—"}</td></tr>`).join("");
    el.innerHTML = groupedBars(st) + `<div class="stack" style="margin-top:14px">${stack}</div><div class="legend">${legend}</div>
      <table style="margin-top:12px"><tr><th>stage</th><th>current</th><th>baseline</th><th>path</th><th>bytes</th></tr>${rows}</table>`;
  } else if (curTab === "Power Analysis") {
    const th = S.thermal;
    if (!th) { el.innerHTML = `<div class="empty">no thermal/power profile captured for this model yet</div>`; return; }
    const chart = seriesChart(th);
    const scalars = Object.entries(th).filter(([, v]) => !Array.isArray(v));
    el.innerHTML = chart +
      (scalars.length ? `<table>${scalars.map(([k, v]) =>
        `<tr><th>${esc(k)}</th><td>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</td></tr>`).join("")}</table>` : "");
  } else if (curTab === "Scaling") {
    const c = S.config || {}, tp = S.topology, env = S.env || {};
    let html = `<table>${Object.entries({...env, ...c}).map(([k, v]) =>
      `<tr><th>${esc(k)}</th><td>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</td></tr>`).join("")}</table>`;
    if (tp) html += `<h3 style="color:var(--dim);font-size:12px;margin:14px 0 6px">BOARD TOPOLOGY</h3>
      <table>${Object.entries(tp).map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</td></tr>`).join("")}</table>`;
    el.innerHTML = html;
  }
}

function render(S) {
  LAST = S;
  const b = $("livebadge");
  if (S.run.live) { b.className = "badge live"; b.textContent = "Live"; }
  else { b.className = "badge idle"; b.textContent = S.run.age_s != null ? "Idle" : "No run"; }
  $("model").textContent = (S.model && S.model.slug) || "…";
  $("runinfo").textContent = [S.run.id, S.run.state,
    S.run.iteration != null ? "iter " + S.run.iteration : ""].filter(Boolean).join(" · ");
  const now = new Date();
  $("updated").textContent = "Last updated: " + now.toLocaleTimeString([], {hour: "numeric", minute: "2-digit", second: "2-digit"}) +
    (S.run.age_s != null && !S.run.live ? " · last write " + Math.round(S.run.age_s) + "s ago" : "");
  renderCards(S); renderPerf(S); renderOpps(S); renderHistory(S); renderTab(S);
}

$("tabbar").innerHTML = TABS.map(t =>
  `<button data-t="${esc(t)}" class="${t === curTab ? "on" : ""}">${esc(t)}</button>`).join("");
$("tabbar").addEventListener("click", (e) => {
  const t = e.target && e.target.dataset ? e.target.dataset.t : null;
  if (!t) return;
  curTab = t;
  document.querySelectorAll("#tabbar button").forEach(b => b.classList.toggle("on", b.dataset.t === t));
  if (LAST) renderTab(LAST);
});

// Delegated: the history panel is rebuilt on every poll, so the listener lives on the container.
$("histbody").addEventListener("click", (e) => {
  const d = e.target && e.target.dataset ? e.target.dataset : null;
  if (!d) return;
  if (d.mode === "speedup") curHistSpeedup = true;
  else if (d.stage) { curHistSpeedup = false; curHistStage = d.stage; }
  else return;
  if (LAST) renderHistory(LAST);
});

async function poll() {
  try {
    const r = await fetch("/api/state", {cache: "no-store"});
    if (r.ok) render(await r.json());
  } catch (e) { /* keep last frame; next poll retries */ }
}
poll();
setInterval(poll, 2000);
</script>
</body>
</html>
"""
