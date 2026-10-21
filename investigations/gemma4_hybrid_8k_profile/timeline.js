/* SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
 * SPDX-License-Identifier: Apache-2.0 */
"use strict";

const byId = id => document.getElementById(id);
const fmt = (n, digits = 2) => n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const canvas = byId("timeline"), ctx = canvas.getContext("2d"), tooltip = byId("tooltip");
const left = 210, right = 20, top = 32, laneHeight = 19, groupHeight = 214, height = 472;
let data, cases, prefix = 253952, viewStart = 0, viewEnd = 1, total = 1, width = 900, dragging = null;

function showTip(event, title, rows, sources = []) {
  tooltip.innerHTML = `<strong>${escapeHTML(title)}</strong>` + rows.map(([k, v]) => `<div>${escapeHTML(k)}: <b>${escapeHTML(v)}</b></div>`).join("") +
    (sources.length ? '<div class="sources">' + sources.map(s => `<div>${escapeHTML(s.split("/").pop())}</div>`).join("") + "</div>" : "");
  tooltip.hidden = false;
  tooltip.style.left = `${Math.max(8, Math.min(event.clientX + 14, innerWidth - tooltip.offsetWidth - 10))}px`;
  tooltip.style.top = `${Math.max(8, Math.min(event.clientY + 14, innerHeight - tooltip.offsetHeight - 10))}px`;
}

function setPrefix(value) {
  prefix = value;
  cases = ["legacy", "hybrid"].map(path => data.cases.find(c => c.path === path && c.prefix === prefix));
  const [legacy, hybrid] = cases, delta = hybrid.total_ms - legacy.total_ms;
  total = hybrid.total_ms * 1000;
  document.querySelectorAll("[data-prefix]").forEach(b => {
    const selected = Number(b.dataset.prefix) === prefix;
    b.classList.toggle("selected", selected);
    b.setAttribute("aria-pressed", selected);
  });
  byId("cards").innerHTML = [["Legacy", `${fmt(legacy.total_ms)} ms`], ["Hybrid", `${fmt(hybrid.total_ms)} ms`], ["Added device time", `+${fmt(delta)} ms`], ["Hybrid / legacy", `${fmt(hybrid.total_ms / legacy.total_ms)}×`]]
    .map(([label, value]) => `<div class="card"><span>${label}</span><strong>${value}</strong></div>`).join("");
  byId("bars").innerHTML = cases.map((c, i) => `<div class="bar-row"><span>${c.path}</span><div class="bar">${c.categories_ms.map((ms, category) => `<div class="segment" data-case="${i}" data-category="${category}" style="background:${data.categories[category][1]};width:${ms / hybrid.total_ms * 100}%"></div>`).join("")}</div><span class="bar-number">${fmt(c.total_ms)} ms</span></div>`).join("");
  byId("legend").innerHTML = data.categories.map(([name, color]) => `<span><i class="swatch" style="background:${color}"></i>${name}</span>`).join("");
  byId("breakdown").innerHTML = data.categories.map(([name, color], i) => {
    const delta = hybrid.categories_ms[i] - legacy.categories_ms[i];
    return `<tr><td><i class="swatch" style="background:${color}"></i>${name}</td><td>${fmt(legacy.categories_ms[i])}</td><td>${fmt(hybrid.categories_ms[i])}</td><td>${delta >= 0 ? "+" : ""}${fmt(delta)}</td></tr>`;
  }).join("");
  byId("requests").innerHTML = hybrid.per_request.map(r => `<tr><td>${r.request + 1}</td><td>${fmt(r.tokens, 0)}</td><td>${fmt(r.global.mean_ms, 3)}</td><td>${fmt(r.sliding.mean_ms, 3)}</td></tr>`).join("");
  byId("takeaway").textContent = prefix ? "Global SDPA contributes 80% of the slowdown; sliding-window attention adds 15%. The 2560-token request is the global-attention outlier." : "Sliding-window attention contributes 58% of the slowdown. Device gaps add 9.7 ms. Projection and MLP matmul times remain unchanged.";
  byId("bars").querySelectorAll(".segment").forEach(element => {
    element.addEventListener("pointermove", event => {
      const c = cases[Number(element.dataset.case)], category = Number(element.dataset.category);
      showTip(event, data.categories[category][0], [["Path", c.path], ["Time", `${fmt(c.categories_ms[category], 3)} ms`], ["Share of replay", `${fmt(c.categories_ms[category] / c.total_ms * 100, 1)}%`], [category === 8 ? "Gap count" : "Program count", c.counts[category]]]);
    });
    element.addEventListener("pointerleave", () => { tooltip.hidden = true; });
  });
  fit();
}

function resize() {
  width = canvas.clientWidth;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * devicePixelRatio);
  canvas.height = Math.round(height * devicePixelRatio);
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  draw();
}

function draw() {
  if (!cases) return;
  const plotWidth = width - left - right, span = viewEnd - viewStart;
  const x = time => left + (time - viewStart) / span * plotWidth;
  ctx.clearRect(0, 0, width, height);
  ctx.font = "11px system-ui";
  for (let tick = 0; tick <= 8; tick++) {
    const time = viewStart + span * tick / 8, xpos = x(time);
    ctx.fillStyle = "#64748b";
    ctx.textAlign = "center";
    ctx.fillText(`${fmt(time / 1000, span < 10000 ? 2 : 1)} ms`, xpos, 17);
    ctx.strokeStyle = "#edf1f6";
    ctx.beginPath(); ctx.moveTo(xpos, top); ctx.lineTo(xpos, height - 16); ctx.stroke();
  }
  cases.forEach((c, track) => {
    const y0 = top + track * groupHeight;
    ctx.textAlign = "left";
    ctx.fillStyle = "#26364a";
    ctx.font = "bold 12px system-ui";
    ctx.fillText(`${c.path.toUpperCase()} · chip ${c.chip} · ${fmt(c.total_ms)} ms`, 12, y0 + 12);
    ctx.font = "10px system-ui";
    data.categories.forEach(([name, color], category) => {
      ctx.fillStyle = "#64748b";
      ctx.fillText(name, 12, y0 + 32 + category * laneHeight);
    });
    ctx.save(); ctx.beginPath(); ctx.rect(left, y0 + 20, plotWidth, 9 * laneHeight); ctx.clip();
    function rect(start, duration, category) {
      if (start > viewEnd || start + duration < viewStart) return;
      ctx.fillStyle = data.categories[category][1];
      const a = x(Math.max(start, viewStart)), b = x(Math.min(start + duration, viewEnd));
      ctx.fillRect(a, y0 + 21 + category * laneHeight, Math.max(0.6, b - a), laneHeight - 3);
    }
    let end = 0;
    c.events.forEach(e => {
      if (e[0] > end) rect(end, e[0] - end, 8);
      rect(e[0], e[1], e[2]);
      end = e[0] + e[1];
    });
    ctx.restore();
  });
  byId("range").textContent = `${fmt(viewStart / 1000, 3)}–${fmt(viewEnd / 1000, 3)} ms shown · ${fmt(total / (viewEnd - viewStart), 1)}× zoom`;
}

function fit() { viewStart = 0; viewEnd = total; tooltip.hidden = true; resize(); }
function changeView(start, span) {
  span = Math.min(total, Math.max(20, span));
  viewStart = Math.max(0, Math.min(start, total - span)); viewEnd = viewStart + span;
  tooltip.hidden = true; draw();
}
function zoom(factor, anchor = 0.5) {
  const span = viewEnd - viewStart, next = Math.max(20, Math.min(total, span * factor));
  changeView(viewStart + anchor * (span - next), next);
}

canvas.addEventListener("wheel", event => {
  if (!cases) return;
  event.preventDefault();
  const xpos = event.clientX - canvas.getBoundingClientRect().left;
  zoom(event.deltaY > 0 ? 1.4 : 1 / 1.4, Math.max(0, Math.min(1, (xpos - left) / (width - left - right))));
}, { passive: false });
canvas.addEventListener("pointerdown", event => {
  dragging = { x: event.clientX, start: viewStart, span: viewEnd - viewStart };
  canvas.setPointerCapture(event.pointerId); tooltip.hidden = true;
});
canvas.addEventListener("pointerup", () => { dragging = null; });
canvas.addEventListener("pointercancel", () => { dragging = null; });
canvas.addEventListener("dblclick", fit);
canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });
canvas.addEventListener("pointermove", event => {
  if (!cases) return;
  if (dragging) { changeView(dragging.start - (event.clientX - dragging.x) / (width - left - right) * dragging.span, dragging.span); return; }
  const bounds = canvas.getBoundingClientRect(), xpos = event.clientX - bounds.left, ypos = event.clientY - bounds.top;
  const track = Math.floor((ypos - top) / groupHeight), c = cases[track];
  const category = Math.floor((ypos - top - track * groupHeight - 21) / laneHeight);
  if (!c || category < 0 || category > 8 || xpos < left || xpos > width - right) { tooltip.hidden = true; return; }
  const time = viewStart + (xpos - left) / (width - left - right) * (viewEnd - viewStart);
  let end = 0, hit = null;
  for (const e of c.events) {
    if (category === 8 && time >= end && time < e[0]) { hit = [end, e[0] - end, 8, -1, -1, -1, -1]; break; }
    if (e[2] === category && time >= e[0] && time <= e[0] + e[1]) { hit = e; break; }
    end = e[0] + e[1];
  }
  if (!hit) { tooltip.hidden = true; return; }
  const [start, duration, cat, source, layer, request, phase] = hit;
  const rows = [["Path", c.path], ["Chip", c.chip], ["Prefix", fmt(c.prefix, 0)], ["Start", `${fmt(start / 1000, 3)} ms`], ["Duration", `${fmt(duration, 3)} µs`]];
  if (layer >= 0) rows.push(["Layer", layer + 1]);
  if (request >= 0) rows.push(["Request", request + 1], ["Query tokens", c.per_request[request].tokens]);
  if (cat < 2) rows.push(["Includes", cat === 0 ? "Attention + internal all-gather" : "Attention + internal halo"]);
  showTip(event, phase >= 0 ? data.matmul_names[phase] : data.categories[cat][0], rows, source >= 0 ? data.sources[source] : []);
});
byId("fit").addEventListener("click", fit);
byId("zoom-in").addEventListener("click", () => zoom(0.5));
byId("zoom-out").addEventListener("click", () => zoom(2));
document.querySelectorAll("[data-prefix]").forEach(button => button.addEventListener("click", () => { if (data) setPrefix(Number(button.dataset.prefix)); }));
window.addEventListener("resize", resize);
fetch("timeline.json").then(response => {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}).then(result => {
  data = result; byId("status").hidden = true; byId("content").hidden = false; setPrefix(prefix);
}).catch(error => {
  byId("status").textContent = `Could not load timeline.json (${error.message}). Serve this directory with python -m http.server 8520.`;
});
