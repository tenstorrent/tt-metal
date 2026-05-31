# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Build a self-contained HTML visualization from the kimi-50k chunked-prefill perf
sweep markdown (kimi_50k_chunked_perf_sweep.md).

Usage:
  python tests/nightly/blackhole/sdpa/make_kimi_sweep_viz.py \
      [kimi_50k_chunked_perf_sweep.md] [kimi_50k_chunked_sweep_viz.html]

No external libraries — charts are inline SVG/JS, dark theme, works offline.
"""
import json
import os
import re
import sys


def parse_md(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            # 13 cols = with subblock fields; 11 = legacy without.
            if len(cells) not in (11, 13):
                continue
            if cells[0] in ("seq/dev", "") or cells[0].startswith("--"):
                continue
            try:
                per_device = int(cells[0])
            except ValueError:
                continue
            chunk_lbl, target_lbl = cells[1], cells[2]
            try:
                q = int(cells[3])
                k = int(cells[4])
                n_prefix = int(cells[5])
            except ValueError:
                continue
            if len(cells) == 13:
                qk_sb = cells[9] if cells[9] not in ("-", "") else None
                av_sb = cells[10] if cells[10] not in ("-", "") else None
                util_idx, status_idx = 11, 12
            else:
                qk_sb = av_sb = None
                util_idx, status_idx = 9, 10
            status_raw = cells[status_idx]
            ok = status_raw == "OK"

            def num(s):
                s = s.replace("*", "").replace("%", "").strip()
                try:
                    return float(s)
                except ValueError:
                    return None

            fpu_min = fpu_max = None
            if "-" in cells[8] and cells[8] != "-":
                parts = cells[8].split("-")
                try:
                    fpu_min, fpu_max = float(parts[0]), float(parts[1])
                except ValueError:
                    pass

            rows.append(
                {
                    "per_device": per_device,
                    "chunk_sp8": per_device * 8,
                    "chunk_lbl": chunk_lbl,
                    "target": target_lbl,
                    "q": q,
                    "k": k,
                    "n_prefix": n_prefix,
                    "dur": num(cells[6]) if ok else None,
                    "cores": int(num(cells[7])) if ok and num(cells[7]) is not None else None,
                    "fpu_min": fpu_min,
                    "fpu_max": fpu_max,
                    "qk_sb": qk_sb,
                    "av_sb": av_sb,
                    "util": num(cells[util_idx]) if ok else None,
                    "status": "OK" if ok else status_raw,
                }
            )
    return rows


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kimi-50k Chunked-Prefill Perf Sweep</title>
<style>
  :root{
    --bg:#0b0f14; --panel:#121821; --panel2:#0e141c; --line:#1f2937;
    --ink:#e6edf3; --dim:#8b9bb0; --dim2:#5c6b80;
    --accent:#34d3c4; --accent2:#f0b429; --bad:#ef5d6b; --good:#3ddc84; --warn:#f0883e;
    --mono:'SFMono-Regular',ui-monospace,'JetBrains Mono',Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.5}
  a{color:var(--accent)}
  .wrap{max-width:1240px;margin:0 auto;padding:0 22px 90px}
  header.top{padding:28px 0 14px;border-bottom:1px solid var(--line)}
  header.top h1{margin:0;font-size:24px;letter-spacing:-.4px}
  header.top .sub{color:var(--dim);font-size:13px;margin-top:6px;font-family:var(--mono)}
  .pill{display:inline-block;font-family:var(--mono);font-size:11px;padding:2px 8px;border-radius:20px;
    border:1px solid var(--line);color:var(--dim);margin-right:6px;margin-top:8px}
  nav.tabs{position:sticky;top:0;z-index:30;background:rgba(11,15,20,.93);backdrop-filter:blur(8px);
    border-bottom:1px solid var(--line);margin:0 -22px 22px;padding:0 22px;display:flex;gap:2px;overflow-x:auto}
  nav.tabs button{appearance:none;background:none;border:none;color:var(--dim);font-family:var(--sans);
    font-size:13.5px;padding:13px 14px 11px;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap}
  nav.tabs button.active{color:var(--ink);border-bottom-color:var(--accent)}
  .tab{display:none} .tab.active{display:block}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
  h2{font-size:13px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);margin:0 0 12px}
  .controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
  label{color:var(--dim);font-size:12px;margin-right:6px}
  select{background:var(--panel2);color:var(--ink);border:1px solid var(--line);border-radius:6px;
    padding:5px 8px;font-family:var(--mono);font-size:13px}
  table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12.5px}
  th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
  th{color:var(--dim);text-transform:uppercase;font-size:10.5px;letter-spacing:.5px;position:sticky;top:46px;background:var(--panel)}
  td.l,th.l{text-align:left}
  tr:hover td{background:var(--panel2)}
  .util{font-weight:700;color:var(--accent)}
  .best{color:var(--accent2)}
  .scroll{max-height:560px;overflow:auto;border:1px solid var(--line);border-radius:8px}
  .legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;font-family:var(--mono);font-size:12px}
  .legend span{display:inline-flex;align-items:center;gap:6px;color:var(--dim)}
  .sw{width:13px;height:13px;border-radius:3px;display:inline-block}
  .note{color:var(--dim2);font-size:12px;margin-top:8px;font-family:var(--mono)}
  svg text{font-family:var(--mono);fill:var(--dim)}
  .tip{position:fixed;pointer-events:none;background:#000;border:1px solid var(--accent);border-radius:6px;
    padding:6px 9px;font-family:var(--mono);font-size:12px;color:var(--ink);display:none;z-index:99;white-space:nowrap}
  .kpi{display:flex;gap:14px;flex-wrap:wrap}
  .kpi .box{flex:1;min-width:160px;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
  .kpi .box .v{font-size:22px;font-weight:700;color:var(--accent);font-family:var(--mono)}
  .kpi .box .v.amber{color:var(--accent2)}
  .kpi .box .lbl{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <h1>Kimi-50k Chunked-Prefill Perf Sweep</h1>
    <div class="sub">ring joint SDPA · pure-compute math utilization · target 50k+5k (sp=8)</div>
    <div id="pills"></div>
  </header>

  <nav class="tabs">
    <button data-tab="best" class="active">Best configs</button>
    <button data-tab="keff">k_chunk effect</button>
    <button data-tab="qeff">q_chunk effect</button>
    <button data-tab="seff">seq/dev effect</button>
    <button data-tab="heat">Heatmap (q×k)</button>
    <button data-tab="sblk">Subblock</button>
    <button data-tab="raw">Raw data</button>
  </nav>

  <section class="tab active" id="tab-best">
    <div class="card">
      <h2>Headline + global maxima</h2>
      <div class="kpi" id="kpi"></div>
      <div class="note" id="best-note"></div>
    </div>
    <div class="card">
      <h2>Top 20 configurations by math util</h2>
      <div class="scroll"><table id="top-tbl"></table></div>
    </div>
    <div class="card">
      <h2>Best config per seq/dev (its peak over q × k)</h2>
      <div class="scroll"><table id="bestper-tbl"></table></div>
    </div>
  </section>

  <section class="tab" id="tab-keff">
    <div class="card">
      <h2>Math util vs k_chunk — how k tiling drives util</h2>
      <div class="controls">
        <div><label>seq/dev</label><select id="keff-sd"></select></div>
        <span class="note">one line per q_chunk · line ends where the k-loop hit OOM / covered the shard</span>
      </div>
      <div id="keff-chart"></div>
      <div class="legend" id="keff-legend"></div>
    </div>
  </section>

  <section class="tab" id="tab-qeff">
    <div class="card">
      <h2>q_chunk effect — best util reachable per q (over all k)</h2>
      <div class="controls">
        <div><label>seq/dev</label><select id="qeff-sd"></select></div>
      </div>
      <div id="qeff-chart"></div>
      <div class="note">bars = peak util at that q_chunk; the k that achieved it is labeled.</div>
    </div>
  </section>

  <section class="tab" id="tab-seff">
    <div class="card">
      <h2>seq/dev effect — peak util across the whole sweep range</h2>
      <div class="controls">
        <div><label>q_chunk</label><select id="seff-q"></select></div>
      </div>
      <div id="seff-chart"></div>
      <div class="note">x = seq/dev (sp=8 chunk size in parens) · peak util over k at the selected q_chunk.</div>
    </div>
  </section>

  <section class="tab" id="tab-heat">
    <div class="card">
      <h2>q × k utilization heatmap</h2>
      <div class="controls">
        <div><label>seq/dev</label><select id="heat-sd"></select></div>
      </div>
      <div id="heat-chart"></div>
      <div class="note">brighter = higher math util · grey = OOM / not measured.</div>
    </div>
  </section>

  <section class="tab" id="tab-sblk">
    <div class="card">
      <h2>Matmul subblock geometry vs util — the bonus field</h2>
      <p class="note">Each config's two matmul output-subblocks (tiles, h×w) as chosen by
      <code>ring_joint_sdpa_program_factory.cpp</code>: <b>QK&#8868;</b> (first) and <b>attn@V</b> (second).
      Bars = peak math util reached by each distinct <code>qk / av</code> subblock pair across the whole sweep.</p>
      <div id="sblk-chart"></div>
      <div class="legend" id="sblk-legend"></div>
    </div>
    <div class="card">
      <h2>Per subblock-pair detail</h2>
      <div class="scroll"><table id="sblk-tbl"></table></div>
    </div>
  </section>

  <section class="tab" id="tab-raw">
    <div class="card">
      <h2>All measurements</h2>
      <div class="controls"><label>sort</label>
        <select id="raw-sort">
          <option value="util">util desc</option>
          <option value="order">sweep order</option>
          <option value="sd">seq/dev, q, k</option>
        </select>
      </div>
      <div class="scroll"><table id="raw-tbl"></table></div>
    </div>
  </section>
</div>
<div class="tip" id="tip"></div>

<script>
const DATA = /*DATA*/;
const OK = DATA.filter(d => d.status === "OK");
const QCOLORS = {32:"#34d3c4",64:"#f0b429",96:"#a371f7",128:"#ef5d6b",160:"#3ddc84",192:"#f0883e"};
const tip = document.getElementById('tip');
function showTip(html,e){tip.innerHTML=html;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';}
function hideTip(){tip.style.display='none';}
const fmt = (v,d=1)=> v==null?'–':v.toFixed(d);
const sds = [...new Set(OK.map(d=>d.per_device))].sort((a,b)=>a-b);
const qs  = [...new Set(OK.map(d=>d.q))].sort((a,b)=>a-b);
const sdLabel = sd => { const r = DATA.find(d=>d.per_device===sd); return sd + (r? " ("+r.chunk_lbl+")":""); };

// util -> color (dim teal -> bright)
function utilColor(u, umin, umax){
  if(u==null) return '#2a3340';
  const t = umax>umin ? (u-umin)/(umax-umin) : 1;
  // interpolate dark slate -> teal -> amber
  const stops = [[26,40,55],[29,107,99],[52,211,196],[240,180,41]];
  const seg = t*(stops.length-1); const i = Math.min(stops.length-2, Math.floor(seg)); const f = seg-i;
  const c = stops[i].map((a,j)=>Math.round(a+(stops[i+1][j]-a)*f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ---- generic SVG line chart ----
function lineChart(el, series, {xlab,ylab,xticks,yMin,yMax}){
  const W=920,H=420,m={l:58,r:20,t:18,b:48};
  const xs = [...new Set(series.flatMap(s=>s.points.map(p=>p.x)))].sort((a,b)=>a-b);
  const xmin=Math.min(...xs), xmax=Math.max(...xs);
  const ymin = yMin!=null?yMin:Math.min(...series.flatMap(s=>s.points.map(p=>p.y)));
  const ymax = yMax!=null?yMax:Math.max(...series.flatMap(s=>s.points.map(p=>p.y)));
  const X = x => m.l + (xmax>xmin?(x-xmin)/(xmax-xmin):0.5)*(W-m.l-m.r);
  const Y = y => H-m.b - (ymax>ymin?(y-ymin)/(ymax-ymin):0.5)*(H-m.t-m.b);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  // grid + y ticks
  const yn=6;
  for(let i=0;i<=yn;i++){const yv=ymin+(ymax-ymin)*i/yn;const y=Y(yv);
    s+=`<line x1="${m.l}" y1="${y}" x2="${W-m.r}" y2="${y}" stroke="#1f2937"/>`;
    s+=`<text x="${m.l-8}" y="${y+4}" text-anchor="end" font-size="11">${yv.toFixed(0)}</text>`;}
  // x ticks
  (xticks||xs).forEach(xv=>{const x=X(xv);
    s+=`<line x1="${x}" y1="${m.t}" x2="${x}" y2="${H-m.b}" stroke="#161d27"/>`;
    s+=`<text x="${x}" y="${H-m.b+16}" text-anchor="middle" font-size="10">${xv}</text>`;});
  s+=`<text x="${m.l+(W-m.l-m.r)/2}" y="${H-6}" text-anchor="middle" font-size="11">${xlab}</text>`;
  s+=`<text transform="translate(14 ${m.t+(H-m.t-m.b)/2}) rotate(-90)" text-anchor="middle" font-size="11">${ylab}</text>`;
  series.forEach(se=>{
    const pts=se.points.slice().sort((a,b)=>a.x-b.x);
    if(pts.length>1){let d=pts.map((p,i)=>(i?'L':'M')+X(p.x)+' '+Y(p.y)).join(' ');
      s+=`<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2"/>`;}
    pts.forEach(p=>{s+=`<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="3.5" fill="${se.color}" class="pt"
      data-tip="${se.name} · k=${p.x} · util ${p.y.toFixed(1)}%${p.extra?(' · '+p.extra):''}"/>`;});
  });
  s+=`</svg>`;
  el.innerHTML=s;
  el.querySelectorAll('.pt').forEach(c=>{
    c.addEventListener('mousemove',e=>showTip(c.getAttribute('data-tip'),e));
    c.addEventListener('mouseleave',hideTip);});
}

// ---- bar chart ----
function barChart(el, items, {ylab,labelKey}){
  const W=920,H=380,m={l:58,r:20,t:18,b:64};
  const ymax=Math.max(...items.map(d=>d.v))*1.12;
  const bw=(W-m.l-m.r)/items.length;
  const Y=v=>H-m.b-(v/ymax)*(H-m.t-m.b);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  for(let i=0;i<=5;i++){const yv=ymax*i/5;const y=Y(yv);
    s+=`<line x1="${m.l}" y1="${y}" x2="${W-m.r}" y2="${y}" stroke="#1f2937"/>`;
    s+=`<text x="${m.l-8}" y="${y+4}" text-anchor="end" font-size="11">${yv.toFixed(0)}</text>`;}
  items.forEach((d,i)=>{const x=m.l+i*bw+bw*0.18,w=bw*0.64,y=Y(d.v);
    s+=`<rect x="${x}" y="${y}" width="${w}" height="${H-m.b-y}" rx="4" fill="${d.color||'#34d3c4'}"
        class="pt" data-tip="${d.tip||''}"/>`;
    s+=`<text x="${x+w/2}" y="${y-6}" text-anchor="middle" font-size="11" fill="#e6edf3">${d.v.toFixed(1)}</text>`;
    s+=`<text x="${x+w/2}" y="${H-m.b+16}" text-anchor="middle" font-size="11">${d.label}</text>`;
    if(d.sub) s+=`<text x="${x+w/2}" y="${H-m.b+30}" text-anchor="middle" font-size="9.5" fill="#5c6b80">${d.sub}</text>`;});
  s+=`<text transform="translate(14 ${m.t+(H-m.t-m.b)/2}) rotate(-90)" text-anchor="middle" font-size="11">${ylab}</text>`;
  s+=`</svg>`; el.innerHTML=s;
  el.querySelectorAll('.pt').forEach(c=>{c.addEventListener('mousemove',e=>showTip(c.getAttribute('data-tip'),e));c.addEventListener('mouseleave',hideTip);});
}

// ---- heatmap ----
function heatmap(el, sd){
  const rows = OK.filter(d=>d.per_device===sd);
  const ks=[...new Set(OK.map(d=>d.k))].sort((a,b)=>a-b);
  const umin=Math.min(...OK.map(d=>d.util)), umax=Math.max(...OK.map(d=>d.util));
  const cell=24, m={l:48,t:24,r:14,b:40};
  const W=m.l+ks.length*cell+m.r, H=m.t+qs.length*cell+m.b;
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${Math.min(W,1180)}px">`;
  qs.forEach((q,qi)=>{ s+=`<text x="${m.l-6}" y="${m.t+qi*cell+cell/2+4}" text-anchor="end" font-size="11">${q}</text>`;
    ks.forEach((k,ki)=>{ const r=rows.find(d=>d.q===q&&d.k===k);
      const x=m.l+ki*cell,y=m.t+qi*cell;
      const col=r?utilColor(r.util,umin,umax):'#161d27';
      s+=`<rect x="${x}" y="${y}" width="${cell-2}" height="${cell-2}" rx="3" fill="${col}" class="pt"
          data-tip="${r?('q='+q+' k='+k+' · util '+r.util.toFixed(1)+'% · '+r.dur.toFixed(2)+'ms · sb '+(r.qk_sb||'?')+'/'+(r.av_sb||'?')):('q='+q+' k='+k+' · n/a')}"/>`;});});
  ks.forEach((k,ki)=>{ if(ki%2===0||ks.length<14) s+=`<text x="${m.l+ki*cell+cell/2}" y="${H-m.b+16}" text-anchor="middle" font-size="9" transform="rotate(40 ${m.l+ki*cell+cell/2} ${H-m.b+16})">${k}</text>`;});
  s+=`<text x="${m.l+ks.length*cell/2}" y="${H-6}" text-anchor="middle" font-size="11">k_chunk</text>`;
  s+=`<text transform="translate(12 ${m.t+qs.length*cell/2}) rotate(-90)" text-anchor="middle" font-size="11">q_chunk</text>`;
  s+=`</svg>`; el.innerHTML=s;
  el.querySelectorAll('.pt').forEach(c=>{c.addEventListener('mousemove',e=>showTip(c.getAttribute('data-tip'),e));c.addEventListener('mouseleave',hideTip);});
}

// ---- populate ----
function fillSelect(id, vals, labeler){const el=document.getElementById(id);
  el.innerHTML=vals.map(v=>`<option value="${v}">${labeler?labeler(v):v}</option>`).join('');}
function setPills(){
  const nFail=DATA.length-OK.length;
  document.getElementById('pills').innerHTML =
    [`${OK.length} configs measured`, `${sds.length} seq/dev`, `q ∈ {${qs.join(', ')}}`,
     `${nFail} OOM/fail`, `100 compute cores (CCL stripped)`]
    .map(p=>`<span class="pill">${p}</span>`).join('');
}

function tab_best(){
  const best = OK.slice().sort((a,b)=>b.util-a.util);
  const headline = OK.filter(d=>d.target.startsWith('50k+5k')).sort((a,b)=>b.util-a.util)[0];
  const top1 = best[0];
  const kpi=document.getElementById('kpi');
  const boxes=[];
  if(headline) boxes.push(`<div class="box"><div class="v">${headline.util.toFixed(1)}%</div>
     <div class="lbl">best @ 50k+5k (q${headline.q}/k${headline.k})</div></div>`);
  if(top1) boxes.push(`<div class="box"><div class="v amber">${top1.util.toFixed(1)}%</div>
     <div class="lbl">global best — ${top1.per_device} (${top1.chunk_lbl}) q${top1.q}/k${top1.k}</div></div>`);
  boxes.push(`<div class="box"><div class="v">${OK.length}</div><div class="lbl">measurements</div></div>`);
  kpi.innerHTML=boxes.join('');
  document.getElementById('best-note').textContent =
    top1? `Fastest kernel among the best: ${Math.min(...best.slice(0,20).map(d=>d.dur)).toFixed(2)} ms.`:'';
  // top 20
  let t=`<tr><th class="l">rank</th><th>seq/dev</th><th>chunk(sp8)</th><th class="l">target</th><th>q</th><th>k</th><th>n_prefix</th><th>dur ms</th><th>cores</th><th>qk sb</th><th>av sb</th><th>util</th></tr>`;
  best.slice(0,20).forEach((d,i)=>{ t+=`<tr><td>${i+1}</td><td>${d.per_device}</td><td>${d.chunk_lbl}</td>
    <td class="l">${d.target}</td><td>${d.q}</td><td>${d.k}</td><td>${d.n_prefix}</td>
    <td>${fmt(d.dur,2)}</td><td>${d.cores}</td><td>${d.qk_sb||'–'}</td><td>${d.av_sb||'–'}</td>
    <td class="util">${d.util.toFixed(1)}%</td></tr>`;});
  document.getElementById('top-tbl').innerHTML=t;
  // best per seq/dev
  let b=`<tr><th>seq/dev</th><th>chunk(sp8)</th><th class="l">target</th><th>best q</th><th>best k</th><th>qk sb</th><th>av sb</th><th>dur ms</th><th>peak util</th></tr>`;
  sds.forEach(sd=>{const r=OK.filter(d=>d.per_device===sd).sort((a,b)=>b.util-a.util)[0]; if(!r)return;
    b+=`<tr><td>${sd}</td><td>${r.chunk_lbl}</td><td class="l">${r.target}</td><td>${r.q}</td><td>${r.k}</td>
        <td>${r.qk_sb||'–'}</td><td>${r.av_sb||'–'}</td><td>${fmt(r.dur,2)}</td><td class="util">${r.util.toFixed(1)}%</td></tr>`;});
  document.getElementById('bestper-tbl').innerHTML=b;
}

function tab_keff(){
  const sd=+document.getElementById('keff-sd').value;
  const series=qs.map(q=>({name:'q='+q,color:QCOLORS[q]||'#8b9bb0',
    points:OK.filter(d=>d.per_device===sd&&d.q===q).map(d=>({x:d.k,y:d.util,
      extra:d.dur.toFixed(2)+'ms · sb '+(d.qk_sb||'?')+'/'+(d.av_sb||'?')}))}))
    .filter(s=>s.points.length);
  lineChart(document.getElementById('keff-chart'),series,{xlab:'k_chunk',ylab:'math util %'});
  document.getElementById('keff-legend').innerHTML =
    series.map(s=>`<span><span class="sw" style="background:${s.color}"></span>${s.name}</span>`).join('');
}
function tab_qeff(){
  const sd=+document.getElementById('qeff-sd').value;
  const items=qs.map(q=>{const r=OK.filter(d=>d.per_device===sd&&d.q===q).sort((a,b)=>b.util-a.util)[0];
    return r?{label:'q='+q,v:r.util,sub:'@k='+r.k,color:QCOLORS[q]||'#34d3c4',
      tip:`q=${q} · peak ${r.util.toFixed(1)}% @ k=${r.k} · ${r.dur.toFixed(2)}ms`}:null;}).filter(Boolean);
  barChart(document.getElementById('qeff-chart'),items,{ylab:'peak math util %'});
}
function tab_seff(){
  const q=+document.getElementById('seff-q').value;
  const items=sds.map(sd=>{const r=OK.filter(d=>d.per_device===sd&&d.q===q).sort((a,b)=>b.util-a.util)[0];
    return r?{label:''+sd,v:r.util,sub:r.chunk_lbl+' @k'+r.k,color:'#34d3c4',
      tip:`seq/dev=${sd} (${r.chunk_lbl}) · peak ${r.util.toFixed(1)}% @ k=${r.k}`}:null;}).filter(Boolean);
  barChart(document.getElementById('seff-chart'),items,{ylab:'peak math util %'});
}
function tab_heat(){ heatmap(document.getElementById('heat-chart'), +document.getElementById('heat-sd').value); }

function tab_sblk(){
  const withSb = OK.filter(d=>d.qk_sb&&d.av_sb);
  const el=document.getElementById('sblk-chart');
  if(!withSb.length){ el.innerHTML='<p class="note">No subblock data in this run (older sweep without the bonus field).</p>';
    document.getElementById('sblk-tbl').innerHTML=''; document.getElementById('sblk-legend').innerHTML=''; return; }
  // group by qk/av pair, keep the peak-util config and the set of q that hit it
  const groups={};
  withSb.forEach(d=>{const key=d.qk_sb+' / '+d.av_sb;
    (groups[key]=groups[key]||[]).push(d);});
  const items=Object.entries(groups).map(([key,arr])=>{
    const best=arr.slice().sort((a,b)=>b.util-a.util)[0];
    const qset=[...new Set(arr.map(a=>a.q))].sort((a,b)=>a-b);
    return {label:key,v:best.util,sub:'n='+arr.length,color:QCOLORS[best.q]||'#34d3c4',
      tip:`qk/av = ${key} · peak ${best.util.toFixed(1)}% @ seq/dev ${best.per_device} q${best.q}/k${best.k} · q∈{${qset.join(',')}} · ${arr.length} configs`};
  }).sort((a,b)=>b.v-a.v);
  barChart(el,items,{ylab:'peak math util %'});
  document.getElementById('sblk-legend').innerHTML =
    '<span class="note">bar label = qk_out_subblock / av_out_subblock (tiles, h×w) · color = q_chunk of the peak config</span>';
  // detail table
  let t=`<tr><th class="l">qk / av subblock</th><th>configs</th><th>q's seen</th><th>peak util</th><th>best seq/dev</th><th>best q</th><th>best k</th><th>min dur ms</th></tr>`;
  Object.entries(groups).map(([key,arr])=>{
    const best=arr.slice().sort((a,b)=>b.util-a.util)[0];
    return {key,arr,best};
  }).sort((a,b)=>b.best.util-a.best.util).forEach(({key,arr,best})=>{
    const qset=[...new Set(arr.map(a=>a.q))].sort((a,b)=>a-b);
    t+=`<tr><td class="l">${key}</td><td>${arr.length}</td><td>${qset.join(', ')}</td>
        <td class="util">${best.util.toFixed(1)}%</td><td>${best.per_device}</td><td>${best.q}</td><td>${best.k}</td>
        <td>${Math.min(...arr.map(a=>a.dur)).toFixed(2)}</td></tr>`;});
  document.getElementById('sblk-tbl').innerHTML=t;
}

function tab_raw(){
  const mode=document.getElementById('raw-sort').value;
  let r=DATA.slice();
  if(mode==='util') r.sort((a,b)=>(b.util||-1)-(a.util||-1));
  else if(mode==='sd') r.sort((a,b)=>a.per_device-b.per_device||a.q-b.q||a.k-b.k);
  let t=`<tr><th>seq/dev</th><th>chunk(sp8)</th><th class="l">target</th><th>q</th><th>k</th><th>n_prefix</th><th>dur ms</th><th>cores</th><th>FPU%</th><th>qk sb</th><th>av sb</th><th>util</th><th class="l">status</th></tr>`;
  r.forEach(d=>{const okc=d.status==='OK';
    t+=`<tr><td>${d.per_device}</td><td>${d.chunk_lbl}</td><td class="l">${d.target}</td><td>${d.q}</td><td>${d.k}</td>
      <td>${d.n_prefix}</td><td>${fmt(d.dur,2)}</td><td>${d.cores??'–'}</td>
      <td>${d.fpu_min!=null?d.fpu_min.toFixed(0)+'-'+d.fpu_max.toFixed(0):'–'}</td>
      <td>${d.qk_sb||'–'}</td><td>${d.av_sb||'–'}</td>
      <td class="${okc?'util':''}">${okc?d.util.toFixed(1)+'%':'–'}</td>
      <td class="l" style="color:${okc?'#3ddc84':'#ef5d6b'}">${d.status}</td></tr>`;});
  document.getElementById('raw-tbl').innerHTML=t;
}

// wire up
document.querySelectorAll('nav.tabs button').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('nav.tabs button').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); document.getElementById('tab-'+b.dataset.tab).classList.add('active');
}));
fillSelect('keff-sd',sds,sdLabel); fillSelect('qeff-sd',sds,sdLabel);
fillSelect('heat-sd',sds,sdLabel); fillSelect('seff-q',qs);
// default seq/dev = 640 if present (the 50k+5k headline)
[['keff-sd'],['qeff-sd'],['heat-sd']].forEach(([id])=>{ if(sds.includes(640)) document.getElementById(id).value=640;});
document.getElementById('keff-sd').addEventListener('change',tab_keff);
document.getElementById('qeff-sd').addEventListener('change',tab_qeff);
document.getElementById('seff-q').addEventListener('change',tab_seff);
document.getElementById('heat-sd').addEventListener('change',tab_heat);
document.getElementById('raw-sort').addEventListener('change',tab_raw);
setPills(); tab_best(); tab_keff(); tab_qeff(); tab_seff(); tab_heat(); tab_sblk(); tab_raw();
</script>
</body>
</html>
"""


def main():
    md = sys.argv[1] if len(sys.argv) > 1 else "kimi_50k_chunked_perf_sweep.md"
    out = sys.argv[2] if len(sys.argv) > 2 else "kimi_50k_chunked_sweep_viz.html"
    if not os.path.exists(md):
        print(f"ERROR: {md} not found", file=sys.stderr)
        sys.exit(1)
    rows = parse_md(md)
    ok = [r for r in rows if r["status"] == "OK"]
    print(f"parsed {len(rows)} rows ({len(ok)} OK) from {md}")
    html = HTML_TEMPLATE.replace("/*DATA*/", json.dumps(rows))
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
