#!/usr/bin/env python3
"""Generate visualization.html embedding both ring_iter 0 schedule CSVs."""
from pathlib import Path

ROOT = Path(__file__).parent
CURRENT_CSV = (ROOT / "mcast-ring-0-ring_iter0_timestamps.csv").read_text()
IDEA_CSV = (ROOT / "idea-ring_iter0_timestamps.csv").read_text()

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ring Iter 0 — Reverse-K Optimization</title>
<style>
  :root {
    --bg: #0b0d12;
    --bg-card: #161922;
    --bg-card-hi: #1d2230;
    --fg: #e6e8ee;
    --fg-dim: #9aa3b2;
    --accent: #6ea8fe;
    --accent-2: #f59e0b;
    --good: #4ade80;
    --bad: #ef4444;
    --border: #262b38;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; }
  header { padding: 28px 32px 16px; border-bottom: 1px solid var(--border); }
  header h1 { margin: 0 0 6px; font-size: 24px; font-weight: 600; }
  header .sub { color: var(--fg-dim); font-size: 14px; }
  main { padding: 24px 32px 64px; max-width: 1700px; }
  section { margin-bottom: 28px; }
  section h2 { font-size: 16px; font-weight: 600; margin: 0 0 8px; color: var(--fg); }
  section .info { color: var(--fg-dim); font-size: 12px; margin-bottom: 10px; }

  .explainer { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px; }
  .card h3 { margin: 0 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--accent); font-weight: 600; }
  .card.idea h3 { color: var(--good); }
  .card p { margin: 0; line-height: 1.55; color: var(--fg); }
  .card code { background: rgba(255,255,255,0.06); padding: 1px 6px; border-radius: 4px; font-size: 12.5px; }

  .stats-row { display: flex; gap: 12px; align-items: stretch; flex-wrap: wrap; }
  .stat { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px; min-width: 180px; flex: 1; }
  .stat .label { color: var(--fg-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
  .stat .num { font-size: 28px; font-weight: 600; line-height: 1.1; }
  .stat .sub { color: var(--fg-dim); font-size: 12px; margin-top: 2px; }
  .stat.current .num { color: #fda4af; }
  .stat.idea .num { color: var(--good); }
  .stat.speedup .num { color: var(--accent); }
  .stat.arrow { display: flex; align-items: center; justify-content: center; flex: 0 0 50px; min-width: 50px; background: transparent; border: none; font-size: 28px; color: var(--fg-dim); }

  .controls { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; padding: 12px 16px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; }
  .controls label { display: flex; gap: 8px; align-items: center; color: var(--fg-dim); font-size: 13px; }
  .controls select, .controls input[type=range] { background: var(--bg-card-hi); color: var(--fg); border: 1px solid var(--border); border-radius: 5px; padding: 4px 8px; font: inherit; font-size: 13px; }
  .controls input[type=range] { padding: 0; width: 130px; }
  .controls .size-display { color: var(--fg); font-variant-numeric: tabular-nums; min-width: 36px; }

  .grid-block { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px; margin-bottom: 16px; }
  .grid-block .title-row { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 10px; flex-wrap: wrap; gap: 12px; }
  .grid-block h2 { margin: 0; }
  .grid-block .meta { color: var(--fg-dim); font-size: 12px; }
  .grid-block .meta b { color: var(--fg); font-weight: 600; }

  .canvas-wrap { position: relative; overflow-x: auto; padding-bottom: 4px; }
  .canvas-stack { position: relative; display: inline-block; }
  canvas.grid { display: block; image-rendering: pixelated; cursor: crosshair; background: #0a0a14; }
  canvas.kv-bar { display: block; image-rendering: pixelated; margin-top: 4px; background: #0a0a14; }
  .axis-label { font-size: 11px; color: var(--fg-dim); margin: 6px 0 0; font-style: italic; }

  .legend { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px; }
  .legend .swatch { display: inline-block; width: 14px; height: 14px; border-radius: 3px; vertical-align: middle; margin-right: 6px; }
  .legend > div { display: flex; align-items: center; font-size: 13px; color: var(--fg-dim); }
  .legend .gradient { display: inline-block; width: 220px; height: 14px; border-radius: 3px; vertical-align: middle; margin: 0 8px; background: linear-gradient(to right, hsl(0,70%,55%), hsl(60,70%,55%), hsl(120,70%,55%), hsl(180,70%,55%), hsl(240,70%,55%), hsl(300,70%,55%), hsl(360,70%,55%)); }

  #tooltip { position: fixed; pointer-events: none; background: #1d2230; border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; font-size: 12px; line-height: 1.5; color: var(--fg); box-shadow: 0 6px 20px rgba(0,0,0,0.5); z-index: 100; opacity: 0; transition: opacity 0.08s; max-width: 260px; }
  #tooltip.show { opacity: 1; }
  #tooltip .ttkey { color: var(--fg-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
  #tooltip .ttval { font-weight: 600; }
  #tooltip .ttbig { font-size: 14px; font-weight: 600; color: var(--accent); margin-bottom: 4px; }

  .footer-note { color: var(--fg-dim); font-size: 12px; line-height: 1.55; padding: 16px; border-left: 3px solid var(--border); margin-top: 24px; }
</style>
</head>
<body>
  <header>
    <h1>Ring Iter 0 — Reverse-K Optimization</h1>
    <div class="sub">Eliminating mcast-barrier discards by interleaving forward (light Q) and reverse (heavy Q) K traversal.</div>
  </header>

  <main>
    <section class="explainer">
      <div class="card current">
        <h3>Current — mcast barrier</h3>
        <p>Each of 6 q_iters runs 20 K mcast steps. At every step, all 100 cores wait for K[k] to be broadcast. Cores outside the causal triangle still pay the full barrier + bandwidth cost, then discard the result. Of 11,600 active cell-cycles, <b>5,510 (47.5%) are discards</b>.</p>
      </div>
      <div class="card idea">
        <h3>Idea — reverse-K</h3>
        <p>Process light Q forward (K0→K19) and heavy Q reverse (K19→K0). With pair-based zigzag, each timestamp needs at most <b>2 distinct K values</b> system-wide. No discards. <b>120 → 63 cycles</b> per core.</p>
      </div>
    </section>

    <section class="stats-row">
      <div class="stat current"><div class="label">Current</div><div class="num">120</div><div class="sub">timestamps · 52.5% useful</div></div>
      <div class="stat arrow"><span>→</span></div>
      <div class="stat idea"><div class="label">Reverse-K</div><div class="num">63</div><div class="sub">timestamps · 100% useful</div></div>
      <div class="stat speedup"><div class="label">Speedup</div><div class="num">1.90×</div><div class="sub">if barrier-bound</div></div>
      <div class="stat"><div class="label">Discards eliminated</div><div class="num">5,510</div><div class="sub">barrier-cycles wasted</div></div>
    </section>

    <section class="controls">
      <label>Color by
        <select id="color-mode">
          <option value="k">K index</option>
          <option value="q">Q index</option>
          <option value="fr">forward / reverse</option>
          <option value="binary">compute vs discard</option>
        </select>
      </label>
      <label>Cell size
        <input type="range" id="cell-size" min="3" max="14" value="10">
        <span class="size-display" id="cell-size-display">10px</span>
      </label>
      <label>Highlight @ idea t=
        <input type="range" id="hl-ts" min="-1" max="62" value="-1">
        <span class="size-display" id="hl-ts-display">off</span>
      </label>
      <label>Cores shown
        <input type="range" id="rows-shown" min="1" max="100" value="10">
        <span class="size-display" id="rows-shown-display">10</span>
      </label>
    </section>

    <section>
      <div class="grid-block">
        <div class="title-row">
          <h2>Current schedule — synchronous K mcast</h2>
          <div class="meta">100 cores × <b>120</b> barrier steps. Columns are <code>q&lt;i&gt;_k&lt;j&gt;</code>. <span style="color:#fda4af">Discards (D)</span> shown as dark gray.</div>
        </div>
        <div class="canvas-wrap">
          <div class="canvas-stack">
            <canvas class="grid" id="canvas-current"></canvas>
            <canvas class="kv-bar" id="kv-current"></canvas>
          </div>
        </div>
        <div class="axis-label">↑ rows = cores (top: core 0). ↓ bottom strip: distinct K-chunks delivered per timestamp (always 1 — single mcast K).</div>
      </div>

      <div class="grid-block">
        <div class="title-row">
          <h2>Reverse-K schedule — interleaved fwd / rev</h2>
          <div class="meta">100 cores × <b>63</b> timestamps. <span style="color:#4ade80">All cells useful</span>. Cores in last row (rows 90–99) finish in 42 timestamps (2 pairs each).</div>
        </div>
        <div class="canvas-wrap">
          <div class="canvas-stack">
            <canvas class="grid" id="canvas-idea"></canvas>
            <canvas class="kv-bar" id="kv-idea"></canvas>
          </div>
        </div>
        <div class="axis-label">↓ bottom strip: distinct K-chunks delivered per timestamp (1 or 2 — fwd and rev).</div>
      </div>
    </section>

    <section class="legend">
      <div><span class="swatch" style="background:#2a2a3a"></span> discard / barrier waste</div>
      <div><span class="swatch" style="background:#0a0a14"></span> empty (idle core)</div>
      <div id="legend-color">K index <span class="gradient"></span> 0 → 19</div>
    </section>

    <div class="footer-note">
      <b>Reading the diagrams.</b> Hover any cell for the full <code>Q_K_V</code> tuple. Switch <i>Color by</i> to see different views — color by <b>K index</b> reveals the diagonal structure of the reverse-K schedule (each core sweeps K downward through reverse halves). Use <b>Highlight K</b> to isolate cells using a specific K-chunk and see its delivery pattern across the system.
    </div>
  </main>

  <div id="tooltip"></div>

  <script type="text/plain" id="current-csv">__CURRENT_CSV__</script>
  <script type="text/plain" id="idea-csv">__IDEA_CSV__</script>

  <script>
    // --- parse CSVs ---
    function parseCSV(text) {
      const lines = text.trim().split('\n');
      const header = lines[0].split(',');
      const rows = lines.slice(1)
        .map(l => l.split(','))
        .filter(r => /^\d+$/.test(r[0]));
      return { header, rows, numCols: header.length - 3, numRows: rows.length };
    }
    const current = parseCSV(document.getElementById('current-csv').textContent);
    const idea    = parseCSV(document.getElementById('idea-csv').textContent);

    function parseCell(s) {
      if (!s || s === '' || s === '-') return null;
      if (s === 'D') return { type: 'D' };
      const m = /^Q(\d+)_K(\d+)_V(\d+)$/.exec(s);
      if (!m) return null;
      return { type: 'C', q: +m[1], k: +m[2], v: +m[3] };
    }

    function precompute(data) {
      const cells = new Array(data.numRows);
      const distinctKs = new Array(data.numCols);
      for (let c = 0; c < data.numCols; c++) distinctKs[c] = new Set();
      let computeCount = 0, discardCount = 0;
      for (let r = 0; r < data.numRows; r++) {
        const row = data.rows[r];
        const arr = new Array(data.numCols);
        for (let c = 0; c < data.numCols; c++) {
          const cell = parseCell(row[c + 3]);
          arr[c] = cell;
          if (cell && cell.type === 'C') { distinctKs[c].add(cell.k); computeCount++; }
          else if (cell && cell.type === 'D') { discardCount++; }
        }
        cells[r] = arr;
      }
      return { cells, distinctKs: distinctKs.map(s => [...s].sort((a,b)=>a-b)), computeCount, discardCount };
    }
    const cur = precompute(current);
    const ide = precompute(idea);

    // Per-core lookup: cell-string -> column index in current schedule.
    // Used to find where each core does the *same* work in current as at a given idea timestamp.
    const currentValueToCol = current.rows.map(row => {
      const m = new Map();
      for (let c = 0; c < current.numCols; c++) {
        const v = row[c + 3];
        if (v && v !== 'D' && v !== '-' && v !== '') m.set(v, c);
      }
      return m;
    });

    // For idea timestamp `ts`, return Map<rowIdx, Set<colIdx>> of cells in current
    // where each core does the same (Q,K,V) work.
    function currentCellsForIdeaTs(ts) {
      if (ts < 0) return null;
      const map = new Map();
      for (let r = 0; r < idea.rows.length; r++) {
        const v = idea.rows[r][ts + 3];
        if (!v || v === 'D' || v === '-') continue;
        const c = currentValueToCol[r].get(v);
        if (c === undefined) continue;
        let s = map.get(r);
        if (!s) { s = new Set(); map.set(r, s); }
        s.add(c);
      }
      return map;
    }

    // --- color logic ---
    function colorFor(cell, mode, dim) {
      if (!cell) return '#0a0a14';
      if (cell.type === 'D') return dim ? '#15151c' : '#2a2a3a';
      if (dim) return '#1c1f29';
      switch (mode) {
        case 'k': {
          const hue = (cell.k / 20) * 360;
          return `hsl(${hue}, 72%, 56%)`;
        }
        case 'q': {
          const hue = (cell.q / 20) * 360;
          return `hsl(${hue}, 65%, 56%)`;
        }
        case 'fr': {
          return cell.q < 10 ? '#3b82f6' : '#f97316';
        }
        case 'binary':
        default:
          return '#3b82f6';
      }
    }

    // --- draw grid ---
    // isHl: null (no highlight) or function(r, c) -> boolean
    function drawGrid(canvas, data, pre, cellSize, mode, isHl, rowsShown) {
      const ctx = canvas.getContext('2d');
      const rows = Math.min(rowsShown, data.numRows);
      const W = data.numCols * cellSize;
      const H = rows * cellSize;
      canvas.width = W; canvas.height = H;
      canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
      ctx.fillStyle = '#0a0a14';
      ctx.fillRect(0, 0, W, H);
      for (let r = 0; r < rows; r++) {
        const row = pre.cells[r];
        for (let c = 0; c < data.numCols; c++) {
          const cell = row[c];
          if (!cell) continue;
          const dim = isHl !== null && !isHl(r, c);
          ctx.fillStyle = colorFor(cell, mode, dim);
          ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        }
      }
    }

    // --- draw KV bar (distinct K values per timestamp, max 2 stacked) ---
    // isHlCol: null (no highlight) or function(c) -> boolean
    function drawKvBar(canvas, data, pre, cellSize, mode, isHlCol, rowsShown) {
      const W = data.numCols * cellSize;
      const ROWS = 2;
      const H = ROWS * cellSize + 4;
      canvas.width = W; canvas.height = H;
      canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#0a0a14';
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#262b38';
      ctx.fillRect(0, 0, W, 2);
      const rows = Math.min(rowsShown, data.numRows);
      for (let c = 0; c < data.numCols; c++) {
        const ks = (rows === data.numRows)
          ? pre.distinctKs[c]
          : (() => {
              const s = new Set();
              for (let r = 0; r < rows; r++) {
                const cell = pre.cells[r][c];
                if (cell && cell.type === 'C') s.add(cell.k);
              }
              return [...s].sort((a, b) => a - b);
            })();
        const dim = isHlCol !== null && !isHlCol(c);
        for (let i = 0; i < ks.length && i < ROWS; i++) {
          const k = ks[i];
          if (dim) {
            ctx.fillStyle = '#1c1f29';
          } else if (mode === 'k' || mode === 'binary' || mode === 'q') {
            const hue = (k / 20) * 360;
            ctx.fillStyle = `hsl(${hue}, 72%, 56%)`;
          } else if (mode === 'fr') {
            ctx.fillStyle = i === 0 ? '#3b82f6' : '#f97316';
          } else {
            ctx.fillStyle = '#3b82f6';
          }
          ctx.fillRect(c * cellSize, 4 + i * cellSize, cellSize, cellSize);
        }
      }
    }

    // --- tooltip ---
    const tooltip = document.getElementById('tooltip');
    function showTip(e, html) {
      tooltip.innerHTML = html;
      tooltip.classList.add('show');
      const px = e.clientX + 14;
      const py = e.clientY + 14;
      tooltip.style.left = Math.min(px, window.innerWidth - 280) + 'px';
      tooltip.style.top = Math.min(py, window.innerHeight - 100) + 'px';
    }
    function hideTip() { tooltip.classList.remove('show'); }

    function bindCellHover(canvas, data, pre, getCellSize, getRowsShown, label) {
      canvas.addEventListener('mousemove', (e) => {
        const cs = getCellSize();
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const c = Math.floor(x / cs);
        const r = Math.floor(y / cs);
        const maxR = Math.min(getRowsShown(), data.numRows);
        if (r < 0 || r >= maxR || c < 0 || c >= data.numCols) { hideTip(); return; }
        const cell = pre.cells[r][c];
        const row = data.rows[r];
        const coreLabel = `Core ${row[0]} (x=${row[1]}, y=${row[2]})`;
        const tsLabel = `${label === 'current' ? data.header[c+3] : 't=' + c}`;
        let body;
        if (!cell) body = `<div class="ttbig">empty</div>`;
        else if (cell.type === 'D') body = `<div class="ttbig" style="color:#fda4af">DISCARD</div><div class="ttkey">barrier wait, no compute</div>`;
        else body = `<div class="ttbig" style="color:#4ade80">Q${cell.q} · K${cell.k} · V${cell.v}</div><div class="ttkey">${cell.q < 10 ? 'forward (light Q)' : 'reverse (heavy Q)'}</div>`;
        showTip(e, `<div class="ttkey">${coreLabel} @ ${tsLabel}</div>${body}`);
      });
      canvas.addEventListener('mouseleave', hideTip);
    }
    function bindKvHover(canvas, data, pre, getCellSize, getRowsShown, label) {
      canvas.addEventListener('mousemove', (e) => {
        const cs = getCellSize();
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const c = Math.floor(x / cs);
        if (c < 0 || c >= data.numCols) { hideTip(); return; }
        const rows = Math.min(getRowsShown(), data.numRows);
        const ksSet = new Set();
        let cmp = 0, dsc = 0;
        for (let r = 0; r < rows; r++) {
          const cell = pre.cells[r][c];
          if (cell && cell.type === 'C') { ksSet.add(cell.k); cmp++; }
          else if (cell && cell.type === 'D') dsc++;
        }
        const ks = [...ksSet].sort((a, b) => a - b);
        const tsLabel = label === 'current' ? data.header[c+3] : 't=' + c;
        showTip(e, `<div class="ttkey">${tsLabel}</div><div class="ttbig">${ks.length} distinct K · {${ks.join(', ')}}</div><div class="ttkey">visible cores: ${cmp} compute, ${dsc} discard</div>`);
      });
      canvas.addEventListener('mouseleave', hideTip);
    }

    // --- render orchestration ---
    const els = {
      cur: document.getElementById('canvas-current'),
      ide: document.getElementById('canvas-idea'),
      kvCur: document.getElementById('kv-current'),
      kvIde: document.getElementById('kv-idea'),
      mode: document.getElementById('color-mode'),
      size: document.getElementById('cell-size'),
      sizeDisp: document.getElementById('cell-size-display'),
      hlTs: document.getElementById('hl-ts'),
      hlTsDisp: document.getElementById('hl-ts-display'),
      rows: document.getElementById('rows-shown'),
      rowsDisp: document.getElementById('rows-shown-display'),
      legend: document.getElementById('legend-color'),
    };

    function getMode() { return els.mode.value; }
    function getSize() { return parseInt(els.size.value, 10); }
    function getHlTs() { return parseInt(els.hlTs.value, 10); }
    function getHlKs() {
      const ts = getHlTs();
      if (ts < 0) return null;
      const ks = ide.distinctKs[ts] || [];
      return new Set(ks);
    }
    function getRows() { return parseInt(els.rows.value, 10); }

    function updateLegend() {
      const m = getMode();
      let inner;
      if (m === 'k')      inner = 'K index <span class="gradient"></span> 0 → 19';
      else if (m === 'q') inner = 'Q index <span class="gradient"></span> 0 → 19';
      else if (m === 'fr') inner = '<span class="swatch" style="background:#3b82f6"></span> forward (light Q 0–9) &nbsp; <span class="swatch" style="background:#f97316"></span> reverse (heavy Q 10–19)';
      else inner = '<span class="swatch" style="background:#3b82f6"></span> compute';
      els.legend.innerHTML = inner;
    }

    function renderAll() {
      const cs = getSize();
      const m = getMode();
      const ts = getHlTs();
      const hlKs = getHlKs();
      const rs = getRows();
      els.sizeDisp.textContent = cs + 'px';
      els.rowsDisp.textContent = rs;
      if (ts < 0) {
        els.hlTsDisp.textContent = 'off';
      } else {
        const ks = [...(hlKs || [])].sort((a,b)=>a-b);
        els.hlTsDisp.textContent = `t=${ts} · {${ks.map(k=>'K'+k).join(', ')}}`;
      }

      // idea: highlight = single column ts
      const ideaCellHl = ts < 0 ? null : (r, c) => c === ts;
      const ideaColHl  = ts < 0 ? null : (c) => c === ts;

      // current: highlight = the cells where each core does the *same work* as at idea ts
      const curMap = currentCellsForIdeaTs(ts);
      const visibleRows = Math.min(rs, current.numRows);
      const curBarCols = (() => {
        if (!curMap) return null;
        const s = new Set();
        for (let r = 0; r < visibleRows; r++) {
          const set = curMap.get(r);
          if (set) for (const c of set) s.add(c);
        }
        return s;
      })();
      const curCellHl = !curMap ? null : (r, c) => {
        const s = curMap.get(r);
        return !!s && s.has(c);
      };
      const curColHl  = !curBarCols ? null : (c) => curBarCols.has(c);

      drawGrid(els.cur, current, cur, cs, m, curCellHl,  rs);
      drawGrid(els.ide, idea,    ide, cs, m, ideaCellHl, rs);
      drawKvBar(els.kvCur, current, cur, cs, m, curColHl,  rs);
      drawKvBar(els.kvIde, idea,    ide, cs, m, ideaColHl, rs);
      updateLegend();
    }

    bindCellHover(els.cur, current, cur, getSize, getRows, 'current');
    bindCellHover(els.ide, idea,    ide, getSize, getRows, 'idea');
    bindKvHover(els.kvCur, current, cur, getSize, getRows, 'current');
    bindKvHover(els.kvIde, idea,    ide, getSize, getRows, 'idea');

    els.mode.addEventListener('change', renderAll);
    els.size.addEventListener('input', renderAll);
    els.hlTs.addEventListener('input', renderAll);
    els.rows.addEventListener('input', renderAll);

    renderAll();
  </script>
</body>
</html>
"""

html = TEMPLATE.replace("__CURRENT_CSV__", CURRENT_CSV).replace("__IDEA_CSV__", IDEA_CSV)
out = ROOT / "visualization.html"
out.write_text(html)
print(f"wrote {out} ({len(html):,} bytes)")
