// Stubbed-DOM headless validation harness for the sparse-MLA perf report.
const fs = require('fs');
const DIR = __dirname;
const payload = fs.readFileSync(DIR + '/_payload.json', 'utf8');
let appjs = fs.readFileSync(DIR + '/_app.js', 'utf8');

const RENDER_LOG = [];   // every innerHTML / textContent assignment, for NaN/undefined scanning
let ELCOUNT = 0;

function mkEl(tag) {
  const el = {
    tagName: (tag || 'div').toUpperCase(),
    _id: '_el' + (ELCOUNT++),
    attrs: {}, children: [], style: {}, dataset: {}, _text: '', _html: '',
    disabled: false, title: '', onclick: null, parentNode: null,
    clientWidth: 1000, clientHeight: 600,
    classList: (() => { const s = new Set(); return {
      add: (...c) => c.forEach(x => s.add(x)), remove: (...c) => c.forEach(x => s.delete(x)),
      toggle: (c) => (s.has(c) ? (s.delete(c), false) : (s.add(c), true)), contains: (c) => s.has(c), _set: s }; })(),
    setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'class') this._class = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    removeAttribute(k) { delete this.attrs[k]; },
    appendChild(c) { c.parentNode = this; this.children.push(c); return c; },
    removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); return c; },
    remove() { if (this.parentNode) this.parentNode.removeChild(this); },
    querySelector(sel) { this._qs = this._qs || {}; if (!this._qs[sel]) this._qs[sel] = mkEl('span'); return this._qs[sel]; },
    querySelectorAll() { return []; },
    closest() { return null; },
    addEventListener() {}, removeEventListener() {},
    getBoundingClientRect() { return { left: 0, top: 0, right: 1000, bottom: 600, width: 1000, height: 600 }; },
    getContext() { return null; },
    focus() {}, blur() {}, contains() { return false; },
  };
  Object.defineProperty(el, 'innerHTML', {
    get() { return this._html; },
    set(v) { this._html = String(v); this.children = []; RENDER_LOG.push('[innerHTML ' + (this.attrs.id || this.tagName) + '] ' + v); },
  });
  Object.defineProperty(el, 'textContent', {
    get() { return this._text; },
    set(v) { this._text = String(v); RENDER_LOG.push('[text] ' + v); },
  });
  Object.defineProperty(el, 'className', {
    get() { return this._class || ''; }, set(v) { this._class = String(v); this.attrs.class = String(v); },
  });
  return el;
}

const idCache = {};
const qsaCache = {};
function seg(vals, key) { return vals.map(v => { const b = mkEl('button'); b.dataset[key] = v; return b; }); }

const document = {
  getElementById(id) {
    if (!idCache[id]) { const e = mkEl('div'); e.attrs.id = id; idCache[id] = e; }
    return idCache[id];
  },
  querySelectorAll(sel) {
    if (qsaCache[sel]) return qsaCache[sel];
    let r = [];
    if (sel === '#segVariant button') r = seg(['deepseek_v32', 'glm_5_1'], 'vr');
    else if (sel === '#segScenario button') r = seg(['warm', 'cold', 'long'], 's');
    else if (sel === '#segMode button') r = seg(['sparse', 'dense'], 'm');
    else if (sel === '#segView button') r = seg(['semantic', 'ops'], 'v');
    else if (sel === '.gzoom button') r = seg(['out', 'in', 'reset'], 'z');
    else r = []; // #opTable th[data-sort] etc.
    qsaCache[sel] = r; return r;
  },
  querySelector() { return null; },
  createElementNS(_ns, t) { return mkEl(t); },
  createElement(t) { return mkEl(t); },
  documentElement: mkEl('html'),
  addEventListener() {}, removeEventListener() {},
};
document.getElementById('payload')._text = payload; // seed the payload BEFORE eval

globalThis.document = document;
globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.matchMedia = () => ({ matches: false, addEventListener() {}, addListener() {} });
globalThis.requestAnimationFrame = (f) => { try { f(0); } catch (e) {} return 0; };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '' });

// expose inner functions for firing controls
appjs += '\n;globalThis.__api={draw,drawGraph,openDrawer,syncPressed,expanded};\n';

const vm = require('vm');
const fails = [];
function check(name, fn) { try { fn(); console.log('  ok  ' + name); } catch (e) { fails.push(name + ': ' + (e && e.stack || e)); console.log('  FAIL ' + name + ' :: ' + (e && e.message || e)); } }

// 1. load (runs drawStatic(); draw(); setupPanZoom())
check('load (drawStatic + draw + setupPanZoom)', () => vm.runInThisContext(appjs));
const api = globalThis.__api || {};

// helpers
function fire(sel, pred) { document.querySelectorAll(sel).forEach(b => { if (!pred || pred(b)) { if (typeof b.onclick === 'function') b.onclick({ stopPropagation() {}, target: {} }); } }); }
function svgClassCounts() {
  const svg = document.getElementById('svgGraph'); let node = 0, onode = 0, edge = 0;
  (function walk(el) { for (const c of el.children) { const cl = c.attrs.class || ''; if (/\bonode\b/.test(cl)) onode++; else if (/\bnode\b/.test(cl)) node++; if (/\bedge\b/.test(cl)) edge++; walk(c); } })(svg);
  return { node, onode, edge };
}

// 2–4. per VARIANT: fire the variant toggle, then every scenario × mode + view/expand/drawer + structural
const P = JSON.parse(payload);
for (const vr of ['deepseek_v32', 'glm_5_1']) {
  check('variant=' + vr, () => fire('#segVariant button', b => b.dataset.vr === vr));
  for (const m of ['sparse', 'dense']) {
    check(`[${vr}] mode=${m}`, () => fire('#segMode button', b => b.dataset.m === m));
    for (const s of ['warm', 'cold', 'long']) {
      check(`[${vr}]   scenario=${s} (${m})`, () => fire('#segScenario button', b => b.dataset.s === s));
    }
  }
  check(`[${vr}] view=ops`, () => fire('#segView button', b => b.dataset.v === 'ops'));
  check(`[${vr}] view=semantic`, () => fire('#segView button', b => b.dataset.v === 'semantic'));
  // sparse/warm/semantic before per-node interactions (drawers open only for the current graph)
  fire('#segMode button', b => b.dataset.m === 'sparse');
  fire('#segScenario button', b => b.dataset.s === 'warm');
  fire('#segView button', b => b.dataset.v === 'semantic');
  check(`[${vr}] expand s3`, () => { api.expanded && api.expanded.add('s3'); api.drawGraph(); });
  check(`[${vr}] openDrawer(s3/s8)`, () => { api.openDrawer('s3'); api.openDrawer('s8'); });
  check(`[${vr}] drawer body has file:line`, () => { api.openDrawer('s3'); const b = document.getElementById('drBody'); if (!/file|indexer\.py|:\d/.test(b._html)) throw new Error('drawer body missing file:line'); });
  api.expanded && api.expanded.clear();
  // structural: semantic .node>0/.onode==0; ops .onode>0
  fire('#segView button', b => b.dataset.v === 'semantic'); api.drawGraph();
  const sem = svgClassCounts();
  check(`[${vr}] semantic .node>0 & .onode==0`, () => { if (!(sem.node > 0 && sem.onode === 0)) throw new Error(JSON.stringify(sem)); });
  fire('#segView button', b => b.dataset.v === 'ops'); api.drawGraph();
  const ops = svgClassCounts();
  check(`[${vr}] ops .onode>0`, () => { if (!(ops.onode > 0)) throw new Error(JSON.stringify(ops)); });
  console.log(`  [counts ${vr}] semantic ${JSON.stringify(sem)} ops ${JSON.stringify(ops)}`);
  fire('#segView button', b => b.dataset.v === 'semantic');
}
check('gzoom buttons', () => fire('.gzoom button'));

// 5. block sums == scenario total (from payload), all variants × 6 combos
for (const vr of Object.keys(P.variants)) {
  const bt0 = P.variants[vr].data.block_timing;
  for (const m of ['sparse', 'dense']) for (const s of ['warm', 'cold', 'long']) {
    check(`block sum == total ${vr}/${m}/${s}`, () => {
      const bt = bt0[m][s]; const sum = Object.values(bt.nodes).reduce((a, b) => a + b, 0);
      if (Math.round(sum) !== Math.round(bt.total_ns)) throw new Error(`sum ${sum} != total ${bt.total_ns}`);
    });
  }
}

// 6. no NaN / undefined in any rendered output
check('no NaN/undefined in rendered HTML/text', () => {
  const bad = RENDER_LOG.filter(x => /NaN|undefined\s*ms|>\s*undefined|undefinedms|\bundefined\b/.test(x));
  if (bad.length) throw new Error(bad.length + ' offending renders, first: ' + bad[0].slice(0, 200));
});

console.log('\n' + (fails.length ? `FAILURES: ${fails.length}` : 'ALL CHECKS PASSED'));
if (fails.length) { fails.forEach(f => console.log(' - ' + f.split('\n')[0])); process.exit(1); }
