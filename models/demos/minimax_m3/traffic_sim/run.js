#!/usr/bin/env node
// Run one simulation (or a concurrency sweep) from the command line.
// Usage: node run.js [--data DIR] [--preset NAME] [--set key=value ...] [--conc 32,64,128] [--json]
//   values are JSON-parsed when possible (e.g. --set mesh=[4,4] --set batch=true --set split=[1,1,1,5,...])
'use strict';
const fs = require('fs'), path = require('path');
const SIM = require('./sim_core.js');
const PRESETS = require('./presets.js');

function loadAll(dataDir) {
  const header = JSON.parse(fs.readFileSync(path.join(dataDir, 'traffic.json')));
  const b = fs.readFileSync(path.join(dataDir, 'traffic.bin'));
  const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  const TR = SIM.loadTraffic(buf, header);
  const cal = SIM.calibrate(JSON.parse(fs.readFileSync(path.join(__dirname, 'calib_data.json'))));
  return { TR, cal };
}

function parseArgs(argv) {
  const a = { data: require('./lib/paths.js').DATA, set: {}, conc: null, preset: null, json: false };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i];
    if (k === '--data') a.data = argv[++i];
    else if (k === '--preset') a.preset = argv[++i];
    else if (k === '--study-preset') a.studyPreset = argv[++i]; // e.g. g4_k0 -> best grid config of that scenario
    else if (k === '--conc') a.conc = argv[++i].split(',').map(Number);
    else if (k === '--json') a.json = true;
    else if (k === '--set') { const [key, ...v] = argv[++i].split('='); const s = v.join('='); try { a.set[key] = JSON.parse(s); } catch (e) { a.set[key] = s; } }
  }
  return a;
}

function fmt(r) {
  if (r.error) return `ERROR ${r.error}`;
  const k = (x) => (x / 1000).toFixed(1) + 'k';
  return `C=${String(r.cfg.concurrency).padStart(4)} useful ${k(r.usefulTps).padStart(7)} tok/s | proc ${k(r.processedTps).padStart(7)} | TTFT p50 ${r.ttftP50.toFixed(2)}s p90 ${r.ttftP90.toFixed(2)}s | hit ${(100 * r.hitRate).toFixed(1)}% (inf ${(100 * r.infHitRate).toFixed(1)}%) | reprefill ${(100 * r.reprefillFrac).toFixed(1)}% pad ${(100 * r.padFrac).toFixed(1)}% | util ${(100 * r.maxUtil).toFixed(0)}% | chunk ${r.avgChunkTok.toFixed(0)} segs ${r.avgSegsPerChunk.toFixed(2)} | done ${r.done} warm ${r.warmupS.toFixed(0)}s ev ${r.events}`;
}

if (require.main === module) {
  const a = parseArgs(process.argv);
  const { TR, cal } = loadAll(a.data);
  let sp = {};
  if (a.studyPreset) {
    const { withFeatures } = require('./study.js');
    const R = JSON.parse(fs.readFileSync(require('./lib/paths.js').STUDY)).scenarios[a.studyPreset];
    sp = Object.assign(withFeatures(R.base, R.bestKeys), R.grid[0].extra);
  }
  const base = Object.assign({}, a.preset ? PRESETS.byName(a.preset).cfg : {}, sp, a.set);
  const concs = a.conc || [base.concurrency || SIM.DEFAULTS.concurrency];
  const plan = SIM.planSummary(SIM.makePlan(base, cal));
  if (!a.json) console.log('plan', JSON.stringify(plan));
  const out = [];
  for (const c of concs) {
    const t0 = Date.now();
    const r = SIM.simulate(TR, cal, Object.assign({}, base, { concurrency: c }));
    r.wallMs = Date.now() - t0;
    out.push(r);
    if (!a.json) console.log(fmt(r) + ` | ${r.wallMs} ms`);
  }
  if (a.json) console.log(JSON.stringify(out.map((r) => { const x = Object.assign({}, r); delete x.cfg; return x; })));
}
module.exports = { loadAll, fmt };
