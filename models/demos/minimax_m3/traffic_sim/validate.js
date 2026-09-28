#!/usr/bin/env node
// Calibration report: fitted efficiencies, per-rank stage times vs the 16-stage runs, matrix cells vs #57827 tables.
// Usage: node validate.js [--json out.json]
'use strict';
const fs = require('fs'), path = require('path');
const SIM = require('./sim_core.js');
const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'calib_data.json')));
const cal = SIM.calibrate(data);
const f2 = (x) => (x == null || isNaN(x) ? '   -  ' : x.toFixed(2).padStart(6));
const k = (x) => (x == null ? '   -  ' : (x / 1000).toFixed(1).padStart(5) + 'k');

console.log('== measured efficiencies (fraction of roofline) per mesh');
for (const m in cal.effs) {
  console.log(` [${m}] moe:  ` + Object.entries(cal.effs[m].moe).map(([o, v]) => `${o}=${v.toFixed(3)}`).join(' '));
  console.log(` [${m}] dense:` + Object.entries(cal.effs[m].dense).map(([o, v]) => `${o}=${v.toFixed(3)}`).join(' '));
}
console.log('== pipeline fit', JSON.stringify(cal.pipe), JSON.stringify(cal.fit));

// zone totals: model vs measured per mesh at the profile point
console.log('== zone layer totals (T=5120, K=51200): model (opEff=0, no pipeline mult) vs measured');
for (const m in data.zones) {
  const [sp, tp] = m.split('x').map(Number);
  const c = { sp, tp, P: sp * tp, T: 5120, idxB: 2, imb: 1.2, varLayout: false, bounded: false, msaLocal: false };
  const seg = [{ n: 5120, k: 51200, cap: 56320 }];
  const lat = { ccl: 0.04, op: 0.01 };
  const tm = SIM.layerMs('moe', c, seg, cal.effs[m].moe, lat, 'seq');
  const td = SIM.layerMs('dense', c, seg, cal.effs[m].dense, lat, 'seq');
  console.log(`  ${m}: moe ${tm.toFixed(2)} vs ${data.zones[m]['layer03_sparse'].mean}, dense ${td.toFixed(2)} vs ${data.zones[m]['layer00_dense'].mean}`);
}

console.log('== per-rank stage ms: model / measured  (16 x [2,4])');
const out = { ranks: {}, cells: {} };
for (const run of ['A', 'B', 'C']) {
  const r = data.pipeline[run]; const counts = r.layers.split(',').map(Number); const T = r.chunk;
  const plan = SIM.makePlan({ chunk: T, split: counts, stages: 16, cache: 'inf' }, cal);
  for (const row of r.rows) {
    const o = new Float64Array(16);
    SIM.chunkStageMs(plan, T, [{ n: T, k: Math.floor(row.cached / T) * T, cap: row.cached + 51200 }], o);
    const pick = [0, 1, 3, 8, 15];
    console.log(`  ${run} c=${String(row.cached).padStart(6)}  ` + pick.map((s) => `r${s} ${o[s].toFixed(1)}/${row.rank_ms[s].toFixed(1)}`).join('  '));
    out.ranks[`${run}_${row.cached}`] = { model: Array.from(o), meas: row.rank_ms };
  }
}

console.log('== matrix cells: loaded steady NEW tok/s and idle TTFT ms, model vs measured');
let errs = [];
for (const run of ['A', 'B', 'C']) {
  const r = data.pipeline[run]; const counts = r.layers.split(',').map(Number); const T = r.chunk;
  console.log(` run ${run} (chunk ${T}, split ${r.layers})`);
  console.log('   cached     new | loaded new tok/s model/meas | idle TTFT ms model/meas');
  for (const row of data.tables[run]) {
    const m = SIM.matrixCell(cal, { chunk: T, split: counts, stages: 16 }, row.cached, row.new, 4, 30);
    const e1 = row.loaded_steady_new_tps ? m.newTps / row.loaded_steady_new_tps - 1 : null;
    const e2 = row.idle_ttft_ms_median ? m.idleTtftMs / row.idle_ttft_ms_median - 1 : null;
    if (e1 != null) errs.push(Math.abs(e1)); if (e2 != null) errs.push(Math.abs(e2));
    console.log(`   ${String(row.cached).padStart(6)} ${String(row.new).padStart(6)} | ${k(m.newTps)} / ${k(row.loaded_steady_new_tps)} (${e1 == null ? '-' : (100 * e1).toFixed(0) + '%'}) | ${m.idleTtftMs.toFixed(0).padStart(5)} / ${row.idle_ttft_ms_median == null ? '-' : row.idle_ttft_ms_median.toFixed(0)} (${e2 == null ? '-' : (100 * e2).toFixed(0) + '%'})`);
    out.cells[`${run}_${row.cached}_${row.new}`] = { model: m, meas: row };
  }
}
errs.sort((a, b) => a - b);
console.log(`== |error| median ${(100 * errs[Math.floor(errs.length / 2)]).toFixed(1)}%  p90 ${(100 * errs[Math.floor(0.9 * errs.length)]).toFixed(1)}%  max ${(100 * errs[errs.length - 1]).toFixed(1)}%`);
const j = process.argv.indexOf('--json');
if (j > 0) fs.writeFileSync(process.argv[j + 1], JSON.stringify({ cal, ...out }, null, 1));
