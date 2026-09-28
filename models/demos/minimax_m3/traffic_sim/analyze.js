#!/usr/bin/env node
// Print the study results: greedy roadmap, leave-one-out, grid top-N, per scenario.
// Usage: node analyze.js [results/study.json] [--points]
'use strict';
const fs = require('fs');
const f = process.argv[2] && !process.argv[2].startsWith('--') ? process.argv[2] : require('./lib/paths.js').STUDY;
const R = JSON.parse(fs.readFileSync(f));
const k = (x) => (x == null ? '    -' : (x / 1000).toFixed(1).padStart(5) + 'k');
const pc = (x) => (x == null ? '  - ' : (100 * x).toFixed(0).padStart(3) + '%');
const at = (a) => (a ? `C=${String(a.conc).padStart(4)} p50 ${a.ttftP50.toFixed(1).padStart(5)}s p90 ${a.ttftP90.toFixed(1).padStart(5)}s hit ${pc(a.hitRate)}/${pc(a.infHitRate)} repf ${pc(a.reprefillFrac)} pad ${pc(a.padFrac)} util ${pc(a.maxUtil)} chunk ${Math.round(a.avgChunkTok)} segs ${a.avgSegsPerChunk.toFixed(1)}` : '(no point meets SLO)');
for (const [key, S] of Object.entries(R.scenarios)) {
  console.log(`\n=== ${key}: ${S.label}   (goodput = useful tok/s at p90 TTFT <= ${R.slo}s)`);
  console.log('greedy roadmap:');
  let prev = null;
  for (const g of S.greedy) {
    const gain = prev ? `x${(g.goodput / Math.max(1, prev)).toFixed(2)}` : '     ';
    const inf = g.inf ? ` | inf-cache ${k(g.inf.goodput)}` : '';
    console.log(`  ${(g.add || 'BASE').padEnd(10)} ${k(g.goodput)} ${gain.padStart(6)} | ${at(g.at)}${inf}`);
    prev = g.goodput;
  }
  if (S.full) console.log(`  FULL       ${k(S.full.goodput)}        | ${at(S.full.at)}`);
  console.log('leave-one-out (full stack minus feature):');
  for (const l of S.loo) console.log(`  -${l.remove.padEnd(10)} ${k(l.goodput)} x${l.loss.toFixed(2)}`);
  console.log('grid top 5:');
  for (const g of S.grid.slice(0, 5)) console.log(`  ${k(g.goodput)} ${JSON.stringify(g.extra)} split ${g.plan.counts.join(',')} | ${at(g.at)}`);
  if (process.argv.includes('--points')) {
    const b = S.grid[0];
    console.log('best grid config curve:');
    for (const p of b.points) console.log(`   ${at(p)} useful ${k(p.usefulTps)} proc ${k(p.processedTps)}`);
  }
}
