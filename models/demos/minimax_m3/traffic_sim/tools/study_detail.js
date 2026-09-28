#!/usr/bin/env node
// Detail view of study.json: candidate gains at every greedy step, and the sensitivity table.
'use strict';
const R = JSON.parse(require('fs').readFileSync(process.argv[2] || require('../lib/paths.js').STUDY));
const k = (x) => (x / 1000).toFixed(1) + 'k';
for (const [key, S] of Object.entries(R.scenarios)) {
  console.log(`\n=== ${key} ${S.label}`);
  S.greedy.forEach((g, i) => {
    if (!g.candidates) return;
    const prev = S.greedy[i - 1].goodput;
    console.log(`  step ${i} (from ${k(prev)}): ` + g.candidates.map((c) => `${c.key} x${(c.goodput / Math.max(1, prev)).toFixed(2)}`).join('  '));
  });
  console.log('  sensitivity (best grid config ' + JSON.stringify(S.grid[0].extra) + ' = ' + k(S.grid[0].goodput) + '):');
  for (const s of S.sens) console.log(`    ${s.name.padEnd(18)} goodput ${k(s.goodput).padStart(7)}  peak ${k(s.peak).padStart(7)}`);
}
