#!/usr/bin/env node
// Print the README findings tables (scenario summary + per-feature "G / LOO" with tier) from study.json.
'use strict';
const R = JSON.parse(require('fs').readFileSync(process.argv[2] || require('../lib/paths.js').STUDY));
const { FEATURES } = require('../study.js');
const k = (x) => (x >= 1e5 ? (x / 1000).toFixed(0) : (x / 1000).toFixed(1)) + 'k';
const SC = Object.keys(R.scenarios);
console.log('| scenario | today | greedy full stack | best grid config | best config with ∞ cache |\n|---|---|---|---|---|');
for (const s of SC) {
  const S = R.scenarios[s], g = S.grid[0], e = g.extra;
  const inf = (S.sens || []).find((x) => x.name === 'inf cache');
  const lanes = e.arenaTokens ? `${e.arenaTokens / 1e6}M arena` : `${e.lanes} lanes`;
  console.log(`| ${S.label} | ${k(S.greedy[0].goodput)} | ${k(S.full.goodput)} | **${k(g.goodput)}** (${e.stages}×[${e.mesh}]${e.replicas > 1 ? '×' + e.replicas : ''}, ${lanes}, ${e.budget ? 'budget ' + e.budget / 1024 + 'k' : 'chunk ' + e.chunk}) | ${inf ? k(inf.goodput) : '–'} |`);
}
// order and tier from the reference scenario (8 galaxies, today's kernels), same rule as the artifact
const RANK = R.scenarios.g8_k0 ? 'g8_k0' : SC[0];
const rows = FEATURES.map((f) => {
  let best = 1;
  const cells = SC.map((s) => {
    const S = R.scenarios[s];
    const gi = S.greedy.findIndex((g) => g.add === f.key);
    const G = gi > 0 ? S.greedy[gi].goodput / S.greedy[gi - 1].goodput : null;
    const l = S.loo.find((x) => x.remove === f.key);
    const L = l && l.goodput > 0 ? S.full.goodput / l.goodput : null;
    if (s === RANK) best = Math.max(best, G || 1, L || 1);
    return `×${G ? G.toFixed(2) : '–'} #${gi} / ×${L ? L.toFixed(2) : '–'}`;
  });
  return { tier: best >= 1.25 ? 'P0' : best >= 1.07 ? 'P1' : 'P2', best, line: `| ${f.name} | ${cells.join(' | ')} |` };
}).sort((a, b) => b.best - a.best);
console.log('\n| tier | feature | ' + SC.map((s) => R.scenarios[s].label).join(' | ') + ' |\n|---|---|' + SC.map(() => '---').join('|') + '|');
for (const r of rows) console.log(`| ${r.tier} ${r.line.slice(1)}`);
