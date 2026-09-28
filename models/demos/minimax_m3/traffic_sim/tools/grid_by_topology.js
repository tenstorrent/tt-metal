#!/usr/bin/env node
// Best goodput per topology (stages x mesh x replicas) in each scenario's grid.
'use strict';
const R = JSON.parse(require('fs').readFileSync(process.argv[2] || require('../lib/paths.js').STUDY));
for (const [key, S] of Object.entries(R.scenarios)) {
  const best = new Map();
  for (const g of S.grid) {
    const t = `${g.extra.stages}x[${g.extra.mesh}]x${g.extra.replicas}`;
    if (!best.has(t) || g.goodput > best.get(t)) best.set(t, g.goodput);
  }
  console.log(key, [...best.entries()].sort((a, b) => b[1] - a[1]).map(([t, v]) => `${t} ${(v / 1000).toFixed(1)}k`).join(' | '));
}
