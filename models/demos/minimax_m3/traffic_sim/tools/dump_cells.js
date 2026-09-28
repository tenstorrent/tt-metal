#!/usr/bin/env node
// Debug helper: print per-cell, per-position stage medians of the 16-stage runs (calib_data.json).
// Usage: node tools/dump_cells.js [run] [ranks comma list]
'use strict';
const d = require('../calib_data.json');
const run = process.argv[2] || 'B';
const ranks = (process.argv[3] || '0,1,3,8,15').split(',').map(Number);
for (const c of d.pipeline[run].cells) {
  const pos = [0, Math.floor((c.n_ch - 1) / 2), c.n_ch - 1].filter((v, i, a) => a.indexOf(v) === i);
  console.log(`${run} cached=${String(c.cached).padStart(6)} new=${String(c.new).padStart(5)} nch=${String(c.n_ch).padStart(2)} period=${(c.period_ms || 0).toFixed(1).padStart(6)} | ` +
    ranks.map((r) => `r${r}:` + pos.map((p) => (c.pos_ms[r][p] == null ? '-' : c.pos_ms[r][p].toFixed(1))).join('/')).join('  '));
}
