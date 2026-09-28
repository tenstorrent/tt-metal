#!/usr/bin/env node
// Traffic statistics of the preprocessed corpus: end-to-start delays, api_time, per-request new tokens.
'use strict';
const { loadAll } = require('../run.js');
const { TR } = loadAll(process.argv[2] || require('../lib/paths.js').DATA);
const q = (a, p) => a[Math.min(a.length - 1, Math.floor(p * (a.length - 1)))];
const show = (name, a) => { a.sort((x, y) => x - y); const m = a.reduce((x, y) => x + y, 0) / a.length; console.log(`${name.padEnd(22)} n=${a.length} mean=${m.toFixed(1)} p10=${q(a, .1).toFixed(1)} p50=${q(a, .5).toFixed(1)} p90=${q(a, .9).toFixed(1)} p99=${q(a, .99).toFixed(1)} max=${a[a.length - 1].toFixed(0)}`); };
const dMain = [], dSub = [], api = [], out = [], cyc = [];
for (let r = 0; r < TR.nR; r++) {
  const s = TR.req_stream[r]; const first = TR.st_first[s];
  api.push(TR.req_api[r]); out.push(TR.req_out[r]);
  if (r === first) continue;
  (TR.st_kind[s] === 0 ? dMain : dSub).push(TR.req_delay[r]);
}
show('delay main (s)', dMain); show('delay sub (s)', dSub); show('api_time (s)', api); show('out tokens', out);
// main-agent tree duration and requests/hour per trace
let reqs = 0, dur = 0; for (let t = 0; t < TR.nT; t++) { reqs += TR.trReqs[t]; dur += TR.tr_dur[t]; }
console.log(`traces ${TR.nT}: total recorded duration ${(dur / 3600).toFixed(0)} h, ${reqs} requests -> ${(reqs / (dur / 3600)).toFixed(1)} req per trace-hour`);
const capped = (cap) => { let s = 0; for (const x of dMain.concat(dSub)) s += Math.min(x, cap); return s / 3600; };
for (const cap of [10, 60, 300, Infinity]) console.log(`sum of delays capped at ${cap}s: ${capped(cap).toFixed(0)} h`);
