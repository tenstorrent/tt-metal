#!/usr/bin/env node
// The feature/hyperparameter study behind the artifact's tables and presets.
//   greedy roadmap : from today's config, repeatedly add the feature with the largest goodput gain
//   leave-one-out  : from the full stack, remove one feature at a time
//   grid           : topology x batching budget x lane arena on the best stack -> presets
// Scenarios: {4, 8} galaxies x {today's kernels (opEff 0), roofline kernels (opEff 1)}.
// Usage: JOB=<slurm job> ./on_node.sh node study.js [--workers 36] [--out results/study.json] [--quick]
'use strict';
const fs = require('fs'), path = require('path');
const { Pool, summarize } = require('./lib/pool.js');

const SLO = 10; // p90 TTFT seconds for goodput
const CONCS = [8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 576, 640, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048, 2560, 3072, 4096];

const FEATURES = [
  { key: 'bounded', name: 'Bounded dense gather ([0,kv_len) instead of lane capacity)', cfg: { boundedDense: true } },
  { key: 'unaligned', name: 'Unaligned (block-granular) resume', cfg: { unaligned: true } },
  { key: 'pool', name: 'Slot lanes + paged KV pool (4 x 1M lanes per stage, copy-in/out)', cfg: { cache: 'pool', lanes: 4, laneArena: false, laneScope: 'stage' } },
  { key: 'arena', name: 'Variable-size lanes (contiguous arena, 4M tokens)', cfg: { laneArena: true, arenaTokens: 4e6 }, requires: ['pool'] },
  { key: 'host', name: 'Host-DRAM KV tier (1 TB/galaxy, 64 GB/s PCIe)', cfg: { hostTier: true }, requires: ['pool'] },
  { key: 'idxdedup', name: 'index_k cache not replicated over TP (store once)', cfg: { idxDerep: true } },
  { key: 'idxbf8', name: 'index_k cache bf8 instead of bf16', cfg: { idxBf16: false } },
  { key: 'var', name: 'Variable chunk / flexible SP layout (a2a KV write)', cfg: { layout: 'var' } },
  { key: 'batch', name: 'Multi-request batching (16k token budget)', cfg: { batch: true, budget: 16384 } },
  { key: 'fused', name: 'Fused multi-user attention', cfg: { attn: 'fused' }, requires: ['batch'] },
  { key: 'async', name: 'Async stage handoff (overlap D2D)', cfg: { asyncHandoff: true } },
  { key: 'msa', name: 'MSA SP-local indexer (no K/V/index prefix all-gather)', cfg: { msaLocal: true } },
  { key: 'srpt', name: 'Shortest-first scheduling (30 s aging)', cfg: { policy: 'srpt' } },
];

function scenarios() {
  const base4 = { galaxies: 4, stages: 16, mesh: [2, 4], split: 'auto', chunk: 2048, cache: 'slots' };
  const base8 = { galaxies: 8, stages: 32, mesh: [2, 4], split: 'auto', chunk: 2048, cache: 'slots' };
  return [
    { key: 'g4_k0', label: '4 galaxies, today\'s kernels', base: Object.assign({ opEff: 0 }, base4) },
    { key: 'g4_k1', label: '4 galaxies, roofline kernels', base: Object.assign({ opEff: 1 }, base4) },
    { key: 'g8_k0', label: '8 galaxies, today\'s kernels', base: Object.assign({ opEff: 0 }, base8) },
    { key: 'g8_k1', label: '8 galaxies, roofline kernels', base: Object.assign({ opEff: 1 }, base8) },
  ];
}

function withFeatures(base, keys) {
  const cfg = Object.assign({}, base);
  for (const k of keys) Object.assign(cfg, FEATURES.find((f) => f.key === k).cfg);
  return cfg;
}
const allowed = (keys, f) => (f.requires || []).every((r) => keys.includes(r));

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const out = get('--out', require('./lib/paths.js').STUDY);
  const quick = args.includes('--quick');
  const pool = new Pool(Number(get('--workers', 36)));
  const concs = quick ? CONCS.filter((_, i) => i % 3 === 0) : CONCS;
  const cache = new Map();
  const evalKeys = async (sc, keys, extra) => {
    const cfg = Object.assign(withFeatures(sc.base, keys), extra || {});
    const id = JSON.stringify(cfg);
    if (!cache.has(id)) cache.set(id, pool.evalCfg(id, cfg, concs, SLO).then((r) => Object.assign(r, summarize(r.points, SLO))));
    return cache.get(id);
  };
  const t0 = Date.now();
  const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s]`, ...a);
  const res = { slo: SLO, concs, features: FEATURES, scenarios: {} };
  await Promise.all(scenarios().map(async (sc) => {
    const R = { label: sc.label, base: sc.base, greedy: [], loo: [], grid: [] };
    res.scenarios[sc.key] = R;
    // ---- greedy roadmap
    let keys = [];
    const infRef = (ks) => evalKeys(sc, ks, { cache: 'inf', hostTier: false }).then((r) => ({ goodput: r.goodput, at: r.at }));
    let cur = await evalKeys(sc, keys);
    R.greedy.push({ add: null, keys: [], goodput: cur.goodput, at: cur.at, peak: cur.peak, points: cur.points, plan: cur.plan, inf: await infRef([]) });
    log(sc.key, 'base', Math.round(cur.goodput));
    while (true) {
      const cands = FEATURES.filter((f) => !keys.includes(f.key) && allowed(keys, f));
      if (!cands.length) break;
      const rs = await Promise.all(cands.map((f) => evalKeys(sc, keys.concat([f.key]))));
      const step = cands.map((f, i) => ({ key: f.key, goodput: rs[i].goodput, gain: rs[i].goodput / Math.max(1, cur.goodput) }));
      step.sort((a, b) => b.goodput - a.goodput);
      const best = step[0];
      const r = rs[cands.findIndex((f) => f.key === best.key)];
      keys = keys.concat([best.key]);
      R.greedy.push({ add: best.key, keys: keys.slice(), goodput: r.goodput, at: r.at, peak: r.peak, points: r.points, plan: r.plan, candidates: step, inf: await infRef(keys) });
      log(sc.key, '+', best.key, Math.round(r.goodput), `(x${best.gain.toFixed(2)})`);
      cur = r;
    }
    // ---- leave-one-out from the full stack
    const full = FEATURES.map((f) => f.key);
    const fr = await evalKeys(sc, full);
    R.full = { goodput: fr.goodput, at: fr.at, peak: fr.peak, points: fr.points, plan: fr.plan };
    const loo = await Promise.all(FEATURES.map((f) => {
      const ks = full.filter((k) => k !== f.key && !(FEATURES.find((g) => g.key === k).requires || []).includes(f.key));
      return evalKeys(sc, ks).then((r) => ({ remove: f.key, goodput: r.goodput, loss: r.goodput / Math.max(1, fr.goodput), at: r.at }));
    }));
    R.loo = loo;
    // ---- grid on the best stack: features that did not hurt in the greedy
    const good = R.greedy.slice(1).filter((g, i) => g.goodput >= R.greedy[i].goodput * 0.995).map((g) => g.add);
    R.bestKeys = good;
    const topos = sc.base.galaxies === 4
      ? [[16, [2, 4], 1], [8, [4, 4], 1], [4, [8, 4], 1], [16, [4, 2], 1], [8, [2, 4], 2]]
      : [[32, [2, 4], 1], [16, [4, 4], 1], [8, [8, 4], 1], [16, [2, 4], 2], [32, [4, 2], 1], [8, [4, 4], 2]];
    const budgets = good.includes('batch') ? [4096, 8192, 16384, 32768] : [1024, 2048, 4096, 5120];
    const lanesOpt = !good.includes('pool') ? [null] : good.includes('arena') ? [{ arenaTokens: 2e6 }, { arenaTokens: 4e6 }, { arenaTokens: 8e6 }] : [{ lanes: 2 }, { lanes: 4 }, { lanes: 6 }];
    const jobs = [];
    for (const [S, mesh, reps] of topos) for (const b of budgets) for (const lo of lanesOpt) {
      const extra = Object.assign({ stages: S, mesh, replicas: reps }, lo || {});
      if (good.includes('batch')) extra.budget = b; else extra.chunk = b;
      jobs.push(evalKeys(sc, good, extra).then((r) => ({ extra, goodput: r.goodput, at: r.at, peak: r.peak, points: r.points, plan: r.plan })));
    }
    R.grid = (await Promise.all(jobs)).sort((a, b) => b.goodput - a.goodput);
    log(sc.key, 'grid best', Math.round(R.grid[0].goodput), JSON.stringify(R.grid[0].extra));
    // ---- best config: infinite-cache reference, seeds, sensitivity to the unverified assumptions
    const bestExtra = R.grid[0].extra;
    const sens = [
      ['inf cache', { cache: 'inf', hostTier: false }], ['seed 2', { seed: 2 }], ['seed 3', { seed: 3 }],
      ['host 0.5 TB/gx', { hostGBPerGalaxy: 512 }], ['host 2 TB/gx', { hostGBPerGalaxy: 2048 }],
      ['PCIe 16 GB/s/gx', { pcieGBsPerGalaxy: 16 }], ['PCIe 256 GB/s/gx', { pcieGBsPerGalaxy: 256 }],
      ['decode 90 tok/s', { decodeTps: 90 }], ['decode 360 tok/s', { decodeTps: 360 }],
      ['reserve 6 GB/chip', { reserveGB: 6 }], ['SLO-free peak', {}],
      ['TP=4 mesh only', sc.base.galaxies === 4 ? { mesh: [2, 4], stages: 16 } : { mesh: [2, 4], stages: 32 }],
    ];
    R.sens = await Promise.all(sens.map(([name, d]) => evalKeys(sc, good, Object.assign({}, bestExtra, d)).then((r) => ({ name, goodput: r.goodput, peak: r.peak ? r.peak.usefulTps : 0, at: r.at }))));
  }));
  pool.close();
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, JSON.stringify(res));
  log('wrote', out, `${cache.size} configs`);
}

if (require.main === module) main();
module.exports = { FEATURES, scenarios, withFeatures, CONCS, SLO };
