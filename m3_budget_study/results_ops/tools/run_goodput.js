#!/usr/bin/env node
// Goodput of Pavlo's AgentX traffic sim (sim_core.js from branch philei/m3-traffic-sim) for 16 x [2,4] / 16 x [4,2]
// on 4 galaxies, with an optional replacement of CAL.effs['2x4'] / ['4x2'].
//
//   node run_goodput.js                                   # 2x4 + 4x2, full stack, budgets 4096/8192, SLO 10 and 3 s
//   node run_goodput.js --effs new_effs.json --features full,near --out goodput_new.json
//   node run_goodput.js --topo 4x2 --features pool,host,batch --budget 8192 --slo 10
//
// Options
//   --topo 2x4,4x2          stage meshes (16 stages each, 4 galaxies, one replica)
//   --features full,near    feature sets (see SETS below) or a '+'-joined list of feature keys, e.g. pool+host+batch
//   --budget 4096,8192      batching token budget (feature 'batch'; ignored as chunk when batching is off: chunk 2048)
//   --slo 10,3              p90 TTFT SLOs in seconds; each SLO gets its own sweep + cliff bisection, as in lib/pool.js
//   --effs FILE[,FILE..]    JSON {"2x4": {"moe": {op: eff}, "dense": {...}}, "4x2": {...}} (or {"effs": {...}}),
//                           merged per op over the calibrated CAL.effs, left to right (later files win)
//   --no-moe-mult           drop the pipeline MoE multiplier (CAL.pipe.moeMult, fitted against the original effs)
//   --pipe JSON|FILE        merge into CAL.pipe, e.g. '{"ringC":0.35}'. The dense ring-joint compute efficiency the
//                           pipeline uses is pipe.ringC (x effs[mesh].dense.ring_c / effs['2x4'].dense.ring_c), and the
//                           scan efficiency is pipe.ringScan: effs.dense.ring_c/ring_scan alone do not move a 2x4 run
//   --arena N               lane arena tokens when the set has 'arena' (default 2e6 = the study's best grid cell)
//   --lanes N               fixed lanes when the set has 'pool' but not 'arena' (default 4, the feature default)
//   --set key=value         extra sim config (JSON-parsed), applied last
//   --workers N             worker threads (default 16); --data DIR traffic dir (default $M3SIM_DATA or Pavlo's copy)
//   --out FILE              write all results, including every concurrency point, as JSON
'use strict';
const fs = require('fs'), path = require('path'), os = require('os');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

// Feature definitions: copied from study.js on philei/m3-traffic-sim (FEATURES).
const FEATURES = {
  bounded: { boundedDense: true },
  unaligned: { unaligned: true },
  pool: { cache: 'pool', lanes: 4, laneArena: false, laneScope: 'stage' },
  arena: { laneArena: true, arenaTokens: 4e6 },
  host: { hostTier: true },
  idxdedup: { idxDerep: true },
  idxbf8: { idxBf16: false },
  var: { layout: 'var' },
  batch: { batch: true, budget: 16384 },
  fused: { attn: 'fused' },
  async: { asyncHandoff: true },
  msa: { msaLocal: true },
  srpt: { policy: 'srpt' },
};
const SETS = {
  // g4_k0 (4 galaxies, today's kernels) greedy-kept stack = the study's grid features (results/study.json bestKeys)
  full: ['pool', 'arena', 'host', 'idxdedup', 'batch', 'var', 'async', 'idxbf8', 'msa', 'srpt', 'fused', 'unaligned'],
  // Best match to "near-term features only: 16x[4,2] 2-3% worse than 16x[2,4]" (p90 <= 10 s): P0 + P1 without the
  // variable chunk, and without the SP-local indexer. Pavlo never names the set; see pavlo_reference.md for candidates.
  near: ['pool', 'bounded', 'host', 'async', 'batch', 'idxdedup'],
  // README roadmap tiers P0 + P1 (slot lanes + pool, bounded dense gather, host tier, async handoff, batching,
  // index_k de-replication, variable chunk)
  p0p1: ['pool', 'bounded', 'host', 'async', 'batch', 'idxdedup', 'var'],
  today: [],
};
// study.js CONCS and the goodput rule of lib/pool.js (early stop past the knee, 4 bisection steps at the cliff)
const CONCS = [8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 576, 640, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048, 2560, 3072, 4096];
const BASE = { galaxies: 4, stages: 16, split: 'auto', chunk: 2048, cache: 'slots', opEff: 0, replicas: 1 };
const KEEP = ['usefulTps', 'processedTps', 'ttftP50', 'ttftP90', 'ttftP99', 'hitRate', 'infHitRate', 'reprefillFrac', 'padFrac', 'maxUtil', 'avgChunkTok', 'avgSegsPerChunk', 'done', 'error'];
const DEFAULT_DATA = process.env.M3SIM_DATA || '/data/philei/m3_traffic_sim/data';

// summarize / refineCliff: copied from lib/pool.js on philei/m3-traffic-sim
function summarize(points, slo) {
  let best = null, peak = null, g = 0;
  const pts = points.filter((p) => !p.error).sort((a, b) => a.conc - b.conc);
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    if (!peak || p.usefulTps > peak.usefulTps) peak = p;
    if (p.ttftP90 <= slo) {
      if (!best || p.usefulTps > best.usefulTps) best = p;
      g = Math.max(g, p.usefulTps);
      const q = pts[i + 1];
      if (q && q.ttftP90 > slo && q.usefulTps > p.usefulTps) {
        const w = (Math.log(slo) - Math.log(Math.max(1e-3, p.ttftP90))) / (Math.log(q.ttftP90) - Math.log(Math.max(1e-3, p.ttftP90)));
        g = Math.max(g, p.usefulTps + Math.min(1, Math.max(0, w)) * (q.usefulTps - p.usefulTps));
      }
    }
  }
  return { goodput: g, at: best, peak };
}
function refineCliff(points, slo, steps, run) {
  const ok = (p) => !p.error && p.ttftP90 <= slo;
  for (let i = 0; i < steps; i++) {
    const pts = points.filter((p) => !p.error).sort((a, b) => a.conc - b.conc);
    let lo = null;
    for (const p of pts) if (ok(p) && (!lo || p.usefulTps >= lo.usefulTps)) lo = p;
    if (!lo) return;
    const hi = pts.find((p) => p.conc > lo.conc && !ok(p));
    if (!hi) return;
    const mid = Math.round(Math.sqrt(lo.conc * hi.conc) / 8) * 8;
    if (mid <= lo.conc || mid >= hi.conc) return;
    const m = run(mid);
    if (m.error) return;
  }
}

function buildCal(SIM, opts) {
  const cal = SIM.calibrate(JSON.parse(fs.readFileSync(path.join(__dirname, 'calib_data.json'))));
  for (const f of opts.effs || []) {
    let over = JSON.parse(fs.readFileSync(f));
    if (over.effs) over = over.effs;
    for (const mesh in over) {
      if (mesh === 'about') continue;
      cal.effs[mesh] = cal.effs[mesh] || { moe: {}, dense: {} };
      for (const kind of ['moe', 'dense']) Object.assign(cal.effs[mesh][kind], (over[mesh] || {})[kind] || {});
    }
  }
  if (opts.noMoeMult) cal.pipe.moeMult = [1, 0];
  if (opts.pipe) Object.assign(cal.pipe, opts.pipe);
  return cal;
}

if (!isMainThread) {
  const SIM = require('./sim_core.js');
  const hdr = JSON.parse(fs.readFileSync(path.join(workerData.data, 'traffic.json')));
  const b = fs.readFileSync(path.join(workerData.data, 'traffic.bin'));
  const TR = SIM.loadTraffic(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), hdr);
  const cal = buildCal(SIM, workerData.opts);
  parentPort.on('message', (job) => {
    const pts = []; let over = 0, prevU = -1;
    const one = (c) => {
      let r;
      try { r = SIM.simulate(TR, cal, Object.assign({}, job.cfg, { concurrency: c })); } catch (e) { r = { error: String((e && e.stack) || e) }; }
      const m = { conc: c };
      for (const k of KEEP) if (r[k] !== undefined) m[k] = typeof r[k] === 'number' ? +r[k].toPrecision(5) : r[k];
      pts.push(m);
      return m;
    };
    for (const c of CONCS) {
      const m = one(c);
      if (m.error) break;
      if (m.ttftP90 > 4 * job.slo && m.usefulTps <= prevU * 1.02) over++; else over = 0;
      prevU = Math.max(prevU, m.usefulTps);
      if (over >= 2) break;
    }
    refineCliff(pts, job.slo, 4, one);
    const plan = SIM.planSummary(SIM.makePlan(job.cfg, cal));
    parentPort.postMessage({ id: job.id, points: pts, plan });
  });
  parentPort.postMessage({ ready: true });
} else if (require.main === module) {
  main().catch((e) => { console.error(e); process.exit(1); });
}

function parse(argv) {
  const a = { topo: ['2x4', '4x2'], features: ['full'], budget: [4096, 8192], slo: [10, 3], effs: null, noMoeMult: false, pipe: null,
    arena: 2e6, lanes: 4, set: {}, workers: 16, data: DEFAULT_DATA, out: null };
  const list = (s, f) => s.split(',').map(f || ((x) => x));
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i], v = () => argv[++i];
    if (k === '--topo') a.topo = list(v());
    else if (k === '--features') a.features = list(v());
    else if (k === '--budget') a.budget = list(v(), Number);
    else if (k === '--slo') a.slo = list(v(), Number);
    else if (k === '--effs') a.effs = list(v(), (x) => path.resolve(x));
    else if (k === '--no-moe-mult') a.noMoeMult = true;
    else if (k === '--pipe') { const x = v(); a.pipe = JSON.parse(fs.existsSync(x) ? fs.readFileSync(x, 'utf8') : x); }
    else if (k === '--arena') a.arena = Number(v());
    else if (k === '--lanes') a.lanes = Number(v());
    else if (k === '--workers') a.workers = Number(v());
    else if (k === '--data') a.data = v();
    else if (k === '--out') a.out = v();
    else if (k === '--set') { const [key, ...r] = v().split('='); const s = r.join('='); try { a.set[key] = JSON.parse(s); } catch (e) { a.set[key] = s; } }
    else throw new Error('unknown option ' + k);
  }
  return a;
}

function configFor(a, topo, setName, budget) {
  const keys = SETS[setName] || setName.split('+');
  for (const k of keys) if (!FEATURES[k]) throw new Error(`unknown feature '${k}' in set '${setName}'`);
  const cfg = Object.assign({}, BASE, { mesh: topo.split('x').map(Number) });
  for (const k of keys) Object.assign(cfg, FEATURES[k]);
  if (keys.includes('arena')) cfg.arenaTokens = a.arena;
  else if (keys.includes('pool')) cfg.lanes = a.lanes;
  if (keys.includes('batch')) cfg.budget = budget; else cfg.chunk = budget; // study grid: budget, or chunk without batching
  return Object.assign(cfg, a.set);
}

async function main() {
  const a = parse(process.argv);
  const n = Math.max(1, Math.min(a.workers, os.cpus().length));
  const workers = [], idle = [], queue = [], cbs = new Map();
  let ready = 0;
  await new Promise((res) => {
    for (let i = 0; i < n; i++) {
      const w = new Worker(__filename, { workerData: { data: a.data, opts: { effs: a.effs, noMoeMult: a.noMoeMult, pipe: a.pipe } } });
      w.on('message', (m) => {
        if (m.ready) { idle.push(w); if (++ready === n) res(); return; }
        cbs.get(m.id)(m); cbs.delete(m.id); idle.push(w); drain();
      });
      w.on('error', (e) => { console.error('worker error', e); process.exit(1); });
      workers.push(w);
    }
  });
  const drain = () => { while (idle.length && queue.length) { const w = idle.pop(); w.postMessage(queue.shift()); } };
  let nid = 0;
  const evalCfg = (cfg, slo) => new Promise((res) => { const id = nid++; cbs.set(id, res); queue.push({ id, cfg, slo }); drain(); });
  const t0 = Date.now();
  const jobs = [];
  for (const setName of a.features) for (const topo of a.topo) for (const budget of a.budget) for (const slo of a.slo) {
    const cfg = configFor(a, topo, setName, budget);
    jobs.push(evalCfg(cfg, slo).then((r) => {
      const s = summarize(r.points, slo);
      return { features: setName, topo: `16x[${topo.replace('x', ',')}]`, budget, slo, goodput: s.goodput, at: s.at, peak: s.peak, cfg, plan: r.plan, points: r.points };
    }));
  }
  const res = await Promise.all(jobs);
  for (const w of workers) w.terminate();
  const k = (x) => (x / 1000).toFixed(1) + 'k';
  console.log(`effs: ${a.effs ? a.effs.join(' + ') : 'calibrated (calib_data.json)'}${a.noMoeMult ? ', no MoE multiplier' : ''}${a.pipe ? ', pipe ' + JSON.stringify(a.pipe) : ''}; data ${a.data}; ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  console.log('features  topo       budget  SLO  goodput   at conc  p90 s  util');
  for (const r of res) {
    console.log(`${r.features.padEnd(9)} ${r.topo.padEnd(10)} ${String(r.budget).padStart(6)} ${String(r.slo).padStart(4)}  ${k(r.goodput).padStart(7)}  ${String(r.at ? r.at.conc : '-').padStart(7)}  ${r.at ? r.at.ttftP90.toFixed(2).padStart(5) : '  -  '}  ${r.at ? (100 * r.at.maxUtil).toFixed(0) + '%' : '-'}`);
  }
  if (a.out) fs.writeFileSync(a.out, JSON.stringify({ args: a, results: res }, null, 1));
}

module.exports = { FEATURES, SETS, summarize, refineCliff, configFor };
