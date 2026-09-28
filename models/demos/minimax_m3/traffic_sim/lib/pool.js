// Worker pool: evaluates configurations over an ascending concurrency grid (early stop past the knee).
'use strict';
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const path = require('path'), os = require('os');

const KEEP = ['usefulTps', 'processedTps', 'newTps', 'reqPerS', 'ttftP50', 'ttftP90', 'ttftP99', 'ttftMean', 'hitRate', 'infHitRate',
  'reprefillFrac', 'padFrac', 'alignLossFrac', 'avgChunkTok', 'avgSegsPerChunk', 'maxUtil', 'done', 'warmupS', 'laneWaitMean',
  'hostTok', 'slotEvictions', 'error', 'warmupTimeout', 'eventCap'];

// goodput: best useful tok/s among points with p90 TTFT <= slo
// goodput: best useful tok/s among points with p90 TTFT <= slo, interpolated (in log p90) to the SLO crossing
// between the last passing and the next failing concurrency so the metric is continuous across the cache cliff
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

// The cache cliff is steep: goodput can change 10% between neighbouring grid points. Bisect (in log concurrency)
// between the highest-throughput point that meets the SLO and the next point that fails it. `run(c)` simulates
// concurrency c, appends it to `points` and returns it. The same rule runs in the artifact's worker.
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

class Pool {
  constructor(n, data) {
    this.n = n || Math.min(32, os.cpus().length); this.data = data || require('./paths.js').DATA;
    this.workers = []; this.queue = []; this.idle = [];
    for (let i = 0; i < this.n; i++) {
      const w = new Worker(__filename, { workerData: { data: this.data } });
      w.on('message', (m) => { if (m.ready) { this.idle.push(w); this.drain(); return; } const cb = w._cb; w._cb = null; this.idle.push(w); cb(m); this.drain(); });
      w.on('error', (e) => { console.error('worker error', e); });
      this.workers.push(w);
    }
  }
  drain() { while (this.idle.length && this.queue.length) { const w = this.idle.pop(); const t = this.queue.shift(); w._cb = t.cb; w.postMessage(t.job); } }
  run(job) { return new Promise((res) => { this.queue.push({ job, cb: res }); this.drain(); }); }
  // job: {id, cfg, concs, slo, stopFactor}
  evalCfg(id, cfg, concs, slo = 10, stopFactor = 4) { return this.run({ id, cfg, concs, slo, stopFactor }); }
  close() { for (const w of this.workers) w.terminate(); }
}

if (!isMainThread) {
  const SIM = require('../sim_core.js');
  const { loadAll } = require('../run.js');
  const { TR, cal } = loadAll(workerData.data);
  parentPort.on('message', (job) => {
    const pts = []; let over = 0, prevU = -1;
    const t0 = Date.now();
    const one = (c) => {
      let r;
      try { r = SIM.simulate(TR, cal, Object.assign({}, job.cfg, { concurrency: c })); } catch (e) { r = { error: String(e && e.stack || e) }; }
      const m = { conc: c };
      for (const k of KEEP) if (r[k] !== undefined) m[k] = typeof r[k] === 'number' ? +r[k].toPrecision(5) : r[k];
      pts.push(m);
      return m;
    };
    for (const c of job.concs) {
      const m = one(c);
      if (m.error) break;
      // stop once well past the knee: p90 far above the SLO and throughput no longer growing
      if (m.ttftP90 > job.stopFactor * job.slo && m.usefulTps <= prevU * 1.02) over++; else over = 0;
      prevU = Math.max(prevU, m.usefulTps);
      if (over >= 2) break;
    }
    refineCliff(pts, job.slo, job.refine === undefined ? 4 : job.refine, one);
    const plan = SIM.planSummary(SIM.makePlan(job.cfg, cal));
    parentPort.postMessage({ id: job.id, cfg: job.cfg, points: pts, plan, wallMs: Date.now() - t0 });
  });
  parentPort.postMessage({ ready: true });
}

module.exports = { Pool, summarize, refineCliff };
