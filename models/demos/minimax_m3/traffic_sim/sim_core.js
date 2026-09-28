/* sim_core.js - MiniMax-M3 pipeline-prefill traffic simulator (AgentX replay x per-op roofline).
 *
 * One file, no dependencies. Used by the artifact (inlined, runs in a Web Worker) and by node (sweep.js,
 * validate.js). See README.md for the model, its calibration and the knobs.
 *
 * Units: cost model in milliseconds; simulation clock in seconds; KV in tokens (64-token blocks in the trace).
 */
(function (root) {
  'use strict';

  // ------------------------------------------------------------------------------------------------------
  // Model + hardware constants
  // ------------------------------------------------------------------------------------------------------
  const M3 = {
    E: 6144, Hq: 64, Hkv: 4, d: 128, Hi: 4, di: 128, sparseKeys: 2048, Ex: 128, topk: 4, I: 3072, Is: 3072,
    Id: 12288, V: 200064, L: 60, nDense: 3, maxCtx: 1048576,
  };
  // Blackhole chip (tt-moe-nappkin lib/system.py): 110 cores x 4.096 kFLOP/cycle x 1.35 GHz (LoFi), HiFi2 = 1/2;
  // 512 GB/s GDDR6, 32 GiB; ethernet 2 links x 200 Gb/s x 2 dirs per axis neighbour (100 GB/s bidir CCL, 50 GB/s uni).
  const HW = { F_lofi: 608e12, F_hifi: 304e12, dram: 512e9, dramCap: 32 * 2 ** 30, linkBi: 100e9, linkUni: 50e9, pcie: 64e9 };
  const BF8 = 1.0625, BF4 = 0.5625, BF16 = 2;
  const MB = 1e6, GB = 1e9;

  const OPS_MOE = ['norm_ag', 'qkv', 'idx_branch', 'misc', 'o_proj', 'attn_rs', 'shared', 'router', 'dispatch', 'experts', 'combine', 'moe_reduce'];
  const OPS_MSA_SEG = ['ag_kv', 'ag_idx', 'indexer', 'sparse', 'kv_a2a'];
  const OPS_DENSE = ['norm_ag', 'qkv', 'misc', 'o_proj', 'attn_rs', 'dense_mlp'];
  const OPS_DENSE_SEG = ['ring', 'kv_a2a'];
  const CCL_OPS = new Set(['norm_ag', 'attn_rs', 'dispatch', 'combine', 'ag_kv', 'ag_idx', 'kv_a2a', 'shared', 'moe_reduce', 'dense_mlp']);
  // zone name (parse_zone_perf.py) -> op
  const ZONE_MOE = {
    qkv: 'attn/qkv_proj', idx_branch: 'attn/index_branch', ag_kv: 'attn/ag_kv', ag_idx: 'attn/ag_index_k', indexer: 'attn/indexer',
    sparse: 'attn/sparse_sdpa', o_proj: 'attn/o_proj', attn_rs: 'attn/ccl_out_reduce_scatter', shared: 'mlp/shared_expert',
    router: 'mlp/router_topk', dispatch: 'mlp/dispatch', experts: 'mlp/experts_mm', combine: 'mlp/combine', moe_reduce: 'mlp/moe_reduce',
  };
  const ZONE_DENSE = { qkv: 'attn/qkv_proj', ring: 'attn/ring_joint_sdpa', o_proj: 'attn/o_proj', attn_rs: 'attn/ccl_out_reduce_scatter', dense_mlp: 'mlp' };
  // target ("fixed kernels") efficiency per op and latency floors (ms)
  const TARGET_EFF = { matmul: 0.7, dram: 0.8, link: 0.8 };
  const OP_CLASS = {
    norm_ag: 'link', qkv: 'matmul', idx_branch: 'matmul', misc: 'dram', o_proj: 'matmul', attn_rs: 'link', shared: 'matmul',
    router: 'matmul', dispatch: 'link', experts: 'matmul', combine: 'link', moe_reduce: 'dram', ag_kv: 'link', ag_idx: 'link',
    indexer: 'matmul', sparse: 'matmul', kv_a2a: 'link', dense_mlp: 'matmul', ring: 'matmul', ring_scan: 'link',
  };
  const LAT_MEAS = { ccl: 0.04, op: 0.01 };
  const LAT_TGT = { ccl: 0.02, op: 0.005 };

  // ------------------------------------------------------------------------------------------------------
  // Per-op roofline (seconds at 100% efficiency).  ctx: {sp,tp,P,T,idxB}  seg: {n,k,cap}
  // ------------------------------------------------------------------------------------------------------
  function roofTok(op, c) {
    const { sp, tp, P, T } = c;
    const E = M3.E, tl = T / sp;
    // routed tokens: the padded chunk tail is trimmed from routing/dispatch/experts/combine (padding_config actual_isl)
    const Tr = c.Tr === undefined ? T : c.Tr, trl = Tr / sp;
    const rsBytes = (tp - 1) / tp * tl * E * BF16;
    const a2a = (m) => (sp > 1 ? sp * m / (4 * HW.linkUni) : m / HW.dram); // linear SP line, bisection-bound
    switch (op) {
      case 'norm_ag': return rsBytes / HW.linkBi;                                  // one TP all-gather (x2 per layer)
      case 'qkv': return Math.max(2 * tl * E * 9216 / tp / HW.F_hifi, E * 9216 / tp * BF8 / HW.dram);
      case 'idx_branch': return Math.max(2 * tl * E * 640 / tp / HW.F_hifi, E * 640 * BF8 / HW.dram);
      case 'misc': return 8 * tl * E / tp * BF16 / HW.dram;                       // rope, residual, typecasts, cache write
      case 'o_proj': return Math.max(2 * tl * 8192 / tp * E / HW.F_hifi, 8192 * E / tp * BF8 / HW.dram);
      case 'attn_rs': return rsBytes / HW.linkBi;
      case 'shared': return Math.max(6 * tl * E * M3.Is / tp / HW.F_hifi, 3 * E * M3.Is / tp * BF16 / HW.dram) + rsBytes / HW.linkBi;
      case 'router': return Math.max(2 * tl * E * M3.Ex / HW.F_hifi, E * M3.Ex * BF16 / HW.dram);
      case 'dispatch': return a2a(trl * (M3.topk / tp) * E * 1.0);
      case 'combine': return a2a(trl * (M3.topk / tp) * E * BF16);
      case 'experts': {
        const flops = Tr * M3.topk / P * 6 * E * M3.I * c.imb;
        const wbytes = M3.Ex / P * 3 * E * M3.I * BF4;
        // today's kernel reads weights then computes (measured: time grows linearly with routed tokens even at
        // 160 tokens/expert); a fixed kernel overlaps them -> blend sum -> max with the op-efficiency knob
        const cc = flops / HW.F_lofi, mm = wbytes / HW.dram, ov = c.ovl || 0;
        return (1 - ov) * (cc + mm) + ov * Math.max(cc, mm);
      }
      case 'moe_reduce': return trl * (M3.topk / tp) * E * BF16 * 2 / HW.dram + rsBytes / HW.linkBi;
      case 'dense_mlp': return Math.max(6 * tl * E * M3.Id / tp / HW.F_hifi, 3 * E * M3.Id / tp * BF16 / HW.dram) + rsBytes / HW.linkBi;
    }
    return 0;
  }
  function roofSeg(op, c, s) {
    const { sp, tp } = c;
    const nl = s.n / sp, kvlen = s.k + s.n;
    const kvHeadB = 2 * (M3.Hkv / tp) * M3.d * BF8;          // K+V bytes/token on one chip (heads over TP)
    switch (op) {
      case 'ag_kv':
        if (sp <= 1) return 0;
        return (sp - 1) / sp * (c.msaLocal ? Math.min(kvlen, 16384) : kvlen) * kvHeadB / HW.linkBi;
      case 'ag_idx':
        if (sp <= 1 || c.msaLocal) return 0;
        return (sp - 1) / sp * kvlen * M3.di * c.idxB / HW.linkBi;
      case 'indexer': // block-pooled index scoring + top-k; per-token part dominates at these sizes
        return 2 * nl * M3.Hi * M3.di * (2048 + kvlen / 128) / HW.F_hifi;
      case 'sparse': {
        const keys = Math.min(M3.sparseKeys, kvlen);
        return Math.max(4 * nl * (M3.Hq / tp) * keys * M3.d / HW.F_hifi, nl * keys * kvHeadB / 32 / HW.dram);
      }
      case 'kv_a2a': // variable layout: route the new K/V/index rows to their owner SP row
        if (!c.varLayout || sp <= 1) return 0;
        return sp * nl * (kvHeadB + M3.di * c.idxB) / (4 * HW.linkUni);
      case 'ring_c': return 4 * nl * (M3.Hq / tp) * (s.k + s.n / 2) * M3.d / HW.F_hifi;
      case 'ring_scan': { // ring-joint walks the whole per-device cache shard (capacity/sp) every chunk
        const scan = c.bounded ? kvlen : Math.max(s.cap, kvlen);
        return scan / sp * kvHeadB / HW.linkBi;
      }
    }
    return 0;
  }
  const isCcl = (op) => CCL_OPS.has(op);
  // Wave quantization of attention kernels: work units = (32-row query tiles per chip) x (query heads per chip),
  // spread over 110 Tensix cores; a short segment leaves cores idle (sequential per-user attention pays this per
  // segment, a fused multi-user kernel pays it once for the whole batch).
  const CORES = 110;
  function waveFactor(nTok, c) {
    const units = Math.max(1, Math.ceil(nTok / c.sp / 32)) * (M3.Hq / c.tp);
    return Math.ceil(units / CORES) * CORES / units;
  }
  const WAVE_OPS = new Set(['sparse', 'indexer', 'ring_c']);

  // ------------------------------------------------------------------------------------------------------
  // Calibration: per-mesh measured efficiencies from zone profiles + pipeline-level fit ([2,4] 16-stage runs)
  // ------------------------------------------------------------------------------------------------------
  const ZONE_T = 5120, ZONE_K = 51200, ZONE_CAP = 56320;
  const IMB0 = 1.2; // expert-load imbalance assumed in the roofline (calibration and DEFAULTS.expertImb)

  function zoneEff(zones, mesh, idxB) {
    const [sp, tp] = mesh;
    const c = { sp, tp, P: sp * tp, T: ZONE_T, idxB, imb: IMB0, varLayout: false, bounded: false, msaLocal: false };
    const seg = { n: ZONE_T, k: ZONE_K, cap: ZONE_CAP };
    const eff = { moe: {}, dense: {} };
    const zm = (name) => (zones[name] ? zones[name].mean : null);
    const moeZ = (k) => zm('layer03_sparse/' + k), denZ = (k) => zm('layer00_dense/' + k);
    const fit = (roof, meas, op) => {
      const lat = isCcl(op) ? LAT_MEAS.ccl : LAT_MEAS.op;
      return Math.min(1, Math.max(0.003, roof * 1e3 / Math.max(1e-3, meas - lat)));
    };
    // MoE layer
    let listed = 0;
    for (const op of OPS_MOE.concat(OPS_MSA_SEG)) {
      if (op === 'norm_ag') {
        const m = (moeZ('input_norm_allgather') + moeZ('post_attn_norm_allgather')) / 2;
        eff.moe[op] = fit(roofTok(op, c), m, op); listed += 2 * m; continue;
      }
      if (op === 'misc' || op === 'kv_a2a') continue;
      let m = moeZ(ZONE_MOE[op]);
      if (op === 'shared') m = moeZ('mlp/shared_expert');
      const r = OPS_MSA_SEG.includes(op) ? roofSeg(op, c, seg) : roofTok(op, c);
      eff.moe[op] = fit(r, m, op); listed += m;
    }
    eff.moe.misc = fit(roofTok('misc', c), Math.max(0.05, zm('layer03_sparse') - listed), 'misc');
    eff.moe.kv_a2a = eff.moe.dispatch;
    // dense layer
    listed = 0;
    for (const op of OPS_DENSE) {
      if (op === 'norm_ag') {
        const m = (denZ('input_norm_allgather') + denZ('post_attn_norm_allgather')) / 2;
        eff.dense[op] = fit(roofTok(op, c), m, op); listed += 2 * m; continue;
      }
      if (op === 'misc') continue;
      const m = denZ(ZONE_DENSE[op]);
      eff.dense[op] = fit(roofTok(op, c), m, op); listed += m;
    }
    const ring = denZ('attn/ring_joint_sdpa'); listed += ring;
    eff.dense.misc = fit(roofTok('misc', c), Math.max(0.05, zm('layer00_dense') - listed), 'misc');
    eff.dense.ring_c = fit(roofSeg('ring_c', c, seg), ring, 'ring');
    eff.dense.ring_scan = 0.015; // overwritten by the pipeline fit
    eff.dense.kv_a2a = eff.moe.dispatch;
    return eff;
  }

  function interpEff(a, b, w) { // geometric interpolation of two eff tables
    const out = { moe: {}, dense: {} };
    for (const k of ['moe', 'dense']) for (const op in a[k]) out[k][op] = Math.exp((1 - w) * Math.log(a[k][op]) + w * Math.log(b[k][op]));
    return out;
  }

  // Build a calibration object from calib_data.json (zones + pipeline rows + tables).
  function calibrate(data) {
    const idxB = BF16; // the measured runs use M3_INDEX_CACHE_BF16=1
    const effs = {};
    for (const m in data.zones) effs[m] = zoneEff(data.zones[m], m.split('x').map(Number), idxB);
    // [4,4] has no profile: geometric midpoint of [2,4] and [8,4] (log-chip-count midpoint)
    if (effs['2x4'] && effs['8x4']) effs['4x4'] = interpEff(effs['2x4'], effs['8x4'], 0.5);
    const cal = { effs, pipe: { moeMult: [1, 0], stageOv: [2, 0], ringC: null, ringScan: 0.015, embed: 1.6, block: [12, 2.6], hop: [12, 0] }, fit: {} };
    fitPipeline(cal, data);
    return cal;
  }

  // Fit the pipeline-level corrections on the [2,4] 16-stage runs A/B/C.
  function fitPipeline(cal, data) {
    const P = data.pipeline; if (!P || !P.B) return;
    const mesh = [2, 4];
    const base = (T) => ({ sp: 2, tp: 4, P: 8, T, idxB: BF16, imb: IMB0, varLayout: false, bounded: false, msaLocal: false });
    const eff = cal.effs['2x4'];
    const ctx = (T, na) => Object.assign(base(T), { Tr: na });
    // samples: every (cell, rank, chunk position) median of the loaded blocks; actual tokens and kv are exact
    const samples = [];
    for (const run of ['A', 'B', 'C']) {
      const r = P[run]; const counts = r.layers.split(',').map(Number); const T = r.chunk;
      for (const cell of r.cells || []) {
        const kA = Math.floor(cell.cached / T) * T;
        for (let pos = 0; pos < cell.n_ch; pos++) {
          const na = Math.min(T, cell.new - pos * T);
          let start = 0;
          counts.forEach((n, s) => {
            const nd = Math.max(0, Math.min(3, start + n) - start); start += n;
            const y = cell.pos_ms[s][pos]; if (y == null) return;
            samples.push({ run, T, na, k: kA + pos * T, cap: cell.cached + 51200, s, n, nd, y });
          });
        }
      }
    }
    cal.fit.samples = samples.length;
    const moeZone = (p) => layerMs('moe', ctx(p.T, p.na), [{ n: p.T, na: p.na, k: p.k, cap: p.cap }], eff.moe, LAT_MEAS, 'seq');
    // 1) MoE-only stages: stage_ms = nMoE * (a + b*5120/T) * moeZone + o0   (o0 >= 0: per-chunk stage overhead)
    const moeS = samples.filter((p) => p.nd === 0);
    const X = moeS.map((p) => { const z = moeZone(p); return [p.n * z, p.n * z * 5120 / p.T, 1]; });
    const Y = moeS.map((p) => p.y);
    let beta = lstsq(X, Y);
    if (beta[2] < 0) { const b2 = lstsq(X.map((x) => [x[0], x[1]]), Y); beta = [b2[0], b2[1], 0]; }
    cal.pipe.moeMult = [beta[0], beta[1]]; cal.pipe.stageOv = [beta[2], 0];
    cal.fit.moe_rmse = rmse(X, Y, beta);
    // 2) dense ring: single-dense-layer stages (B/C ranks 1,2) -> grid search eff_ring_c, eff_ring_scan
    const dense = samples.filter((p) => p.nd === 1 && p.n === 1 && p.s > 0);
    const ov = (T) => cal.pipe.stageOv[0] + cal.pipe.stageOv[1] * T / 1000;
    let best = null;
    for (let lc = Math.log(0.05); lc <= Math.log(1.0); lc += 0.02) for (let ls = Math.log(0.001); ls <= Math.log(0.2); ls += 0.02) {
      const e = Object.assign({}, eff.dense, { ring_c: Math.exp(lc), ring_scan: Math.exp(ls) });
      let err = 0;
      for (const p of dense) {
        const m = layerMs('dense', ctx(p.T, p.na), [{ n: p.T, na: p.na, k: p.k, cap: p.cap }], e, LAT_MEAS, 'seq') + ov(p.T);
        err += (Math.log(m) - Math.log(p.y)) ** 2;
      }
      if (!best || err < best.err) best = { err, rc: Math.exp(lc), rs: Math.exp(ls) };
    }
    cal.pipe.ringC = best.rc; cal.pipe.ringScan = best.rs; cal.fit.dense_rmse_log = Math.sqrt(best.err / Math.max(1, dense.length));
    // 3) embedding = rank0 - rank1 (both one dense layer) in B/C
    const r0 = samples.filter((p) => p.s === 0 && p.n === 1), r1 = new Map(dense.filter((p) => p.s === 1).map((p) => [`${p.run}|${p.k}|${p.na}`, p.y]));
    const emb = r0.map((p) => p.y - (r1.get(`${p.run}|${p.k}|${p.na}`) ?? p.y)).sort((a, b) => a - b);
    cal.pipe.embed = Math.max(0, emb.length ? emb[Math.floor(emb.length / 2)] : 1.6);
  }

  function lstsq(X, y) {
    const n = X[0].length; const A = Array.from({ length: n }, () => new Array(n).fill(0)); const b = new Array(n).fill(0);
    for (let i = 0; i < X.length; i++) for (let j = 0; j < n; j++) { b[j] += X[i][j] * y[i]; for (let k = 0; k < n; k++) A[j][k] += X[i][j] * X[i][k]; }
    for (let i = 0; i < n; i++) { // gaussian elimination
      let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
      [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
      for (let r = i + 1; r < n; r++) { const f = A[r][i] / A[i][i]; for (let k = i; k < n; k++) A[r][k] -= f * A[i][k]; b[r] -= f * b[i]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= A[i][k] * x[k]; x[i] = s / A[i][i]; }
    return x;
  }
  function rmse(X, y, beta) { let e = 0; for (let i = 0; i < X.length; i++) { let p = 0; for (let j = 0; j < beta.length; j++) p += X[i][j] * beta[j]; e += (p - y[i]) ** 2; } return Math.sqrt(e / X.length); }

  // ------------------------------------------------------------------------------------------------------
  // Layer cost (ms) for a batch of segments.  kind: 'moe' | 'dense'; attn: 'seq' | 'fused'
  // ------------------------------------------------------------------------------------------------------
  function opMs(op, roofS, eff, lat) {
    return (isCcl(op) ? lat.ccl : lat.op) + roofS * 1e3 / eff[op];
  }
  function layerMs(kind, c, segs, eff, lat, attn) {
    let t = 0;
    if (kind === 'moe') {
      for (const op of OPS_MOE) t += (op === 'norm_ag' ? 2 : 1) * opMs(op, roofTok(op, c), eff, lat);
      for (const op of OPS_MSA_SEG) {
        if (op === 'kv_a2a' && !c.varLayout) continue;
        if ((op === 'ag_kv' || op === 'ag_idx') && c.sp <= 1) continue;
        if (op === 'ag_idx' && c.msaLocal) continue;
        const l = isCcl(op) ? lat.ccl : lat.op;
        const wave = WAVE_OPS.has(op);
        let sum = 0;
        for (const s of segs) sum += roofSeg(op, c, s) * 1e3 / eff[op] * (wave && attn !== 'fused' ? waveFactor(s.n, c) / waveFactor(ZONE_T, c) : 1);
        if (wave && attn === 'fused') sum *= waveFactor(c.T, c) / waveFactor(ZONE_T, c);
        t += sum + (attn === 'fused' ? l : l * segs.length);
      }
    } else {
      for (const op of OPS_DENSE) t += (op === 'norm_ag' ? 2 : 1) * opMs(op, roofTok(op, c), eff, lat);
      let sum = 0;
      const wf = attn === 'fused' ? waveFactor(c.T, c) / waveFactor(ZONE_T, c) : 0;
      for (const s of segs) {
        const w = attn === 'fused' ? wf : waveFactor(s.n, c) / waveFactor(ZONE_T, c);
        const rc = roofSeg('ring_c', c, s) * 1e3 / eff.ring_c * w, rs = roofSeg('ring_scan', c, s) * 1e3 / eff.ring_scan;
        sum += Math.max(rc, rs);
        if (c.varLayout && c.sp > 1) sum += roofSeg('kv_a2a', c, s) * 1e3 / eff.kv_a2a;
      }
      t += sum + (attn === 'fused' ? lat.op : lat.op * segs.length);
    }
    return t;
  }

  // ------------------------------------------------------------------------------------------------------
  // System plan: stages, layer split, memory -> KV capacity, cost functions
  // ------------------------------------------------------------------------------------------------------
  const DEFAULTS = {
    galaxies: 4, replicas: 1, mesh: [2, 4], stages: 16, split: 'auto',
    opEff: 0,              // 0 = today's measured kernels, 1 = roofline target efficiencies
    asyncHandoff: false,   // stage-to-stage D2D overlapped with compute (no blocking send), link-rate transfer
    boundedDense: false,   // dense ring-joint SDPA scans [0, kv_len) instead of the whole lane capacity
    msaLocal: false,       // MSA: SP-local indexer + top-k merge, fetch only selected K/V blocks (no prefix all-gather)
    idxBf16: true, idxDerep: false, // index_k cache dtype / de-replicated over TP (today: bf16 x TP replicas)
    chunk: 5120, layout: 'fixed', batch: false, budget: 16384, attn: 'seq', policy: 'fcfs',
    cache: 'slots',        // slots | pool | paging | inf
    lanes: 3, laneScope: 'stage', laneLen: M3.maxCtx, laneArena: false, arenaTokens: 4e6,
    slotLen: M3.maxCtx, unaligned: false, copyOverlap: true, copyContention: 0.25,
    hostTier: false, hostGBPerGalaxy: 1024, pcieGBsPerGalaxy: 64,
    reserveGB: 3, expertImb: IMB0, maxInflight: 0,
    concurrency: 64, decodeTps: 180, duration: 1800, seed: 1, idleCap: 10, startMin: 0, startMax: 1, maxWarmup: 1e6,
    gapCap: Infinity,      // AgentX forbids capping recorded idle gaps (only the 10 s system-idle cap applies)
    srptMaxWait: 30,
  };

  function kvBytesPerTokenLayer(cfg, tp) { // whole stage, physical
    return 2 * M3.Hkv * M3.d * BF8 + M3.di * (cfg.idxBf16 ? BF16 : BF8) * (cfg.idxDerep ? 1 : tp);
  }

  function splitLayers(cfg, S, stageCostFn) {
    const L = M3.L, D = M3.nDense;
    if (Array.isArray(cfg.split)) return cfg.split.slice();
    const even = () => { const b = Math.floor(L / S), r = L % S; return Array.from({ length: S }, (_, i) => b + (i < r ? 1 : 0)); };
    if (cfg.split === 'even' || S === 1) return even();
    // auto: enumerate heads for the first m<=3 stages (they hold the dense layers), rest even; minimise weighted bottleneck
    let best = null; const seen = new Set();
    const consider = (c) => {
      const key = c.join(','); if (seen.has(key)) return; seen.add(key);
      const v = stageCostFn(c); if (!best || v < best.v) best = { v, c };
    };
    consider(even());
    for (let m = 1; m <= Math.min(3, S - 1); m++) {
      const heads = []; const rec = (pre) => { if (pre.length === m) { heads.push(pre.slice()); return; } for (let x = 1; x <= 8; x++) { pre.push(x); rec(pre); pre.pop(); } };
      rec([]);
      for (const h of heads) {
        const hs = h.reduce((a, b) => a + b, 0); const rest = L - hs, k = S - m;
        if (rest < k || (m === 3 && hs < D) || hs > 24) continue;
        const b = Math.floor(rest / k), r = rest % k;
        consider(h.concat(Array.from({ length: k }, (_, i) => b + (i >= k - r ? 1 : 0))));
        consider(h.concat(Array.from({ length: k }, (_, i) => b + (i < r ? 1 : 0))));
      }
    }
    return best.c;
  }

  function makePlan(cfgIn, cal) {
    const cfg = Object.assign({}, DEFAULTS, cfgIn);
    const [sp, tp] = cfg.mesh; const P = sp * tp; const S = cfg.stages;
    const chipsRep = 32 * cfg.galaxies / cfg.replicas;
    const errors = [];
    if (S * P > chipsRep + 1e-9) errors.push(`needs ${S * P} chips per replica, have ${chipsRep}`);
    if (!Number.isInteger(cfg.galaxies / cfg.replicas)) errors.push('galaxies must divide evenly into replicas');
    if (M3.Hkv % tp !== 0) errors.push('tp must divide 4 KV heads');
    const meshKey = `${sp}x${tp}`;
    const effMeas = cal.effs[meshKey] || cal.effs['2x4'];
    if (!cal.effs[meshKey]) errors.push(`no profile for mesh ${meshKey}; using [2,4] efficiencies`);
    const a = cfg.opEff;
    const eff = { moe: {}, dense: {} };
    for (const k of ['moe', 'dense']) for (const op in effMeas[k]) {
      let em = effMeas[k][op];
      if (k === 'moe' && cal.pipe.moeMult) em = em; // pipeline multiplier applied on the layer total below
      if (k === 'dense' && op === 'ring_c' && cal.pipe.ringC) em = cal.pipe.ringC * (effMeas.dense.ring_c / cal.effs['2x4'].dense.ring_c);
      if (k === 'dense' && op === 'ring_scan') em = cal.pipe.ringScan;
      const et = TARGET_EFF[OP_CLASS[op] || 'matmul'];
      eff[k][op] = Math.exp((1 - a) * Math.log(Math.min(em, et)) + a * Math.log(et));
    }
    const lat = { ccl: (1 - a) * LAT_MEAS.ccl + a * LAT_TGT.ccl, op: (1 - a) * LAT_MEAS.op + a * LAT_TGT.op };
    const idxB = cfg.idxBf16 ? BF16 : BF8;
    const ctxT = (T) => ({ sp, tp, P, T, idxB, imb: cfg.expertImb, varLayout: cfg.layout === 'var', bounded: cfg.boundedDense, msaLocal: cfg.msaLocal });
    const pm = cal.pipe;
    const moeMult = (T) => (1 - a) * (pm.moeMult[0] + pm.moeMult[1] * 5120 / T) + a * 1;
    const stageOv = (T) => (1 - a) * Math.max(0, pm.stageOv[0] + pm.stageOv[1] * T / 1000) + a * 0.3;
    const embedMs = (1 - a) * pm.embed + a * 0.3;
    const actBytes = (T) => 12 * (T / sp) * M3.E * BF16;
    const Tmax = cfg.layout === 'var' || cfg.batch ? Math.max(cfg.chunk, cfg.batch ? cfg.budget : cfg.chunk) : cfg.chunk;
    // handoff: measured = blocking send + hop latency (fitted, per chunk); async = link-rate transfer, overlapped
    const actXfer = (T) => T * M3.E * BF16 / (P * HW.linkUni * 0.5) * 1e3; // ms, each chip ships its shard
    const blockMs = (T) => (cfg.asyncHandoff ? 0 : Math.max(0, pm.block[0] + pm.block[1] * T / 1000));
    const hopMs = (T) => (cfg.asyncHandoff ? 0.5 + actXfer(T) : Math.max(0.5, pm.hop[0] + pm.hop[1] * T / 1000));
    const layer = (kind, T, segs) => {
      const c = ctxT(T);
      let tr = 0; for (const s of segs) tr += s.na === undefined ? s.n : s.na;
      c.Tr = tr;
      const t = layerMs(kind, c, segs, eff[kind], lat, cfg.attn);
      return kind === 'moe' ? t * moeMult(T) : t;
    };
    // representative-chunk objective for the auto split
    const repPts = [[60e3, 0.3], [140e3, 0.3], [310e3, 0.25], [550e3, 0.15]];
    const Trep = cfg.batch ? cfg.budget : cfg.chunk;
    const laneCap = cfg.cache === 'slots' ? cfg.slotLen : cfg.laneLen;
    const repCost = repPts.map(([kv, w]) => {
      const segs = [{ n: Trep, k: kv, cap: laneCap }];
      return { w, tm: layer('moe', Trep, segs), td: layer('dense', Trep, segs) };
    });
    const stageTimes = (counts, tm, td) => {
      const out = []; let st = 0;
      for (let s = 0; s < counts.length; s++) {
        const n = counts[s]; const nd = Math.max(0, Math.min(M3.nDense, st + n) - st); st += n;
        out.push(nd * td + (n - nd) * tm + stageOv(Trep) + (s === 0 ? embedMs : 0));
      }
      return out;
    };
    const splitCost = (counts) => repCost.reduce((acc, r) => acc + r.w * Math.max(...stageTimes(counts, r.tm, r.td)), 0);
    const counts = splitLayers(cfg, S, splitCost);
    if (counts.length !== S || counts.reduce((x, y) => x + y, 0) !== M3.L) errors.push('split must have one entry per stage summing to 60');
    // per-stage layer kinds + memory
    const stages = []; let st = 0;
    const kvbL = kvBytesPerTokenLayer(cfg, tp);
    const attnW = 110.9e6 * BF8, sharedW = 3 * M3.E * M3.Is * BF16, denseW = 3 * M3.E * M3.Id * BF16, expW = 3 * M3.E * M3.I * BF4;
    let capTok = Infinity;
    for (let s = 0; s < counts.length; s++) {
      const n = counts[s]; const nd = Math.max(0, Math.min(M3.nDense, st + n) - st); const nm = n - nd; st += n;
      let w = nm * (M3.Ex * expW + attnW * sp + sharedW * sp + M3.E * M3.Ex * BF16 * P) + nd * (attnW * sp + denseW * sp);
      if (s === 0) w += M3.V * M3.E * BF16;
      if (s === counts.length - 1) w += M3.V * M3.E * BF8;
      const free = P * (HW.dramCap - cfg.reserveGB * GB - actBytes(Tmax)) - w;
      const cap = free / (n * kvbL);
      capTok = Math.min(capTok, cap);
      stages.push({ n, nd, nm, weightsGBperChip: w / P / GB, capTok: cap });
    }
    capTok = Math.max(0, capTok);
    // cache layout
    let nSlots = 0, poolTok = 0, lanes = Infinity, arena = 0;
    if (cfg.cache === 'slots') { nSlots = Math.floor(capTok / cfg.slotLen); lanes = nSlots; }
    else if (cfg.cache === 'pool') {
      if (cfg.laneArena) { arena = Math.min(capTok, cfg.arenaTokens); poolTok = capTok - arena; }
      else { lanes = cfg.lanes; poolTok = capTok - cfg.lanes * cfg.laneLen; }
      if (poolTok < 0) errors.push('lanes do not fit in memory');
    } else if (cfg.cache === 'paging') poolTok = capTok;
    else poolTok = Infinity;
    if (cfg.cache === 'slots' && nSlots < 1) errors.push('no 1M slot fits in memory');
    const hostTok = cfg.hostTier && (cfg.cache === 'pool' || cfg.cache === 'paging') ? cfg.hostGBPerGalaxy * GB * (cfg.galaxies / cfg.replicas) / (M3.L * kvbL) : 0;
    const gran = 32 * sp;
    return {
      cfg, sp, tp, P, S, counts, stages, eff, lat, layer, stageOv, embedMs, blockMs, hopMs, errors,
      capTok, nSlots, poolTok, lanes, arena, hostTok, kvbL, gran, laneCap,
      pcieBps: cfg.pcieGBsPerGalaxy * GB * (cfg.galaxies / cfg.replicas),
      maxInflight: cfg.maxInflight > 0 ? cfg.maxInflight : 2 * S + 4,
      tokensPerSec: null,
    };
  }

  // stage times (ms) for one chunk; extra[s] adds per-stage copy costs
  function chunkStageMs(plan, T, segs, out) {
    const tm = plan.layer('moe', T, segs), td = plan.layer('dense', T, segs);
    const ov = plan.stageOv(T);
    for (let s = 0; s < plan.S; s++) {
      const st = plan.stages[s];
      out[s] = st.nm * tm + st.nd * td + ov + (s === 0 ? plan.embedMs : 0);
    }
    return out;
  }

  // ------------------------------------------------------------------------------------------------------
  // Traffic (binary written by prep_traffic.py)
  // ------------------------------------------------------------------------------------------------------
  function loadTraffic(buf, header) {
    const A = {};
    const ctor = { u1: Uint8Array, u2: Uint16Array, u4: Uint32Array, i4: Int32Array, f4: Float32Array };
    for (const L of header.layout) A[L.name] = new ctor[L.dtype](buf, L.offset, L.n);
    const nT = A.tr_req0.length, nR = A.req_blocks.length, nS = A.st_kind.length, nP = A.pc_len.length;
    // piece depth (for LRU tie-break) and per-trace piece counts
    const pdepth = new Int32Array(nP);
    for (let q = 0; q < nP; q++) pdepth[q] = A.pc_parent[q] < 0 ? 0 : pdepth[A.pc_parent[q]] + 1;
    const trPieces = new Int32Array(nT), trReqs = new Int32Array(nT), trStreams = new Int32Array(nT);
    for (let t = 0; t < nT; t++) {
      trPieces[t] = (t + 1 < nT ? A.tr_pc0[t + 1] : nP) - A.tr_pc0[t];
      trReqs[t] = (t + 1 < nT ? A.tr_req0[t + 1] : nR) - A.tr_req0[t];
      trStreams[t] = (t + 1 < nT ? A.tr_st0[t + 1] : nS) - A.tr_st0[t];
    }
    // spawn lists: main request -> subagent streams it spawns
    const spawnHead = new Int32Array(nR).fill(-1), spawnNext = new Int32Array(nS).fill(-1);
    for (let s = nS - 1; s >= 0; s--) if (A.st_kind[s] === 1 && A.st_spawn[s] >= 0) { spawnNext[s] = spawnHead[A.st_spawn[s]]; spawnHead[A.st_spawn[s]] = s; }
    return Object.assign(A, { nT, nR, nS, nP, pdepth, trPieces, trReqs, trStreams, spawnHead, spawnNext, stats: header.stats });
  }

  // ------------------------------------------------------------------------------------------------------
  // Utilities: RNG, heaps
  // ------------------------------------------------------------------------------------------------------
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  class EvHeap { // (time, seq) ordered events with payload objects
    constructor() { this.t = []; this.p = []; }
    get size() { return this.t.length; }
    push(t, p) {
      const T = this.t, Pp = this.p; let i = T.length; T.push(t); Pp.push(p);
      while (i > 0) { const j = (i - 1) >> 1; if (T[j] <= t) break; T[i] = T[j]; Pp[i] = Pp[j]; i = j; }
      T[i] = t; Pp[i] = p;
    }
    peekT() { return this.t[0]; }
    pop() {
      const T = this.t, Pp = this.p; const top = Pp[0]; const lt = T.pop(), lp = Pp.pop(); const n = T.length;
      if (n > 0) {
        let i = 0;
        while (true) { let l = 2 * i + 1; if (l >= n) break; const r = l + 1; if (r < n && T[r] < T[l]) l = r; if (T[l] >= lt) break; T[i] = T[l]; Pp[i] = Pp[l]; i = l; }
        T[i] = lt; Pp[i] = lp;
      }
      return top;
    }
    shiftAll(dt) { for (let i = 0; i < this.t.length; i++) this.t[i] -= dt; }
  }

  class LruHeap { // min-heap on (t, -depth) over piece keys, lazy invalidation
    constructor() { this.t = new Float64Array(1024); this.d = new Int32Array(1024); this.k = new Int32Array(1024); this.n = 0; }
    less(i, j) { return this.t[i] < this.t[j] || (this.t[i] === this.t[j] && this.d[i] > this.d[j]); }
    push(t, d, k) {
      if (this.n === this.t.length) { const g = (A, C) => { const B = new C(A.length * 2); B.set(A); return B; }; this.t = g(this.t, Float64Array); this.d = g(this.d, Int32Array); this.k = g(this.k, Int32Array); }
      let i = this.n++; this.t[i] = t; this.d[i] = d; this.k[i] = k;
      while (i > 0) { const j = (i - 1) >> 1; if (!this.less(i, j)) break; this.swap(i, j); i = j; }
    }
    swap(i, j) { let x = this.t[i]; this.t[i] = this.t[j]; this.t[j] = x; x = this.d[i]; this.d[i] = this.d[j]; this.d[j] = x; x = this.k[i]; this.k[i] = this.k[j]; this.k[j] = x; }
    pop() { // returns index 0 content after removal via out fields
      const t = this.t[0], d = this.d[0], k = this.k[0]; this.n--;
      if (this.n > 0) { this.t[0] = this.t[this.n]; this.d[0] = this.d[this.n]; this.k[0] = this.k[this.n]; let i = 0;
        while (true) { let l = 2 * i + 1; if (l >= this.n) break; const r = l + 1; if (r < this.n && this.less(r, l)) l = r; if (!this.less(l, i)) break; this.swap(i, l); i = l; } }
      this.ot = t; this.ok = k; return k;
    }
  }

  // ------------------------------------------------------------------------------------------------------
  // KV residency
  // ------------------------------------------------------------------------------------------------------
  // Content-addressed paged pool over prefix-tree pieces with exact LRU (piece granularity == 64-token pages).
  // Optional second tier (host DRAM): device evictions demote to host; host evictions drop.
  class PoolCache {
    constructor(TR, capBlocks, hostBlocks) {
      this.TR = TR; this.cap = capBlocks; this.hcap = hostBlocks;
      this.n = 0; this.tier = new Uint8Array(1 << 16); this.t = new Float64Array(1 << 16); this.pk = new Int32Array(1 << 16);
      this.len = new Int32Array(1 << 16); this.dep = new Int32Array(1 << 16);
      this.used = 0; this.hused = 0; this.heap = new LruHeap(); this.hheap = new LruHeap(); this.path = new Int32Array(1024);
      this.evictedBlocks = 0;
    }
    allocNs(trace) {
      const TR = this.TR, np = TR.trPieces[trace], p0 = TR.tr_pc0[trace];
      const base = this.n; this.n += np;
      if (this.n > this.tier.length) {
        let L = this.tier.length; while (L < this.n) L *= 2;
        const g = (A, C) => { const B = new C(L); B.set(A); return B; };
        this.tier = g(this.tier, Uint8Array); this.t = g(this.t, Float64Array); this.pk = g(this.pk, Int32Array); this.len = g(this.len, Int32Array); this.dep = g(this.dep, Int32Array);
      }
      for (let i = 0; i < np; i++) {
        const q = p0 + i; const par = TR.pc_parent[q];
        this.pk[base + i] = par < 0 ? -1 : base + (par - p0); this.len[base + i] = TR.pc_len[q]; this.dep[base + i] = TR.pdepth[q];
      }
      return base;
    }
    walk(ns, leafLocal) { // fill this.path leaf -> root, return length
      let k = ns + leafLocal, n = 0;
      while (k >= 0) { if (n === this.path.length) { const B = new Int32Array(n * 2); B.set(this.path); this.path = B; } this.path[n++] = k; k = this.pk[k]; }
      return n;
    }
    hit(n) { // resident prefix from root: [device blocks, host blocks]
      let dev = 0, host = 0, i = n - 1;
      for (; i >= 0; i--) { const k = this.path[i]; if (this.tier[k] !== 1) break; dev += this.len[k]; }
      for (; i >= 0; i--) { const k = this.path[i]; if (this.tier[k] === 0) break; host += this.len[k]; }
      this.hDev = dev; this.hHost = host; return dev + host;
    }
    touch(n, now) {
      for (let i = 0; i < n; i++) {
        const k = this.path[i];
        if (this.tier[k] === 0) this.used += this.len[k];
        else if (this.tier[k] === 2) { this.hused -= this.len[k]; this.used += this.len[k]; }
        this.tier[k] = 1; this.t[k] = now; this.heap.push(now, this.dep[k], k);
      }
      this.evict();
    }
    refresh(n, now) {
      for (let i = 0; i < n; i++) { const k = this.path[i]; if (this.tier[k] === 1) { this.t[k] = now; this.heap.push(now, this.dep[k], k); } }
    }
    evict() {
      while (this.used > this.cap && this.heap.n > 0) {
        const k = this.heap.pop(); const t = this.heap.ot;
        if (this.tier[k] !== 1 || this.t[k] !== t) continue;
        this.used -= this.len[k]; this.evictedBlocks += this.len[k];
        if (this.hcap > 0) { this.tier[k] = 2; this.hused += this.len[k]; this.hheap.push(t, this.dep[k], k); if (this.onDemote) this.onDemote(this.len[k]); }
        else this.tier[k] = 0;
      }
      while (this.hused > this.hcap && this.hheap.n > 0) {
        const k = this.hheap.pop(); const t = this.hheap.ot;
        if (this.tier[k] !== 2 || this.t[k] !== t) continue;
        this.hused -= this.len[k]; this.tier[k] = 0;
      }
      if (this.heap.n > 4 * (this.n + 1024)) this.rebuild(this.heap, 1);
      if (this.hheap.n > 4 * (this.n + 1024)) this.rebuild(this.hheap, 2);
    }
    rebuild(h, tier) {
      const t = h.t.slice(0, h.n), k = h.k.slice(0, h.n); h.n = 0;
      for (let i = 0; i < t.length; i++) if (this.tier[k[i]] === tier && this.t[k[i]] === t[i]) h.push(t[i], this.dep[k[i]], k[i]);
    }
  }

  // Static slots: a slot holds one stream's latest request; LRU over idle slots; busy slots cannot be evicted.
  class SlotCache {
    constructor(N) { this.N = N; this.owner = new Int32Array(N).fill(-1); this.last = new Float64Array(N); this.busy = new Uint8Array(N); this.of = new Map(); this.evictions = 0; }
    acquire(streamKey, now) { // -> {slot, warm} or null
      let s = this.of.get(streamKey);
      if (s !== undefined) { if (this.busy[s]) return null; this.busy[s] = 1; return { slot: s, warm: true }; }
      let pick = -1, lru = Infinity;
      for (let i = 0; i < this.N; i++) {
        if (this.busy[i]) continue;
        if (this.owner[i] < 0) { pick = i; break; }
        if (this.last[i] < lru) { lru = this.last[i]; pick = i; }
      }
      if (pick < 0) return null;
      if (this.owner[pick] >= 0) { this.of.delete(this.owner[pick]); this.evictions++; }
      this.owner[pick] = streamKey; this.of.set(streamKey, pick); this.busy[pick] = 1;
      return { slot: pick, warm: false };
    }
    release(s, now) { this.busy[s] = 0; this.last[s] = now; }
  }

  // ------------------------------------------------------------------------------------------------------
  // The replay simulation
  // ------------------------------------------------------------------------------------------------------
  const EV_READY = 1, EV_DONE = 2, EV_END = 3, EV_PUMP = 4, EV_LANE = 5, EV_FETCHED = 6;

  function simulate(TR, cal, cfgIn, opts) {
    opts = opts || {};
    const plan = makePlan(cfgIn, cal);
    const cfg = plan.cfg;
    if (plan.errors.some((e) => !e.startsWith('no profile'))) return { plan, error: plan.errors.join('; ') };
    const rnd = mulberry32(cfg.seed * 2654435761 >>> 0);
    const S = plan.S, B = 64;
    const ev = new EvHeap();
    let now = 0, t0 = null, tEnd = Infinity, primersLeft = 0, warmDone = false;
    let inflightReqs = 0; // queued + prefill + decode (system idle check)
    // replicas
    const reps = [];
    for (let r = 0; r < cfg.replicas; r++) {
      reps.push({
        id: r, free: new Float64Array(S), busy: new Float64Array(S), queue: [], active: [], exits: [], pumpAt: -1,
        trees: 0, lanesUsed: 0, arenaUsed: 0,
        pool: cfg.cache === 'slots' ? null : new PoolCache(TR, plan.poolTok / B, plan.hostTok / B),
        slots: cfg.cache === 'slots' ? new SlotCache(plan.nSlots) : null, pcieFree: 0,
      });
    }
    // device -> host demotions (write-back) occupy the same PCIe link as host -> device fetches
    for (const rp of reps) if (rp.pool && plan.hostTok > 0) {
      rp.pool.onDemote = (blocks) => { rp.pcieFree = Math.max(rp.pcieFree, now) + blocks * B * M3.L * plan.kvbL / plan.pcieBps; };
    }
    // stats
    const st = { done: 0, useful: 0, newTok: 0, processed: 0, hitTok: 0, inTok: 0, infHitTok: 0, hostTok: 0, reprefill: 0, alignLoss: 0,
      ttft: [], chunks: 0, segs: 0, laneWait: 0, reqs: 0, primers: 0, warmupS: 0, idleWarps: 0, primerTok: 0 };
    let nextTrace = 0; const nsKey = { v: 0 };
    const scratch = new Float64Array(S);

    // ---------- trees (one per concurrency lane)
    function newTree(lane, trace, tstar, now) {
      const rep = reps.reduce((a, b) => (b.trees < a.trees ? b : a), reps[0]);
      rep.trees++;
      const tree = { lane, trace, rep, ns: rep.pool ? rep.pool.allocNs(trace) : 0, key: nsKey.v++, live: 0, waiting: new Map(), pend: new Map(), streamsLeft: 0 };
      const s0 = TR.tr_st0[trace], ns = TR.trStreams[trace];
      tree.s0 = s0;
      const mainFirst = TR.st_first[s0], mainCnt = TR.st_cnt[s0];
      if (tstar === null) { // recycled: start at turn 0 with fresh join counters
        tree.streamsLeft = ns;
        initPend(tree);
        dispatch(tree, mainFirst, now);
        return tree;
      }
      // initial tree: split every stream at t*
      let cur = mainFirst; while (cur < mainFirst + mainCnt && TR.req_t[cur] < tstar) cur++;
      const mainIdx = cur - mainFirst;
      const prof = []; // [req, offset]
      if (cur - 1 >= mainFirst) primer(tree, cur - 1);
      tree.streamsLeft = 1;
      const mainNext = cur < mainFirst + mainCnt ? cur : -1;
      for (let s = s0 + 1; s < s0 + ns; s++) {
        const sp = TR.st_spawn[s] - mainFirst, first = TR.st_first[s], cnt = TR.st_cnt[s];
        if (sp >= mainIdx) { // spawned later by a main turn
          tree.streamsLeft++;
          const j = TR.st_join[s]; if (j >= 0) tree.pend.set(j, (tree.pend.get(j) || 0) + 1);
          continue;
        }
        if (TR.st_t1[s] <= tstar) continue;                  // finished before t*
        let c = first; while (c < first + cnt && TR.req_t[c] < tstar) c++;
        if (c >= first + cnt) continue;                       // only an in-flight request at t*: skip
        tree.streamsLeft++;
        if (c - 1 >= first) primer(tree, c - 1);
        prof.push([c, Math.max(0, TR.req_t[c] - tstar)]);
        const j = TR.st_join[s]; if (j >= 0 && j >= cur) tree.pend.set(j, (tree.pend.get(j) || 0) + 1);
      }
      if (mainNext >= 0) prof.push([mainNext, Math.max(0, TR.req_t[mainNext] - tstar), true]);
      else tree.streamsLeft--; // main already done
      tree.prof = prof;
      if (tree.streamsLeft === 0) tree.dead = true;
      return tree;
    }
    function primer(tree, r) {
      primersLeft++; st.primers++;
      const q = mkReq(tree, r, true); tree.live++; ev.push(0, { e: EV_READY, q });
    }
    function mkReq(tree, r, isPrimer) {
      return { r, tree, primer: isPrimer, tReady: 0, tStart: -1, tDone: 0, hit: 0, host: 0, rem: 0, pos: 0, lane: -1, slot: -1, started: false, fetchedAt: 0 };
    }
    function dispatch(tree, r, t, isJoinImmediate) {
      const q = mkReq(tree, r, false); tree.live++; ev.push(t, { e: EV_READY, q });
    }
    function startProfiling(t) {
      t0 = t; tEnd = t + cfg.duration; warmDone = true; st.warmupS = t;
      for (const tr of trees) {
        if (!tr.prof) continue;
        for (const [r, off, isMain] of tr.prof) {
          if (isMain && (tr.pend.get(r) || 0) > 0) { tr.waiting.set(r, true); continue; }
          dispatch(tr, r, t + off);
        }
        tr.prof = null;
        if (tr.live === 0) recycle(tr, t);
      }
    }
    function recycle(tree, t) {
      tree.rep.trees--;
      const trace = nextTrace++ % TR.nT;
      trees[tree.lane] = newTree(tree.lane, trace, null, t);
    }
    // ---------- request lifecycle
    function onReady(q) {
      const tree = q.tree, rep = tree.rep, r = q.r;
      q.tReady = now; inflightReqs++;
      const blocks = TR.req_blocks[r];
      // overlap-path subagents are dispatched when the spawning main turn is issued
      if (!q.primer && TR.req_stream[r] === tree.s0) {
        for (let s = TR.spawnHead[r]; s >= 0; s = TR.spawnNext[s]) if (TR.st_ovl[s]) dispatch(tree, TR.st_first[s], now + TR.st_off[s]);
      }
      if (rep.pool) {
        const pool = rep.pool; const n = pool.walk(tree.ns, TR.req_leaf[r] - TR.tr_pc0[tree.trace]);
        pool.hit(n);
        let dev = pool.hDev, host = pool.hHost;
        pool.refresh(n, now); // pin: LRU-refresh the resident part of the path while the request is queued
        q.hitRaw = dev + host; q.host = host;
        if (host > 0 && plan.hostTok > 0) { // fetch host part over PCIe before it can be admitted
          const bytes = host * B * M3.L * plan.kvbL;
          const t = Math.max(now, rep.pcieFree) + bytes / plan.pcieBps; rep.pcieFree = t; q.fetchedAt = t;
          st.hostTok += host * B;
          ev.push(t, { e: EV_FETCHED, q }); return;
        }
      }
      enqueue(q);
    }
    function enqueue(q) {
      const rep = q.tree.rep;
      rep.queue.push(q);
      pump(rep);
    }
    function hitTokens(q, rawBlocks) {
      let h = rawBlocks * B;
      const tot = TR.req_blocks[q.r] * B;
      if (h >= tot) h = tot - B; // always recompute the last block (logits)
      if (cfg.layout === 'fixed' && !cfg.unaligned) { const a = Math.floor(h / cfg.chunk) * cfg.chunk; st.alignLoss += q.primer || !warmDone ? 0 : h - a; h = a; }
      return Math.max(0, h);
    }
    function tryStart(q, rep) { // acquire lane/slot, compute hit; false if blocked
      const tree = q.tree, r = q.r;
      if (rep.slots) {
        const streamKey = tree.key * 4096 + (TR.req_stream[r] - tree.s0);
        const a = rep.slots.acquire(streamKey, now); if (!a) return false;
        q.slot = a.slot; q.laneCap = cfg.slotLen;
        const lp = a.warm ? TR.req_lcp_prev[r] : 0;
        q.hitTok = hitTokens(q, lp);
      } else {
        if (plan.lanes !== Infinity && rep.lanesUsed >= plan.lanes) return false;
        const need = TR.req_blocks[r] * B;
        if (cfg.laneArena && cfg.cache === 'pool') { if (rep.arenaUsed + need > plan.arena) return false; rep.arenaUsed += need; q.arena = need; }
        if (plan.lanes !== Infinity) rep.lanesUsed++;
        q.lane = 1;
        // dense ring-joint scans the whole lane: fixed lanes are laneLen; arena lanes / paged kernels are request-sized
        q.laneCap = cfg.cache === 'pool' && !cfg.laneArena ? cfg.laneLen : need;
        q.hitTok = hitTokens(q, q.hitRaw);
      }
      q.started = true; q.tStart = now;
      q.pos = q.hitTok; q.rem = TR.req_blocks[r] * B - q.hitTok; q.first = true;
      st.laneWait += now - (q.fetchedAt || q.tReady);
      return true;
    }
    function queueOrder(rep) {
      if (cfg.policy === 'srpt' && rep.queue.length > 1) {
        const mw = cfg.srptMaxWait;
        rep.queue.sort((a, b) => {
          const aw = now - a.tReady > mw, bw = now - b.tReady > mw;
          if (aw !== bw) return aw ? -1 : 1;
          if (aw) return a.tReady - b.tReady;
          return (TR.req_blocks[a.r] - (a.hitRaw || 0)) - (TR.req_blocks[b.r] - (b.hitRaw || 0));
        });
      }
    }
    // pull segments into one chunk
    function formChunk(rep) {
      const segs = []; let T = 0; const C = cfg.chunk;
      const budget = cfg.batch ? Math.max(cfg.budget, C) : C;
      const cands = rep.active;
      const take = (q, maxTok) => { // one segment of q, <= maxTok padded tokens
        let n, npad;
        if (cfg.layout === 'fixed') { if (maxTok < C) return false; n = Math.min(q.rem, C); npad = C; }
        else { const cap = cfg.batch ? maxTok : Math.min(maxTok, C); if (cap < plan.gran) return false; n = Math.min(q.rem, cap - (cap % plan.gran)); if (n <= 0) return false; npad = Math.ceil(n / plan.gran) * plan.gran; }
        segs.push({ q, n, npad, k: q.pos, first: q.first, last: n === q.rem });
        q.first = false; q.pos += n; q.rem -= n; T += npad;
        return true;
      };
      // 1) continue active requests (they hold lanes), in start order
      for (let i = 0; i < cands.length && T < budget; i++) {
        const q = cands[i];
        while (q.rem > 0 && T < budget) { if (!take(q, budget - T)) break; if (!cfg.batch) break; }
        if (!cfg.batch && segs.length) break;
      }
      // 2) start new requests from the queue
      if (T < budget && (cfg.batch || segs.length === 0)) {
        queueOrder(rep);
        let i = 0;
        while (i < rep.queue.length && T < budget) {
          const q = rep.queue[i];
          if (!tryStart(q, rep)) { if (rep.slots) { i++; continue; } break; }
          rep.queue.splice(i, 1); rep.active.push(q);
          while (q.rem > 0 && T < budget) { if (!take(q, budget - T)) break; if (!cfg.batch) break; }
          if (!cfg.batch) break;
        }
      }
      rep.active = rep.active.filter((q) => q.rem > 0);
      return segs.length ? { segs, T } : null;
    }
    function pump(rep) {
      while (true) {
        while (rep.exits.length && rep.exits[0] <= now + 1e-12) heapPopNum(rep.exits);
        if (rep.free[0] > now + 1e-12) { schedPump(rep, rep.free[0]); return; }
        if (rep.exits.length >= plan.maxInflight) { schedPump(rep, rep.exits[0]); return; }
        if (!rep.active.length && !rep.queue.length) return;
        const ch = formChunk(rep);
        if (!ch) return; // blocked on lanes: a lane release re-pumps
        runChunk(rep, ch);
      }
    }
    function schedPump(rep, t) { if (rep.pumpAt >= 0 && rep.pumpAt <= t + 1e-12 && rep.pumpAt >= now) return; rep.pumpAt = t; ev.push(t, { e: EV_PUMP, rep }); }
    function runChunk(rep, ch) {
      const { segs, T } = ch;
      const cs = segs.map((s) => ({ n: s.npad, na: s.n, k: s.k, cap: s.q.laneCap || plan.laneCap }));
      chunkStageMs(plan, T, cs, scratch);
      // pool copy-in (first segment of a request) / copy-out (last) per stage, DRAM bound
      if (rep.pool && plan.poolTok !== Infinity && cfg.cache === 'pool') {
        let copyTok = 0;
        for (const s of segs) { if (s.first) copyTok += s.q.hitTok; if (s.last) copyTok += TR.req_blocks[s.q.r] * B - s.q.hitTok; }
        if (copyTok > 0) {
          for (let s = 0; s < S; s++) {
            const bytes = copyTok * plan.stages[s].n * plan.kvbL / plan.P * 2;
            const ms = bytes / (HW.dram * 0.5) * 1e3;
            scratch[s] += cfg.copyOverlap ? ms * cfg.copyContention : ms;
          }
        }
      }
      const blk = plan.blockMs(T) / 1e3, hop = plan.hopMs(T) / 1e3;
      let arr = now, end = now, end0 = now;
      for (let s = 0; s < S; s++) {
        const dt = scratch[s] / 1e3;
        const start = Math.max(rep.free[s], arr); end = start + dt;
        rep.free[s] = end + blk; if (warmDone && end > t0 && start < tEnd) rep.busy[s] += dt;
        if (s === 0) end0 = end;
        arr = end + hop;
      }
      heapPushNum(rep.exits, end);
      if (warmDone) { st.chunks++; st.segs += segs.length; st.processed += T; }
      for (const s of segs) {
        if (s.last) {
          ev.push(end, { e: EV_DONE, q: s.q });
          if (cfg.cache !== 'slots' && cfg.laneScope === 'stage') ev.push(end0, { e: EV_LANE, q: s.q, rep });
        }
      }
      schedPump(rep, rep.free[0]);
    }
    function releaseLane(q, rep) {
      if (q.lane < 0) return;
      if (plan.lanes !== Infinity) rep.lanesUsed--;
      if (q.arena) { rep.arenaUsed -= q.arena; q.arena = 0; }
      q.lane = -1; pump(rep);
    }
    function onDone(q) { // prefill complete (TTFT)
      const tree = q.tree, rep = tree.rep, r = q.r;
      q.tDone = now;
      if (rep.slots) { rep.slots.release(q.slot, now); pump(rep); }
      else {
        if (cfg.laneScope !== 'stage') releaseLane(q, rep);
        const pool = rep.pool; const n = pool.walk(tree.ns, TR.req_leaf[r] - TR.tr_pc0[tree.trace]); pool.touch(n, now);
      }
      const inTok = TR.req_blocks[r] * B;
      if (q.primer) { st.primerTok += inTok - q.hitTok; primersLeft--; if (primersLeft === 0 && !warmDone) startProfiling(now); ev.push(now, { e: EV_END, q }); return; }
      if (warmDone && now >= t0 && now <= tEnd) {
        const useful = (TR.req_blocks[r] - TR.req_lcp_best[r]) * B;
        st.done++; st.useful += useful; st.newTok += inTok - q.hitTok; st.hitTok += q.hitTok; st.inTok += inTok;
        st.infHitTok += TR.req_lcp_best[r] * B; st.reprefill += Math.max(0, (inTok - q.hitTok) - useful);
        st.ttft.push(now - q.tReady);
      }
      ev.push(now + TR.req_out[r] / cfg.decodeTps, { e: EV_END, q });
    }
    function onEnd(q) {
      const tree = q.tree, r = q.r;
      inflightReqs--; tree.live--;
      if (!q.primer) {
        const s = TR.req_stream[r], first = TR.st_first[s], cnt = TR.st_cnt[s];
        if (s === tree.s0) {
          for (let x = TR.spawnHead[r]; x >= 0; x = TR.spawnNext[x]) if (!TR.st_ovl[x]) dispatch(tree, TR.st_first[x], now + TR.st_off[x]);
          if (r + 1 < first + cnt) {
            if ((tree.pend.get(r + 1) || 0) > 0) tree.waiting.set(r + 1, true);
            else dispatch(tree, r + 1, now + Math.min(cfg.gapCap, TR.req_delay[r + 1]));
          } else tree.streamsLeft--;
        } else {
          if (r + 1 < first + cnt) dispatch(tree, r + 1, now + Math.min(cfg.gapCap, TR.req_delay[r + 1]));
          else {
            tree.streamsLeft--;
            const j = TR.st_join[s];
            if (j >= 0) {
              const left = (tree.pend.get(j) || 0) - 1; tree.pend.set(j, left);
              if (left <= 0 && tree.waiting.get(j)) { tree.waiting.delete(j); dispatch(tree, j, now); }
            }
          }
        }
      }
      if (tree.live === 0 && warmDone && !tree.prof && trees[tree.lane] === tree) recycle(tree, now);
    }
    function initPend(tree) {
      const s0 = tree.s0, ns = TR.trStreams[tree.trace];
      for (let s = s0 + 1; s < s0 + ns; s++) { const j = TR.st_join[s]; if (j >= 0) tree.pend.set(j, (tree.pend.get(j) || 0) + 1); }
    }
    // ---------- start
    const trees = [];
    for (let l = 0; l < cfg.concurrency; l++) {
      const trace = nextTrace++ % TR.nT;
      const tstar = (cfg.startMin + (cfg.startMax - cfg.startMin) * rnd()) * TR.tr_dur[trace];
      trees.push(newTree(l, trace, tstar, 0));
    }
    if (primersLeft === 0) startProfiling(0);
    // ---------- event loop
    let evCount = 0;
    const maxEv = opts.maxEvents || 5e7;
    while (ev.size) {
      let t = ev.peekT();
      if (warmDone && inflightReqs === 0 && t > now + cfg.idleCap) { ev.shiftAll(t - now - cfg.idleCap); st.idleWarps++; t = ev.peekT(); }
      if (t > tEnd) break;
      if (!warmDone && t > cfg.maxWarmup) { st.warmupTimeout = true; break; }
      now = Math.max(now, t);
      const p = ev.pop(); evCount++;
      if (evCount > maxEv) { st.eventCap = true; break; }
      switch (p.e) {
        case EV_READY: onReady(p.q); break;
        case EV_FETCHED: enqueue(p.q); break;
        case EV_DONE: onDone(p.q); break;
        case EV_END: onEnd(p.q); break;
        case EV_PUMP: if (p.rep.pumpAt === t) p.rep.pumpAt = -1; pump(p.rep); break;
        case EV_LANE: releaseLane(p.q, p.rep); break;
      }
    }
    // ---------- results
    const D = warmDone ? Math.min(cfg.duration, Math.max(1e-9, now - t0)) : 1;
    const tt = Float64Array.from(st.ttft).sort();
    const pct = (p) => (tt.length ? tt[Math.min(tt.length - 1, Math.floor(p * (tt.length - 1)))] : NaN);
    const util = []; for (let s = 0; s < S; s++) { let u = 0; for (const rp of reps) u += rp.busy[s]; util.push(u / (D * reps.length)); }
    return {
      plan: planSummary(plan), cfg,
      usefulTps: st.useful / D, processedTps: st.processed / D, newTps: st.newTok / D, reqPerS: st.done / D,
      ttftP50: pct(0.5), ttftP90: pct(0.9), ttftP99: pct(0.99), ttftMean: tt.length ? tt.reduce((a, b) => a + b, 0) / tt.length : NaN,
      hitRate: st.inTok ? st.hitTok / st.inTok : NaN, infHitRate: st.inTok ? st.infHitTok / st.inTok : NaN,
      reprefillFrac: st.newTok ? st.reprefill / st.newTok : 0, padFrac: st.processed ? 1 - st.newTok / st.processed : 0,
      alignLossFrac: st.inTok ? st.alignLoss / st.inTok : 0,
      avgChunkTok: st.chunks ? st.processed / st.chunks : 0, avgSegsPerChunk: st.chunks ? st.segs / st.chunks : 0,
      stageUtil: util, maxUtil: Math.max(...util), done: st.done, warmupS: st.warmupS, primers: st.primers, primerTok: st.primerTok,
      laneWaitMean: st.done ? st.laneWait / Math.max(1, st.done) : 0, hostTok: st.hostTok, idleWarps: st.idleWarps,
      warmupTimeout: !!st.warmupTimeout, eventCap: !!st.eventCap, events: evCount, duration: D,
      slotEvictions: reps.reduce((a, r) => a + (r.slots ? r.slots.evictions : 0), 0),
    };
  }
  function heapPushNum(h, x) { let i = h.length; h.push(x); while (i > 0) { const j = (i - 1) >> 1; if (h[j] <= x) break; h[i] = h[j]; i = j; } h[i] = x; }
  function heapPopNum(h) { const top = h[0], last = h.pop(), n = h.length; if (n) { let i = 0; while (true) { let l = 2 * i + 1; if (l >= n) break; if (l + 1 < n && h[l + 1] < h[l]) l++; if (h[l] >= last) break; h[i] = h[l]; i = l; } h[i] = last; } return top; }

  function planSummary(plan) {
    return {
      S: plan.S, mesh: [plan.sp, plan.tp], counts: plan.counts, capTok: plan.capTok, nSlots: plan.nSlots, poolTok: plan.poolTok,
      lanes: plan.lanes, arena: plan.arena, hostTok: plan.hostTok, errors: plan.errors, kvbL: plan.kvbL,
      weightsGBperChip: plan.stages.map((s) => +s.weightsGBperChip.toFixed(2)),
    };
  }

  // ------------------------------------------------------------------------------------------------------
  // Matrix-cell replay (validation vs the #57827 tables): U users stream (cached, new) requests
  // ------------------------------------------------------------------------------------------------------
  function matrixCell(cal, cfgIn, cached, nnew, users, reqsPerUser) {
    const plan = makePlan(Object.assign({ cache: 'inf', batch: false, layout: 'fixed' }, cfgIn), cal);
    const S = plan.S, C = plan.cfg.chunk, out = new Float64Array(S);
    const cap = cached + 51200;
    const cachedA = Math.floor(cached / C) * C;
    const nCh = Math.max(1, Math.ceil(nnew / C));
    const push = (free, j, arrT) => {
      const na = Math.min(C, nnew - j * C);
      chunkStageMs(plan, C, [{ n: C, na, k: cachedA + j * C, cap }], out);
      const blk = plan.blockMs(C), hop = plan.hopMs(C);
      let arr = arrT, end = arrT;
      for (let s = 0; s < S; s++) { const st = Math.max(free[s], arr); end = st + out[s]; free[s] = end + blk; arr = end + hop; }
      return end;
    };
    // idle TTFT: one request on an empty pipeline (h2d ~1 ms)
    let free = new Float64Array(S); let e = 0;
    for (let j = 0; j < nCh; j++) e = push(free, j, 1);
    const idle = e;
    // loaded: the producer streams all users' requests back to back, round-robin at chunk granularity
    free = new Float64Array(S);
    const exits = []; let tNow = 0;
    const total = users * reqsPerUser * nCh;
    for (let i = 0; i < total; i++) {
      const j = Math.floor(i / users) % nCh;
      const arrive = Math.max(tNow, free[0]);
      const x = push(free, j, arrive); exits.push(x); tNow = arrive;
    }
    exits.sort((a, b) => a - b);
    const drop = Math.min(16, Math.floor(exits.length / 4));
    const span = exits[exits.length - 1 - drop] - exits[drop];
    const period = span / (exits.length - 1 - 2 * drop);
    const processed = C / period * 1e3;
    return { idleTtftMs: idle, periodMs: period, processedTps: processed, newTps: processed * nnew / (nCh * C) };
  }

  // Fit handoff block/hop against the measured tables (needs the pipeline fit first)
  function fitHandoff(cal, data) {
    // single-chunk cells: measured loaded chunk period - modelled bottleneck stage = blocking handoff per chunk;
    // (idle TTFT - sum of stages) / 16 = per-hop latency
    const pts = { 5120: [], 2048: [] };
    for (const run of ['A', 'B', 'C']) {
      const r = data.pipeline[run]; const T = r.chunk; const counts = r.layers.split(',').map(Number);
      const plan = makePlan({ chunk: T, split: counts, stages: 16, cache: 'inf' }, cal);
      for (const row of data.tables[run]) {
        if (row.new > T || row.loaded_steady_processed_tps == null) continue;
        const cell = (r.cells || []).find((c) => c.cached === row.cached && c.new === row.new);
        const periodMeas = cell && cell.period_ms ? cell.period_ms : T / row.loaded_steady_processed_tps * 1e3;
        const o = new Float64Array(16);
        chunkStageMs(plan, T, [{ n: T, na: row.new, k: Math.floor(row.cached / T) * T, cap: row.cached + 51200 }], o);
        const sum = o.reduce((a, b) => a + b, 0), mx = Math.max(...o);
        pts[T].push({ block: periodMeas - mx, hop: (row.idle_ttft_ms_median - sum) / 16 });
      }
    }
    const med = (a, k) => { const v = a.map((p) => p[k]).sort((x, y) => x - y); return v[Math.floor(v.length / 2)]; };
    const b5 = Math.max(0, med(pts[5120], 'block')), b2 = Math.max(0, med(pts[2048], 'block'));
    const h5 = Math.max(0, med(pts[5120], 'hop')), h2 = Math.max(0, med(pts[2048], 'hop'));
    const s = (y5, y2) => { const b1 = (y5 - y2) / ((5120 - 2048) / 1000); return [y5 - b1 * 5.12, b1]; };
    cal.pipe.block = s(b5, b2); cal.pipe.hop = s(h5, h2);
    cal.fit.block = { b5, b2 }; cal.fit.hop = { h5, h2 };
  }

  function calibrateAll(data) { const cal = calibrate(data); fitHandoff(cal, data); return cal; }

  const API = { M3, HW, DEFAULTS, calibrate: calibrateAll, makePlan, chunkStageMs, loadTraffic, simulate, matrixCell, planSummary, layerMs, roofTok, roofSeg };
  if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.M3SIM = API;
})(typeof self !== 'undefined' ? self : this);
