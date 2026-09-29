#!/usr/bin/env node
// Per-op roofline ms for one MiniMax-M3 prefill layer, using Pavlo's traffic-sim cost model (sim_core.js, copied
// verbatim from branch philei/m3-traffic-sim). Op names are the sim's: roofTok/roofSeg ops plus the dense ring split.
//
//   node roofline_ops.js --mesh 2x4 --T 4096 --segments 4096:141312 --layer sparse
//   node roofline_ops.js --sp 4 --tp 2 --segments 2048:0,2048:65536 --layer both --detail
//   node roofline_ops.js --check          # reproduce layerMs and the zone profiles at the calibration point
//
// Options
//   --mesh SPxTP | --sp N --tp N   stage mesh (EP = SP*TP chips, 128/EP experts per chip)
//   --T N          forward (padded) tokens in the chunk; default = sum of segment n
//   --segments L   comma list of n:k[:cap[:na]] = new tokens, cached tokens, lane capacity (ring_scan only,
//                  default k+n = kv_len-bounded), actual (unpadded) tokens (default n)
//   --Tr N         routed tokens for router-side MoE ops (dispatch/experts/combine/moe_reduce); default sum na
//   --layer sparse|dense|both      (sparse = MoE/MSA layer 3..59, dense = GQA layer 0..2)
//   --idx bf16|bf8 index_k cache dtype (ag_idx, kv_a2a); default bf16 (what the profiles used)
//   --imb X        expert-load imbalance folded into the experts FLOPs (sim default 1.2)
//   --ovl X        experts weight-read/compute overlap 0..1 (0 = today's kernel: read + compute summed)
//   --experts-dtype bf4|bf8   expert weight bytes (sim_core hard-codes bf4; bf8 is recomputed here)
//   --var --msa-local --bounded    sim feature flags (variable layout a2a KV write, SP-local MSA, bounded ring scan)
//   --detail       also print model ms (roof / CAL eff + latency floor, wave factor) and SIM.layerMs
//   --eff M        CAL.effs mesh key for --detail (default: the mesh itself); --effs FILE overrides with a JSON
//                  {"2x4": {"moe": {...}, "dense": {...}}, ...} (per-op merge over the calibrated table)
//   --attn seq|fused   attention mode for --detail wave factors (default seq)
'use strict';
const fs = require('fs'), path = require('path');
const SIM = require('./sim_core.js');

const OPS_SPARSE_TOK = ['norm_ag', 'qkv', 'idx_branch', 'misc', 'o_proj', 'attn_rs', 'shared', 'router', 'dispatch', 'experts', 'combine', 'moe_reduce'];
const OPS_SPARSE_SEG = ['ag_kv', 'ag_idx', 'indexer', 'sparse', 'kv_a2a'];
const OPS_DENSE_TOK = ['norm_ag', 'qkv', 'misc', 'o_proj', 'attn_rs', 'dense_mlp'];
const CCL = new Set(['norm_ag', 'attn_rs', 'dispatch', 'combine', 'ag_kv', 'ag_idx', 'kv_a2a', 'shared', 'moe_reduce', 'dense_mlp']);
const WAVE = new Set(['sparse', 'indexer', 'ring_c']);
const LAT_MEAS = { ccl: 0.04, op: 0.01 };
const ZONE_T = 5120, ZONE_K = 51200, ZONE_CAP = 56320;
const BYTES = { bf4: 0.5625, bf8: 1.0625, bf16: 2 };

function parseArgs(argv) {
  const a = { mesh: null, sp: null, tp: null, T: null, Tr: null, segs: null, layer: 'sparse', idx: 'bf16', imb: 1.2, ovl: 0,
    expDtype: 'bf4', varLayout: false, msaLocal: false, bounded: false, detail: false, eff: null, effs: null, attn: 'seq', check: false };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i], v = () => argv[++i];
    if (k === '--mesh') a.mesh = v();
    else if (k === '--sp') a.sp = Number(v());
    else if (k === '--tp') a.tp = Number(v());
    else if (k === '--T') a.T = Number(v());
    else if (k === '--Tr') a.Tr = Number(v());
    else if (k === '--segments') a.segs = v();
    else if (k === '--layer') a.layer = v();
    else if (k === '--idx') a.idx = v();
    else if (k === '--imb') a.imb = Number(v());
    else if (k === '--ovl') a.ovl = Number(v());
    else if (k === '--experts-dtype') a.expDtype = v();
    else if (k === '--var') a.varLayout = true;
    else if (k === '--msa-local') a.msaLocal = true;
    else if (k === '--bounded') a.bounded = true;
    else if (k === '--detail') a.detail = true;
    else if (k === '--eff') a.eff = v();
    else if (k === '--effs') a.effs = v();
    else if (k === '--attn') a.attn = v();
    else if (k === '--check') a.check = true;
    else throw new Error('unknown option ' + k);
  }
  if (a.mesh) [a.sp, a.tp] = a.mesh.split('x').map(Number);
  if (!a.check && !(a.sp > 0 && a.tp > 0)) throw new Error('need --mesh SPxTP or --sp/--tp');
  if (!BYTES[a.expDtype]) throw new Error('--experts-dtype must be bf4 or bf8');
  return a;
}

function parseSegs(s) {
  return s.split(',').map((x) => {
    const [n, k, cap, na] = x.split(':').map(Number);
    return { n, k: k || 0, cap: cap || (k || 0) + n, na: na === undefined ? n : na };
  });
}

function ctxOf(a, T, Tr) {
  return { sp: a.sp, tp: a.tp, P: a.sp * a.tp, T, Tr, idxB: BYTES[a.idx], imb: a.imb, ovl: a.ovl,
    varLayout: a.varLayout, bounded: a.bounded, msaLocal: a.msaLocal };
}

// experts roofline with a configurable weight dtype (same formula as sim_core roofTok 'experts')
function expertsRoof(c, wB) {
  const M3 = SIM.M3, HW = SIM.HW;
  const flops = c.Tr * M3.topk / c.P * 6 * M3.E * M3.I * c.imb;
  const wbytes = M3.Ex / c.P * 3 * M3.E * M3.I * wB;
  const cc = flops / HW.F_lofi, mm = wbytes / HW.dram, ov = c.ovl || 0;
  return (1 - ov) * (cc + mm) + ov * Math.max(cc, mm);
}

// roofline seconds per op for one layer (norm_ag counts both all-gathers of the layer)
function roofOps(kind, c, segs, expDtype) {
  const r = {};
  if (kind === 'moe') {
    for (const op of OPS_SPARSE_TOK) r[op] = (op === 'norm_ag' ? 2 : 1) * SIM.roofTok(op, c);
    if (expDtype && expDtype !== 'bf4') r.experts = expertsRoof(c, BYTES[expDtype]);
    for (const op of OPS_SPARSE_SEG) {
      let s = 0;
      if (!((op === 'kv_a2a' && !c.varLayout) || ((op === 'ag_kv' || op === 'ag_idx') && c.sp <= 1) || (op === 'ag_idx' && c.msaLocal)))
        for (const g of segs) s += SIM.roofSeg(op, c, g);
      r[op] = s;
    }
  } else {
    for (const op of OPS_DENSE_TOK) r[op] = (op === 'norm_ag' ? 2 : 1) * SIM.roofTok(op, c);
    let rc = 0, rs = 0, ring = 0, a2a = 0;
    for (const g of segs) {
      const x = SIM.roofSeg('ring_c', c, g), y = SIM.roofSeg('ring_scan', c, g);
      rc += x; rs += y; ring += Math.max(x, y);
      if (c.varLayout && c.sp > 1) a2a += SIM.roofSeg('kv_a2a', c, g);
    }
    Object.assign(r, { ring, ring_c: rc, ring_scan: rs, kv_a2a: a2a });
  }
  return r;
}

// model ms per op, the same decomposition as sim_core layerMs (measured effs, latency floors, wave factor)
function waveFactor(nTok, c) {
  const units = Math.max(1, Math.ceil(nTok / c.sp / 32)) * (SIM.M3.Hq / c.tp);
  return Math.ceil(units / 110) * 110 / units;
}
function modelOps(kind, c, segs, eff, attn, expDtype) {
  const lat = LAT_MEAS, m = {};
  const opMs = (op, roofS) => (CCL.has(op) ? lat.ccl : lat.op) + roofS * 1e3 / eff[op];
  if (kind === 'moe') {
    for (const op of OPS_SPARSE_TOK) {
      const roof = op === 'experts' && expDtype && expDtype !== 'bf4' ? expertsRoof(c, BYTES[expDtype]) : SIM.roofTok(op, c);
      m[op] = (op === 'norm_ag' ? 2 : 1) * opMs(op, roof);
    }
    for (const op of OPS_SPARSE_SEG) {
      if ((op === 'kv_a2a' && !c.varLayout) || ((op === 'ag_kv' || op === 'ag_idx') && c.sp <= 1) || (op === 'ag_idx' && c.msaLocal)) { m[op] = 0; continue; }
      const l = CCL.has(op) ? lat.ccl : lat.op, wave = WAVE.has(op);
      let sum = 0;
      for (const s of segs) sum += SIM.roofSeg(op, c, s) * 1e3 / eff[op] * (wave && attn !== 'fused' ? waveFactor(s.n, c) / waveFactor(ZONE_T, c) : 1);
      if (wave && attn === 'fused') sum *= waveFactor(c.T, c) / waveFactor(ZONE_T, c);
      m[op] = sum + (attn === 'fused' ? l : l * segs.length);
    }
  } else {
    for (const op of OPS_DENSE_TOK) m[op] = (op === 'norm_ag' ? 2 : 1) * opMs(op, SIM.roofTok(op, c));
    let ring = 0, a2a = 0;
    const wf = attn === 'fused' ? waveFactor(c.T, c) / waveFactor(ZONE_T, c) : 0;
    for (const s of segs) {
      const w = attn === 'fused' ? wf : waveFactor(s.n, c) / waveFactor(ZONE_T, c);
      ring += Math.max(SIM.roofSeg('ring_c', c, s) * 1e3 / eff.ring_c * w, SIM.roofSeg('ring_scan', c, s) * 1e3 / eff.ring_scan);
      if (c.varLayout && c.sp > 1) a2a += SIM.roofSeg('kv_a2a', c, s) * 1e3 / eff.kv_a2a;
    }
    m.ring = ring + (attn === 'fused' ? lat.op : lat.op * segs.length);
    m.kv_a2a = a2a;
  }
  return m;
}

function loadCal(effsFile) {
  const cal = SIM.calibrate(JSON.parse(fs.readFileSync(path.join(__dirname, 'calib_data.json'))));
  if (effsFile) mergeEffs(cal, JSON.parse(fs.readFileSync(effsFile)));
  return cal;
}
function mergeEffs(cal, over) {
  for (const mesh in over) {
    cal.effs[mesh] = cal.effs[mesh] || { moe: {}, dense: {} };
    for (const kind of ['moe', 'dense']) Object.assign(cal.effs[mesh][kind], (over[mesh] || {})[kind] || {});
  }
}

const ms = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, +(v * 1e3).toPrecision(6)]));
const r6 = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, +v.toPrecision(6)]));

function run(a) {
  const segs = parseSegs(a.segs || `${a.T || ZONE_T}:0`);
  const T = a.T || segs.reduce((x, s) => x + s.n, 0);
  const Tr = a.Tr || segs.reduce((x, s) => x + s.na, 0);
  const c = ctxOf(a, T, Tr);
  const kinds = a.layer === 'both' ? ['moe', 'dense'] : [a.layer === 'dense' ? 'dense' : 'moe'];
  const out = {};
  const cal = a.detail ? loadCal(a.effs) : null;
  for (const kind of kinds) {
    const key = kind === 'moe' ? 'sparse' : 'dense';
    const roof = ms(roofOps(kind, c, segs, a.expDtype));
    if (!a.detail) { out[key] = roof; continue; }
    const mk = a.eff || `${a.sp}x${a.tp}`;
    const eff = (cal.effs[mk] || cal.effs['2x4'])[kind];
    const model = modelOps(kind, c, segs, eff, a.attn, a.expDtype);
    const total = Object.values(model).reduce((x, y) => x + y, 0);
    const layerMs = a.expDtype === 'bf4' ? SIM.layerMs(kind, c, segs, eff, LAT_MEAS, a.attn) : null;
    out[key] = { roof_ms: roof, model_ms: r6(model), model_total_ms: +total.toPrecision(6), sim_layerMs: layerMs && +layerMs.toPrecision(6),
      eff: r6(eff), eff_mesh: cal.effs[mk] ? mk : '2x4 (fallback)',
      note: 'model_ms = roof/eff + latency floor (0.04 ms CCL, 0.01 ms op) x calls, wave factor on sparse/indexer/ring_c; sparse-layer total excludes the pipeline MoE multiplier (CAL.pipe.moeMult)' };
  }
  const res = kinds.length === 1 ? out[kinds[0] === 'moe' ? 'sparse' : 'dense'] : out;
  if (a.detail) return { mesh: `${a.sp}x${a.tp}`, T, Tr, segments: segs, ...(kinds.length === 1 ? res : { layers: res }) };
  return res;
}

// Reproduce layerMs and the measured zones at the calibration point (chunk 5120, 51,200 cached, capacity 56,320).
function check() {
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'calib_data.json')));
  const cal = SIM.calibrate(data);
  const ZM = { qkv: 'attn/qkv_proj', idx_branch: 'attn/index_branch', ag_kv: 'attn/ag_kv', ag_idx: 'attn/ag_index_k', indexer: 'attn/indexer',
    sparse: 'attn/sparse_sdpa', o_proj: 'attn/o_proj', attn_rs: 'attn/ccl_out_reduce_scatter', shared: 'mlp/shared_expert',
    router: 'mlp/router_topk', dispatch: 'mlp/dispatch', experts: 'mlp/experts_mm', combine: 'mlp/combine', moe_reduce: 'mlp/moe_reduce' };
  const ZD = { qkv: 'attn/qkv_proj', ring: 'attn/ring_joint_sdpa', o_proj: 'attn/o_proj', attn_rs: 'attn/ccl_out_reduce_scatter', dense_mlp: 'mlp' };
  const report = {};
  let worst = 0;
  for (const mesh of ['2x4', '4x2', '8x4']) {
    const [sp, tp] = mesh.split('x').map(Number);
    const a = { sp, tp, idx: 'bf16', imb: 1.2, ovl: 0, varLayout: false, bounded: false, msaLocal: false };
    const c = ctxOf(a, ZONE_T, ZONE_T), segs = [{ n: ZONE_T, k: ZONE_K, cap: ZONE_CAP }];
    const z = data.zones[mesh];
    report[mesh] = {};
    for (const [kind, lname, zmap] of [['moe', 'layer03_sparse', ZM], ['dense', 'layer00_dense', ZD]]) {
      const eff = cal.effs[mesh][kind];
      const model = modelOps(kind, c, segs, eff, 'seq');
      const tot = Object.values(model).reduce((x, y) => x + y, 0);
      const lm = SIM.layerMs(kind, c, segs, eff, LAT_MEAS, 'seq');
      worst = Math.max(worst, Math.abs(tot - lm));
      const meas = { norm_ag: z[lname + '/input_norm_allgather'].mean + z[lname + '/post_attn_norm_allgather'].mean };
      for (const op in zmap) meas[op] = z[lname + '/' + zmap[op]].mean;
      const listed = Object.values(meas).reduce((x, y) => x + y, 0);
      meas.misc = z[lname].mean - listed;
      const rows = {};
      for (const op in meas) rows[op] = { model_ms: +model[op].toFixed(4), zone_ms: +meas[op].toFixed(4), eff: eff[op] === undefined ? null : +eff[op].toFixed(4) };
      if (kind === 'dense') rows.ring.eff = +eff.ring_c.toFixed(4);
      report[mesh][kind === 'moe' ? 'sparse' : 'dense'] = { sum_of_ops_ms: +tot.toFixed(4), sim_layerMs: +lm.toFixed(4), zone_layer_ms: z[lname].mean, ops: rows };
    }
  }
  return { condition: 'chunk 5120, 51,200 cached, capacity 56,320, bf16 index_k, imbalance 1.2, CAL = calibrate(calib_data.json)',
    max_abs_diff_sum_vs_layerMs: worst, meshes: report };
}

if (require.main === module) {
  const a = parseArgs(process.argv);
  console.log(JSON.stringify(a.check ? check() : run(a), null, 1));
}
module.exports = { roofOps, modelOps, loadCal, mergeEffs, parseSegs, ctxOf, run, check };
