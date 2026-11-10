"""MKA2: attention-type layers inside the megakernel engine (MK1 engine + the MKA1 SDPA phase, spec v0.3).
Per layer: STREAM mixer (qkv stand-in) -> SDPA phase on the full grid (4 own users x 6 q heads / 1 KV head, KV dtype
MKA_KVDT bfp8 default, context MKA_CTX) -> wo stand-in (real bytes, prefetched during SDPA) + attention AR stall -> fm11 MLP
-> MLP AR stall. ATTN sync on global semaphores; next-layer weight issue throttled by MKA_BUDGET bytes (phase word 13).
Original MK1 bench docstring: Fused Qwen3.8 MLP decode block v2 (one chip of TP8) as ONE generic_op on a 10x8 core rectangle:
68 gate|up cores (one h column each), 80 down cores (2 output tiles each, whole down slice prefetched into L1),
x [32, 5120] bf16 width-sharded over the 80 cores (2 tiles each) and multicast to all; h multicast to all.
Weights BFP4, every core's slice contiguous in one DRAM bank. env FM_ITERS, FM_GRP (gate|up blocks per read)."""
import math, os, time, torch, ttnn
KD = os.environ.get("MKA_KD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels"))
TILE, H, I, M = 32, 5120, 2176, 32
KT, HT = H // TILE, I // TILE   # 160, 68
GX, GY = 10, 8; NC = GX * GY; NGU = HT; NDN = KT // NC; XSW = KT // NC
KB = 16; GRP = int(os.environ.get("FM_GRP", "2")); ITERS = int(os.environ.get("FM_ITERS", "2"))
# MK_MESH=1 (AR integration, stage 0): run the same per-device program on a 1x8 FABRIC_1D_RING column (TT_VISIBLE_DEVICES
# order = ring order), replicated inputs; the mesh shims below keep the rest of this file device-agnostic.
MESH = os.environ.get('MK_MESH', '0') == '1'
if MESH:
    _rcfg = ttnn.FabricRouterConfig(); _rcfg.max_packet_payload_size_bytes = int(os.environ.get('QWEN36_FABRIC_PAYLOAD_BYTES', '8192'))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None,
                           ttnn.FabricTensixConfig.DISABLED, ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT, _rcfg)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, int(os.environ.get('MK_MESH_N', '8'))), trace_region_size=64 << 20)
    _from_torch, _to_torch, _generic_op = ttnn.from_torch, ttnn.to_torch, ttnn.generic_op
    def _ft(*a, **k):
        # tilize on the HOST (the mesh path would tilize on device through a same-size L1 temp, which leaves a hole above
        # the result and pushes the buffer into the CB region), then move to the mesh with the requested memory config
        d = k.pop('device', None); mc = k.pop('memory_config', None)
        if d is None: return _from_torch(*a, **k)
        if 'mesh_mapper' not in k: k['mesh_mapper'] = ttnn.ReplicateTensorToMesh(dev)
        h = _from_torch(*a, **k)
        return ttnn.to_device(h, d, memory_config=mc) if mc is not None else ttnn.to_device(h, d)
    def _tt(t, *a, **k):  # device 0's copy (mesh_identity() checks the other devices)
        return _to_torch(ttnn.get_device_tensors(t)[0], *a, **k) if t.storage_type() == ttnn.StorageType.DEVICE else _to_torch(t, *a, **k)
    PROG_PER_DEV = None  # optional callable(coord) -> ProgramDescriptor (per-device AR args); else the same program everywhere
    def _go(io, prog):
        if isinstance(prog, ttnn.ProgramDescriptor):
            mp = ttnn.MeshProgramDescriptor()
            for r in range(dev.shape[0]):
                for c_ in range(dev.shape[1]):
                    co = ttnn.MeshCoordinate(r, c_)
                    mp[ttnn.MeshCoordinateRange(co, co)] = PROG_PER_DEV(co) if PROG_PER_DEV else prog
            prog = mp
        return _generic_op(io, prog)
    ttnn.from_torch, ttnn.to_torch, ttnn.generic_op = _ft, _tt, _go
    ttnn.close_device = ttnn.close_mesh_device
    def mesh_identity(t):  # max |device d - device 0| over the mesh
        ds = [_to_torch(d).float() for d in ttnn.get_device_tensors(t)]
        return max((d_ - ds[0]).abs().max().item() for d_ in ds)
else:
    dev = ttnn.open_device(device_id=int(os.environ.get('FM_DEV', '0')), trace_region_size=64 << 20)
    def mesh_identity(t): return 0.0
# Setup body: engine/program.py setup_bench (pure refactor, spec §16 / v1.22). Its names land in this module so
# the run loop below (and the MESH _go shim's PROG_PER_DEV) are unchanged.
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'engine'))
import program as _mk_program
globals().update(_mk_program.setup_bench(dev, MESH, mesh_identity))
def pcc_of(t, g): v = ttnn.to_torch(t).float().reshape(M, H); return torch.corrcoef(torch.stack([v.flatten(), g.flatten()]))[0, 1].item(), (v - g).abs().max().item()
for it in range(ITERS):
    t1 = time.time(); y = ttnn.generic_op([x, wgu_all, wd_all, wm_all, xbuf, hbuf, ymix, qten, outs, wo_dummy] + kvs + kvvs + ([wq_all, rsp, qkv_out] if RQKV else []) + ([wo_all, wod] if RWO else []) + ([hg_t, kd_t] if RQNR else []) + ([mk_t] if RMASK else []) + ([wv_all, v_xk, v_rs, v_h7, v_xpk, v_pbuf, v_xo, v_x0, v_id, v_j] if V12 else []) + ([ar_out, ar_blk] if REAL_AR else []) + [out_t], prog); ttnn.synchronize_device(dev)
    if REAL_AR and AR_DBG:   # every AR of the pass: res_k = res_{k-1} + ring * in_k (both chunks dumped by the owners)
        _db = ttnn.to_torch(ar_dbgt).float().reshape(TILE, AR_NAR * 2, H)
        if it == 0: _rprev = xb.clone()
        _worst = []
        for k_ in range(AR_NAR):
            _in, _res = _db[:, 2 * k_], _db[:, 2 * k_ + 1]
            _g = (_rprev + AR.ring * _in).bfloat16().float()
            _p = torch.corrcoef(torch.stack([_res.flatten(), _g.flatten()]))[0, 1].item()
            _worst.append((round(_p, 6), k_, 'attn' if (AR_ATYPE >> k_) & 1 else 'mlp', round((_res - _g).abs().max().item(), 4), round(_g.abs().max().item(), 3)))
            _rprev = _res
        print(f"[ar] iter {it} per-AR check (res = prev + ring*in): min pcc {min(_worst)} all {[w[0] for w in _worst]}", flush=True)
    if REAL_AR and AR_SEQ:   # this pass's golds from the residual the previous pass left (rstd stays the CB 20 stand-in)
        for L_ in range(NL):
            _g, _u, _d = vlayer(L_)
            _xn = (rstd_g[:, None] * _xres.double()).float()
            _pL = (torch.nn.functional.silu(_xn @ _g) * (_xn @ _u)).bfloat16().float() @ _d
            _xres = (_xres + AR.ring * _pL).bfloat16().float()
        v_gold = _pL
    if V12 and not (REAL_AR and AR_GAMMA):   # (MK_AR_GAMMA: v_xo ends the pass holding x_deliver, not the partial)
        _v = ttnn.to_torch(v_xo).float().reshape(TILE, H)[:V_R]; _g2 = v_gold[:V_R]
        if os.environ.get('V12_SAVE') and it == 0: torch.save(ttnn.to_torch(v_xo), os.environ['V12_SAVE'])
        _bad = [j_ for j_ in range(16) if torch.corrcoef(torch.stack([_v[:, j_*320:(j_+1)*320].flatten(), _g2[:, j_*320:(j_+1)*320].flatten()]))[0, 1].item() < 0.999]
        print(f"[v12] iter {it} last-layer MLP (owner xo) pcc {torch.corrcoef(torch.stack([_v.flatten(), _g2.flatten()]))[0, 1].item():.5f} max|err| {(_v - _g2).abs().max().item():.4f} |gold|max {_g2.abs().max().item():.3f} bad owner chunks {_bad}", flush=True)
    if REAL_AR and AR_SEQ:   # x of the next pass's layer 0 as the last AR delivered it into the K-row slots (STREAM (0, k))
        _xk = ttnn.to_torch(v_xk).float().reshape(TILE, NC * 20 * TILE)
        _xd = torch.cat([_xk[:, (10 * (j_ // 2)) * 640 + (j_ % 2) * 320:(10 * (j_ // 2)) * 640 + (j_ % 2 + 1) * 320] for j_ in range(16)], 1)
        _xg = _xres * (AR_GTAB[0, 0][None, :] if AR_GAMMA else 1.0)
        print(f"[ar] iter {it} delivered x{' (gamma)' if AR_GAMMA else ''} pcc {torch.corrcoef(torch.stack([_xd.flatten(), _xg.flatten()]))[0, 1].item():.6f} max|err| {(_xd - _xg).abs().max().item():.4f} |gold|max {_xg.abs().max().item():.3f}", flush=True)
    if REAL_AR and AR_SEQ:   # (R < 32: packed residual, not checked here) the residual stream after this pass
        _ar = ttnn.to_torch(ar_out).float().reshape(TILE, H); _arg = _xres
        print(f"[ar] iter {it} residual pcc {torch.corrcoef(torch.stack([_ar.flatten(), _arg.flatten()]))[0, 1].item():.6f} max|err| {(_ar - _arg).abs().max().item():.4f} |gold|max {_arg.abs().max().item():.3f} mesh identity {mesh_identity(ar_out):.3e}", flush=True)
    p, e = pcc_of(y, gold_last)
    if os.environ.get("FM_DIAG") == "1":
        v = ttnn.to_torch(y).float().reshape(M, H)
        rat = []
        for ci in range(NC):
            cs = slice(ci * NDN * TILE, (ci + 1) * NDN * TILE)
            g_ = gold_last[:, cs]; r_ = (v[:, cs] * g_).sum() / (g_ * g_).sum()
            pc_ = torch.corrcoef(torch.stack([v[:, cs].flatten(), g_.flatten()]))[0, 1].item()
            rat.append((ci, round(r_.item(), 3), round(pc_, 4)))
        bad = [t for t in rat if abs(t[1] - 1) > 0.02 or t[2] < 0.999]
        print(f"[fm11] diag bad cores {len(bad)}: {bad[:40]}", flush=True)
    pm, em = pcc_mix = (lambda v, g: (torch.corrcoef(torch.stack([v.flatten(), g.flatten()]))[0, 1].item(), (v - g).abs().max().item()))(ttnn.to_torch(ymix).float().reshape(M, -1), gold_mix)
    print(f"[mk1] iter {it} mixer pcc {pm:.5f} max|err| {em:.4f}", flush=True)
    if RQKV:
        qv = ttnn.to_torch(qkv_out).float().reshape(NL, M, QN * TILE)
        QRW = V_R if (QSK and V_R < 32) else M   # split-K at R < 32: only rows 0..R-1 of x / the partials travel (v12 faces)
        qv = qv[:, :QRW]
        for L in range(NL):
            if not is_attn_layer(L): continue
            if RWO: gold_qkv[L][:, 64 * TILE:112 * TILE] = torch.sigmoid(gold_qkv[L][:, 64 * TILE:112 * TILE]) if it == 0 else gold_qkv[L][:, 64 * TILE:112 * TILE]
            gq_ = gold_qkv[L][:QRW]
            pq_ = torch.corrcoef(torch.stack([qv[L].flatten(), gq_.flatten()]))[0, 1].item()
            nan_ = [n for n in range(QN) if not torch.isfinite(qv[L][:, n*TILE:(n+1)*TILE]).all()]
            if nan_: print(f"[rqkv] iter {it} layer {L} non-finite tiles {nan_}", flush=True)
            bad_ = [n for n in range(QN) if torch.corrcoef(torch.stack([qv[L][:, n*TILE:(n+1)*TILE].flatten(), gq_[:, n*TILE:(n+1)*TILE].flatten()]))[0, 1].item() < 0.999]
            print(f"[rqkv] iter {it} layer {L} rows {QRW} qkv pcc {pq_:.5f} max|err| {(qv[L]-gq_).abs().max().item():.4f} |gold|max {gq_.abs().max().item():.3f} bad tiles {bad_[:12]}", flush=True)
    vs = ttnn.to_torch(outs).float().reshape(NL, NU, TILE, DHD)
    for L in range(NL):
        v_, g2 = vs[L][:, :NQH * T], golds_s[L][:, :NQH * T]
        if os.environ.get("MKA_DBGRAW") == "1":
            raw = vs[L].reshape(NU, TILE, DHD)
            print(f"[mka2] RAW L{L} O[:, :24, :224] absmax {raw[:, :24, :224].abs().max().item():.3e} mean {raw[:, :24, :224].abs().mean().item():.3e}  den(slot7) min {raw[:, :24, 224:].min().item():.3e} max {raw[:, :24, 224:].max().item():.3e}", flush=True)
        if os.environ.get("MKA_DIAG") == "1":
            for uu in range(NU):
                vu, gu_ = vs[L][uu, :NQH * T], golds_s[L][uu, :NQH * T]
                print(f"[mka2]   L{L} user {uu} pcc {torch.corrcoef(torch.stack([vu.flatten(), gu_.flatten()]))[0, 1].item():.4f} |v|max {vu.abs().max().item():.3e} |g|max {gu_.abs().max().item():.3e} vs-other-layer-gold pcc {torch.corrcoef(torch.stack([vu.flatten(), golds_s[1-L][uu, :NQH*T].flatten()]))[0, 1].item() if NL > 1 else 0:.4f}", flush=True)
        if not is_attn_layer(L): continue
        if RMASK and L in GOLD_NOMASK:   # discrimination: device error vs the masked and the unmasked golden
            gn_ = GOLD_NOMASK[L][:, :NQH * T]
            print(f"[rmask] iter {it} layer {L} |dev-masked| {(v_ - g2).norm().item():.4e}  |dev-unmasked| {(v_ - gn_).norm().item():.4e}  |masked-unmasked| {(g2 - gn_).norm().item():.4e}", flush=True)
        print(f"[mka2] iter {it} layer {L} sdpa pcc {torch.corrcoef(torch.stack([v_.flatten(), g2.flatten()]))[0, 1].item():.5f} max|err| {(v_-g2).abs().max().item():.4f}", flush=True)
    if RQNR:
        kv_ = ttnn.to_torch(kd_t).float().reshape(NL, M, -1)
        for L in range(NL):
            if not is_attn_layer(L): continue
            KRW = V_R if (QSK and V_R < 32) else M
            pk_ = torch.corrcoef(torch.stack([kv_[L][:KRW].flatten(), gold_k[L][:KRW].flatten()]))[0, 1].item()
            print(f"[rqnr] iter {it} layer {L} rows {KRW} k head pcc {pk_:.5f} max|err| {(kv_[L][:KRW] - gold_k[L][:KRW]).abs().max().item():.4f}", flush=True)
    if RWO:
        wv_ = ttnn.to_torch(wod).float().reshape(NL, M, -1)
        for L in range(NL):
            if not is_attn_layer(L): continue
            rows_ = slice(0, 4 * NU)
            pw_ = torch.corrcoef(torch.stack([wv_[L][rows_].flatten(), gold_wo[L][rows_].flatten()]))[0, 1].item()
            print(f"[rwo] iter {it} layer {L} wo pcc {pw_:.5f} max|err| {(wv_[L][rows_] - gold_wo[L][rows_]).abs().max().item():.4f} |gold|max {gold_wo[L].abs().max().item():.3f}", flush=True)
    print(f"[fm11] iter {it} wall {time.time()-t1:.2f}s last-layer pcc {p:.5f} max|err| {e:.4f} |gold|max {gold_last.abs().max().item():.3f}", flush=True)
    if MESH:
        print(f"[mesh] iter {it} devices {dev.get_num_devices()}: max |dev d - dev 0| out {mesh_identity(y):.3e} ymix {mesh_identity(ymix):.3e} sdpa {mesh_identity(outs):.3e}", flush=True)
ttnn.ReadDeviceProfiler(dev)
print('[fm11] profiler read', flush=True)
ttnn.close_device(dev)
print('[fm11] closed', flush=True)
