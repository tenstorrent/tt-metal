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
# rstd the consumers use: MK_AR_RSTD=1 (spec rstd producer) = rsqrt(sum x^2 / H + eps) of the real residual (fp32 sum of the bf16
# residual); else the static CB 20 stand-in rstd_g (the setup x's)
_RSTD_REAL = os.environ.get('MK_AR_RSTD', '0') == '1'
def _rstd_of(x_):
    if not _RSTD_REAL: return rstd_g
    xs = x_.double()
    if xs.shape[0] < M: xs = torch.cat([xs, torch.zeros(M - xs.shape[0], H, dtype=xs.dtype)], 0)
    return 1.0 / torch.sqrt((xs * xs).sum(-1) / H + EPS)
# MK_WALL_N=N (bench-only wall clock): capture the engine launch's (io, program) so the block after the loop can replay it
_WALL = {}
_go_wall0 = ttnn.generic_op
def _go_wall(io, prog):
    if len(io) > 30: _WALL['a'] = (io, prog)   # the engine op (the seed / drain / dump ops have few io tensors)
    return _go_wall0(io, prog)
ttnn.generic_op = _go_wall
for it in range(ITERS):
    if KVA: KVA_PRE(it)   # this verify step's kvpos (MKA_KVA)
    if ENGINE_SEED is not None: ENGINE_SEED()   # MK_L1P2_SEED: static-CB contents (residual from ENGINE_X0, constants) right before the launch
    t1 = time.time(); y = ttnn.generic_op([t_ for t_ in ([t_ for t_ in (x, wgu_all, wd_all, wm_all, xbuf, hbuf, ymix, qten, outs, wo_dummy) if t_ is not None] + kvs + kvvs + ([wq_all, rsp, qkv_out] if RQKV else ([rsp] if V12 else [])) + ([wo_all, wod] if RWO else []) + ([hg_t, kd_t] if RQNR else []) + ([mk_t] if RMASK else []) + ([t_ for t_ in (wv_all, v_xk, v_rs, v_h7, v_xpk, v_pbuf, v_xo, v_x0, v_id, v_j) if t_ is not None] if V12 else []) + ([ar_out, ar_blk] if REAL_AR else []) + (list(GDN_ROLE.get('io', [])) if GDN24 else []) + [out_t]) if t_ is not None], prog); ttnn.synchronize_device(dev)
    _xo_t = XO_DUMP() if XO_DUMP is not None else v_xo   # MK_L1P2_QOVXO: CB_VXO read back right after the engine op (before any other device op)
    _rsp_t = RSP_DUMP() if RSP_DUMP is not None else rsp_sh   # MK_L1P2_SEED: CB 43 read back (the AR's rstd checks), same place
    if ENGINE_DRAIN is not None: ENGINE_DRAIN()   # MK_L1P2_SEED: residual CB 5 -> ENGINE_X0 (the next launch's seed), same place
    _res_t = ENGINE_X0 if ENGINE_X0 is not None else globals().get('ar_out')
    if GDN24 and os.environ.get('E1_GB_TS', '0') == '1':   # E1_GB_TS probe: the GDN cores' wall-clock stamps (arg block words) of this launch
        _ti = GDN_ROLE['role'].role['info']; _o = _ti['ts_off']; _tt_ = ttnn.to_torch(GDN_ROLE['io'][0])
        _tt_ = _tt_.reshape(len(GDN_ROLE['role'].cores), -1)[:, _o:_o + 32 * _ti['NG']].to(torch.int64) & 0xFFFFFFFF
        with open(os.environ.get('E1_GB_TS_OUT', 'gbts') + f'.it{it}.txt', 'w') as _f:   # per core: x y NG then NG x 32 stamps (0 = not reached)
            for c_, r_ in zip(GDN_ROLE['role'].cores, _tt_.tolist()): _f.write(f"{c_.x} {c_.y} {_ti['NG']} " + " ".join(map(str, r_)) + "\n")
    if KVA: KVA_POST(it)   # SDPA golden over the appended cache (+ MKA_KVA_CHECK: bit-compare vs paged_fused_update_cache)
    if GDN24 and it == 0 and callable(GDN_ROLE.get('check')):   # GDN role bit-identity: after the FIRST run only (later runs advance its state)
        print(f"[gdn24] check {'PASS' if GDN_ROLE['check']() else 'FAIL'}", flush=True)
    if GDN24 and callable(GDN_ROLE.get('gdnout_check')): GDN_ROLE['gdnout_check'](it)   # MKA_GDNOUT: gdn_out PCC (every run)
    if REAL_AR and AR_DBG:   # every AR of the pass: res_k = res_{k-1} + ring * in_k (both chunks dumped by the owners)
        _db = ttnn.to_torch(ar_dbgt).float().reshape(TILE, AR_NAR * 2, H)
        if V_R < 32:   # R < 32: input rows < R (tile layout); residual = packed stream (tile t face f at (2t + f) R 32 B) -> [R, H]
            def _raw(u):   # untilized [32, W] -> raw tile element order [W / 32, 1024]
                t_ = u.reshape(TILE, -1, TILE).permute(1, 0, 2)
                return torch.cat([t_[:, :16, :16].reshape(-1, 256), t_[:, :16, 16:].reshape(-1, 256), t_[:, 16:, :16].reshape(-1, 256), t_[:, 16:, 16:].reshape(-1, 256)], 1)
            def _unpk(u):   # [32, H] dump of packed chunks -> [R, H]
                o_ = torch.zeros(V_R, H)
                for j_ in range(16):
                    r_ = _raw(u[:, j_ * 320:(j_ + 1) * 320]).flatten()[:10 * 2 * V_R * 16].reshape(10, 2, V_R, 16)
                    for t_ in range(10):
                        for f_ in range(2): o_[:, j_ * 320 + t_ * 32 + f_ * 16:j_ * 320 + t_ * 32 + f_ * 16 + 16] = r_[t_, f_]
                return o_
            _db = torch.stack([(_db[:V_R, i_] if i_ % 2 == 0 else _unpk(_db[:, i_])) for i_ in range(AR_NAR * 2)], 1)
        if it == 0: _rprev = xb.clone()[:V_R]
        _x0_pass = _rprev.clone()   # this pass's layer-0 mixer x (the residual the previous pass left; x at iter 0)
        _worst = []
        # MKA_UPAR: an attention AR's inputs differ by device parity (own users g = 2u + d % 2): res = prev + ring/2 *
        # (in_dev0 + in_dev1) -- every user row comes from exactly one device per KV-head pair (V_R = 32 only)
        _UP = os.environ.get('MKA_UPAR', '0') == '1' and V_R == 32
        _db1 = _to_torch(ttnn.get_device_tensors(ar_dbgt)[1]).float().reshape(TILE, AR_NAR * 2, H) if _UP else None
        for k_ in range(AR_NAR):
            _in, _res = _db[:, 2 * k_], _db[:, 2 * k_ + 1]
            if _UP and (AR_ATYPE >> k_) & 1: _in = 0.5 * (_in + _db1[:, 2 * k_])
            _g = (_rprev + AR.ring * _in).bfloat16().float()
            _p = torch.corrcoef(torch.stack([_res.flatten(), _g.flatten()]))[0, 1].item()
            _worst.append((round(_p, 6), k_, 'attn' if (AR_ATYPE >> k_) & 1 else 'mlp', round((_res - _g).abs().max().item(), 4), round(_g.abs().max().item(), 3)))
            if os.environ.get('MK_AR_CHK_DEV', '0') == '1' and (_p < 0.99999 or (k_ == 1 and os.environ.get('MK_AR_CHK_ALL', '0') == '1')):   # where, and do the inputs differ by device?
                _e = (_res - _g).abs() > 4 * (2.0 ** (torch.floor(torch.log2(_g.abs() + 1e-30)) - 7))   # > 4 bf16 ulps
                _dvt = [_to_torch(t_).float().reshape(TILE, AR_NAR * 2, H)[:V_R] for t_ in ttnn.get_device_tensors(ar_dbgt)]
                _dv = [t_[:, 2 * k_] for t_ in _dvt]
                _gs = (_rprev + sum(_dv)).bfloat16().float()
                _ps = torch.corrcoef(torch.stack([_res.flatten(), _gs.flatten()]))[0, 1].item()
                _ez = _e.nonzero()
                print(f"[archk] iter {it} AR {k_}: {int(_e.sum())} elems > 4 ulp; rows {sorted(set(_ez[:, 0].tolist()))[:32]}; owner chunks (320 cols) "
                      f"{sorted(set((_ez[:, 1] // 320).tolist()))}; vs prev + sum_d in_d pcc {_ps:.6f} max|err| {(_res - _gs).abs().max().item():.4f}; "
                      f"in max|d - d0| {[round((d_ - _dv[0]).abs().max().item(), 4) for d_ in _dv]}; "
                      f"res max|d - d0| {[round((t_[:, 2 * k_ + 1] - _res).abs().max().item(), 4) for t_ in _dvt]}; "
                      f"prev max|d - d0| {[round((t_[:, 2 * k_ - 1] - _rprev).abs().max().item(), 4) for t_ in _dvt] if k_ else '-'}", flush=True)
                if k_ % 2 == 1 and not (AR_ATYPE >> k_) & 1:   # MLP AR of layer k_ // 2: which (rstd, gamma) reproduces each device's MLP output?
                    L_ = k_ // 2; _g, _u, _d = vlayer(L_)
                    _xm = _rprev if k_ < 2 else None
                    _xmix = _x0_pass if k_ == 1 else _dvt[0][:, 2 * k_ - 3]   # this layer's mixer x (residual before AR k_ - 1)
                    _gt = (lambda i_: AR_GTAB[L_, i_][None, :]) if GAMMA_CONS else (lambda i_: 1.0)
                    def _mlp(xr_, x_, gi_):
                        _xn = (_rstd_of(xr_)[:V_R, None] * (x_ * _gt(gi_)).bfloat16().double()).float()
                        return (torch.nn.functional.silu(_xn @ _g) * (_xn @ _u)).bfloat16().float() @ _d
                    _hyp = {'ok': _mlp(_rprev, _rprev, 1), 'rstd(mixer x)': _mlp(_xmix, _rprev, 1), 'gamma(input_ln)': _mlp(_rprev, _rprev, 0),
                            'x = mixer x': _mlp(_rprev, _xmix, 1)}
                    _pc = lambda a_, b_: round(torch.corrcoef(torch.stack([a_.flatten(), b_.flatten()]))[0, 1].item(), 6)
                    if os.environ.get('MK_AR_SAVE'):
                        torch.save(dict(dvt=[t_[:, :8] for t_ in _dvt], x0=_x0_pass, gtab=AR_GTAB if GAMMA_CONS else None, w=[vlayer(l_) for l_ in range(2)],
                                        eps=EPS, H=H, ring=AR.ring, rsp=ttnn.to_torch(_rsp_t).float()), os.environ['MK_AR_SAVE'] + f'.it{it}.pt')
                        print(f"[archk] saved {os.environ['MK_AR_SAVE']}.it{it}.pt", flush=True)
                    print(f"[archk] iter {it} AR {k_} (L{L_} MLP) hypotheses pcc dev0 / dev1: " + "; ".join(f"{n_} {_pc(_dv[0], h_)} / {_pc(_dv[1], h_)}" for n_, h_ in _hyp.items()), flush=True)
            _rprev = _res
        print(f"[ar] iter {it} per-AR check (res = prev + ring*in{' | UPAR attn: ring/2*(in_d0+in_d1)' if _UP else ''}): min pcc {min(_worst)} all {[w[0] for w in _worst]}", flush=True)
        if it == 0:   # what the first AR's residual started from: the stand-in x (expected) or zero
            _r0, _i0 = _db[:, 1], _db[:, 0]
            _pz = torch.corrcoef(torch.stack([_r0.flatten(), (AR.ring * _i0).bfloat16().float().flatten()]))[0, 1].item()
            print(f"[ar] iter 0 AR 0 residual vs ring*in only (residual started at 0) pcc {_pz:.6f}; vs x + ring*in {_worst[0][0]}", flush=True)
    if REAL_AR and AR_SEQ:   # this pass's golds from the residual the previous pass left (rstd stays the CB 20 stand-in)
        _xin_seq = {}
        for L_ in range(NL):
            _xin_seq[L_] = _xres.clone()   # layer L_'s mixer x (delivered by the previous MLP AR)
            _g, _u, _d = vlayer(L_)
            _xn = (_rstd_of(_xres)[:, None] * _xres.double()).float()
            _pL = (torch.nn.functional.silu(_xn @ _g) * (_xn @ _u)).bfloat16().float() @ _d
            _xres = (_xres + AR.ring * _pL).bfloat16().float()
        v_gold = _pL
    if V12 and REAL_AR and AR_DBG:   # per-layer v12 MLP vs a gold from the residual the AR delivered to that MLP (AR program order:
        # [mixer AR under AAR (attention, and GDN with MK_REAL_GAR)], MLP AR): xn = rstd(xr) * (xr [.] γ_post_L); MLP = silu(xn g) * (xn u) @ d
        _k, _mp, _nr = 0, [], []
        for L_ in range(NL):
            if REAL_AAR and (is_attn_layer(L_) or ('MK_REAL_GAR', '1') in MKDEF): _k += 1
            _xr = _db[:, 2 * (_k - 1) + 1] if _k > 0 else _x0_pass
            _xg = (_xr * AR_GTAB[L_, 1][None, :]).bfloat16().float() if (GAMMA_CONS or AR_GAMMA) else _xr
            _g, _u, _d = vlayer(L_)
            _xn = (_rstd_of(_xr)[:V_R, None] * _xg.double()).float()
            _gm = (torch.nn.functional.silu(_xn @ _g) * (_xn @ _u)).bfloat16().float() @ _d
            _dm = _db[:, 2 * _k]   # this device's MLP output = the MLP AR's input
            _mp.append(round(torch.corrcoef(torch.stack([_dm.flatten(), _gm.flatten()]))[0, 1].item(), 5))
            if globals().get('GAMMA_CONS') or globals().get('AR_GAMMA'):   # [norm] scale vs an explicit RMSNorm with the MODEL γ (= table / MK_NORM_FIX); PCC is scale-blind
                _nf = globals().get('NORM_FIX', 1.0)
                _xt = (_xr.double() * torch.rsqrt(_xr.double().pow(2).mean(-1, keepdim=True) + globals().get('EPS', 1e-6))
                       * (AR_GTAB[L_, 1][None, :].double() / _nf))[:V_R].float()
                _gt_ = (torch.nn.functional.silu(_xt @ _g) * (_xt @ _u)).bfloat16().float() @ _d
                _nr.append(round((_dm.norm() / _gt_.norm()).item(), 4))
            _k += 1
        print(f"[v12] iter {it} per-layer MLP vs AR-delivered-x gold pcc min {min(_mp)} all {_mp}", flush=True)
        if _nr:
            _ok = all(abs(r_ - 1.0) <= 0.03 for r_ in _nr)
            print(f"[norm] iter {it} MLP |dev| / |RMSNorm(x)*gamma_model MLP| per layer {_nr} (MK_NORM_FIX {globals().get('NORM_FIX', 1.0):.6f}) "
                  f"{'PASS' if _ok else 'FAIL'} (1.000 +- 3%; RC has MK_RSTD_HIFI4 + MK_GC_HIFI4)", flush=True)
    if V12 and not (REAL_AR and AR_GAMMA):   # (MK_AR_GAMMA: v_xo ends the pass holding x_deliver, not the partial)
        _v = ttnn.to_torch(_xo_t).float().reshape(TILE, H)[:V_R]; _g2 = v_gold[:V_R]
        if os.environ.get('V12_SAVE') and it == 0: torch.save(ttnn.to_torch(_xo_t), os.environ['V12_SAVE'])
        _bad = [j_ for j_ in range(16) if torch.corrcoef(torch.stack([_v[:, j_*320:(j_+1)*320].flatten(), _g2[:, j_*320:(j_+1)*320].flatten()]))[0, 1].item() < 0.999]
        if REAL_AR and (REAL_AAR or globals().get('GAMMA_CONS')):   # the AR_SEQ gold chains the MLPs only (no mixer ARs, no γ): meaningless here
            print(f"[v12] iter {it} last-layer MLP (owner xo) AR_SEQ gold: NOT VALID under AAR/GAR/MK_GAMMA_CONS (no PCC reported; use the per-layer [v12] line)", flush=True)
        else: print(f"[v12] iter {it} last-layer MLP (owner xo) pcc {torch.corrcoef(torch.stack([_v.flatten(), _g2.flatten()]))[0, 1].item():.5f} max|err| {(_v - _g2).abs().max().item():.4f} |gold|max {_g2.abs().max().item():.3f} bad owner chunks {_bad}", flush=True)
        if os.environ.get('MK_NORM_DIAG') == '1': print(f"[normdiag] iter {it} v12 last-layer MLP |dev|/|gold| {(_v.norm() / _g2.norm()).item():.4f}", flush=True)
    if REAL_AR and AR_SEQ and v_xk is not None:   # x of the next pass's layer 0 as the last AR delivered it into the K-row slots (STREAM (0, k)); MK_V12_CB: CB, not readable
        _xk = ttnn.to_torch(v_xk).float().reshape(TILE, NC * 20 * TILE)
        _xd = torch.cat([_xk[:, (10 * (j_ // 2)) * 640 + (j_ % 2) * 320:(10 * (j_ // 2)) * 640 + (j_ % 2 + 1) * 320] for j_ in range(16)], 1)
        _xg = _xres * (AR_GTAB[0, 0][None, :] if AR_GAMMA else 1.0)
        print(f"[ar] iter {it} delivered x{' (gamma)' if AR_GAMMA else ''} pcc {torch.corrcoef(torch.stack([_xd.flatten(), _xg.flatten()]))[0, 1].item():.6f} max|err| {(_xd - _xg).abs().max().item():.4f} |gold|max {_xg.abs().max().item():.3f}", flush=True)
    if REAL_AR and AR_SEQ:   # (R < 32: packed residual, not checked here) the residual stream after this pass
        _ar = ttnn.to_torch(_res_t).float().reshape(TILE, H); _arg = _xres
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
    if REAL_AR and globals().get('GIN'):   # G-a golds from the x the AR delivered to each GDN layer's mixer: rstd(x) * (x [.] γ_L) @ W
        _gx = {}
        if AR_DBG:   # program order of the ARs as in the qkv regold: [attention / GDN AR], MLP AR per layer
            _k = 0
            for L_ in range(NL):
                if not is_attn_layer(L_): _gx[L_] = _db[:, 2 * (_k - 1) + 1] if _k > 0 else _x0_pass
                _k += 1 + (1 if (REAL_AAR and (is_attn_layer(L_) or ('MK_REAL_GAR', '1') in MKDEF)) else 0)
        elif AR_SEQ:
            _gx = {L_: x_ for L_, x_ in _xin_seq.items() if not is_attn_layer(L_)}
        for g_, L_ in enumerate(GLS):
            if L_ not in _gx: continue
            x_ = _gx[L_]
            if x_.shape[0] < M: x_ = torch.cat([x_, torch.zeros(M - x_.shape[0], H)], 0)
            _rs_ = _rstd_of(x_)   # rstd of the un-scaled residual; γ scales the normalised x
            if AR_GAMMA or GAMMA_CONS: x_ = (x_ * AR_GTAB[L_, 0][None, :]).bfloat16().float()
            gold_gin[g_] = (_rs_[:, None] * (x_.double() @ wgfq[g_].double())).float()
        print(f"[gin] iter {it} in-proj golds from the AR-delivered x for GDN layers {sorted(_gx)}" if _gx else f"[gin] iter {it} in-proj golds: static x (no AR chain to read)", flush=True)
    if globals().get('GIN'):   # MK_GDN_INPROJ: the in-proj tiles (bench-only DRAM copy) vs rstd * x @ W_gin per GDN layer
        gv_ = ttnn.to_torch(gin_dbg).float().reshape(TILE, NGL, GN * TILE)
        for g_ in range(NGL):
            d_, gg_ = gv_[:V_R, g_], gold_gin[g_][:V_R]
            pc_ = torch.corrcoef(torch.stack([d_.flatten(), gg_.flatten()]))[0, 1].item()
            pad0 = bool((gv_[:, g_, GNR:] == 0).all())
            nf_ = int((~torch.isfinite(d_)).sum()); nz_ = bool((d_ == 0).all())   # non-finite elements / never written (all 0)
            if os.environ.get('MK_NORM_DIAG') == '1': print(f"[normdiag] iter {it} gin layer {g_} |dev|/|gold| {(d_.norm() / gg_.norm()).item():.4f}", flush=True)
            print(f"[gin] iter {it} GDN layer {g_} (L{GLS[g_]}) in-proj pcc {pc_:.5f} max|err| {(d_ - gg_).abs().max().item():.4f} |gold|max {gg_.abs().max().item():.3f} pad exact 0 {pad0}"
                  f" non-finite {nf_}{' NOT WRITTEN' if nz_ else ''}", flush=True)
    if ymix is not None:   # (MK_L1P2_TENS: under QSK ymix is a static CB, never written: no stand-in mixer output to check)
        pm, em = pcc_mix = (lambda v, g: (torch.corrcoef(torch.stack([v.flatten(), g.flatten()]))[0, 1].item(), (v - g).abs().max().item()))(ttnn.to_torch(ymix).float().reshape(M, -1), gold_mix)
        print(f"[mk1] iter {it} mixer pcc {pm:.5f} max|err| {em:.4f}", flush=True)
    if REAL_AR and AR_RSTD and AR_DBG:   # rstd producer: the Σ tile the pass's last K-row delivery left on the STREAM cores (rows 0-7 = the
        # K-row partials) vs Σx² of that x: the residual after AR (NAR - 2) (the last AR feeds no delivery inside the pass)
        _rs = ttnn.to_torch(_rsp_t).float().reshape(-1, TILE, _rsp_t.shape[-1])   # [cores, 32, NOWN*32], device 0
        _xl = _db[:, 2 * (AR_NAR - 2) + 1].double()
        _gs = (_xl * _xl).sum(-1)
        _gk = torch.stack([(_xl[:, k_ * 640:(k_ + 1) * 640] ** 2).sum(-1) for k_ in range(8)])   # [8, 32]
        _worst = 0.0
        for ci_ in ((0, 9, 37, 79) if not globals().get('RRED') else ()):
            S_ = _rs[ci_, :, :TILE].double()
            _worst = max(_worst, ((S_[:8] - _gk).abs() / _gk.abs().clamp_min(1e-30)).max().item())
        if globals().get('RRED'):   # MK_AR_RSTD_RED=gdn: rsp tile 0 = the FINAL Σ in column 0 (every STREAM core); the K-row partials
            # (the 8 un-reduced K-row tiles sit in the reducer's static CB 37: not read back)
            _sc = torch.stack([_rs[ci_, :, 0].double() for ci_ in (0, 9, 37, 79)])
            _rel = ((_sc - _gs[None]).abs() / _gs[None]).max().item()
            _off = max(_rs[ci_, :, 1:TILE].abs().max().item() for ci_ in (0, 9, 37, 79))
            _rt = (1 / torch.sqrt(_sc / H + EPS)) / (1 / torch.sqrt(_gs[None] / H + EPS)) - 1
            print(f"[ar] iter {it} rstd Σ (red=gdn): columns 1-31 max|.| {_off:.3e}; "
                  f"Σ max rel err {_rel:.3e} (4 cores); rstd max rel err {_rt.abs().max().item():.3e}", flush=True)
        else:
            S0 = _rs[0, :, :TILE].double()
            _rel = ((S0[:8].sum(0) - _gs).abs() / _gs).max().item()
            _rt = (1 / torch.sqrt(S0[:8].sum(0) / H + EPS)) / (1 / torch.sqrt(_gs / H + EPS)) - 1
            print(f"[ar] iter {it} rstd Σ tile: rows 8-31 max|.| {S0[8:].abs().max().item():.3e}; K-row partial max rel err {_worst:.3e} (4 cores); "
                  f"Σ max rel err {_rel:.3e}; rstd max rel err {_rt.abs().max().item():.3e}", flush=True)
    if REAL_AR and RQKV:   # qkv / k / SDPA / wo golds from the x the AR actually delivered to each attention layer (not the static x)
        _xin = {}
        if AR_DBG:   # residual after every AR of this pass (owner dumps); program order: [attention AR], MLP AR per layer
            _k = 0
            for L_ in range(NL):
                if is_attn_layer(L_) and _k > 0: _xin[L_] = _db[:, 2 * (_k - 1) + 1]
                _k += 1 + (1 if (REAL_AAR and (is_attn_layer(L_) or ('MK_REAL_GAR', '1') in MKDEF)) else 0)
        elif AR_SEQ:
            _xin = {L_: x_ for L_, x_ in _xin_seq.items() if is_attn_layer(L_) and L_ > 0}
        for L_, x_ in _xin.items():
            if x_.shape[0] < M: x_ = torch.cat([x_, torch.zeros(M - x_.shape[0], H)], 0)   # R < 32: rows >= R are don't-care
            _rs_ = _rstd_of(x_)   # rstd of the un-scaled residual (RMSNorm), γ applies to the normalised x
            if AR_GAMMA or GAMMA_CONS: x_ = (x_ * AR_GTAB[L_, 0][None, :]).bfloat16().float()   # MK_AR_GAMMA / MK_GAMMA_CONS: the mixer reads x (.) gamma (layer L's input_layernorm)
            attn_regold(L_, (_rs_[:, None] * (x_.double() @ wfq[L_].double())).float())
        print(f"[ar] iter {it} qkv/SDPA golds from the AR-delivered x for layers {sorted(_xin)}" if _xin else f"[ar] iter {it} qkv/SDPA golds: static x (no AR chain: MK_AR_DBG=1 at R = 32)", flush=True)
    if RQKV:
        qv = ttnn.to_torch(qkv_out).float().reshape(NL, M, QN * TILE)
        QRW = V_R if (QSK and V_R < 32) else M   # split-K at R < 32: only rows 0..R-1 of x / the partials travel (v12 faces)
        qv = qv[:, :QRW]
        for L in range(NL):
            if not is_attn_layer(L): continue
            gq_ = gold_qkv[L][:QRW].clone()   # (no in-place sigmoid: attn_regold may replace gold_qkv[L] every pass)
            if RWO or QSK: gq_[:, 64 * TILE:112 * TILE] = torch.sigmoid(gq_[:, 64 * TILE:112 * TILE])
            pq_ = torch.corrcoef(torch.stack([qv[L].flatten(), gq_.flatten()]))[0, 1].item()
            nan_ = [n for n in range(QN) if not torch.isfinite(qv[L][:, n*TILE:(n+1)*TILE]).all()]
            if nan_: print(f"[rqkv] iter {it} layer {L} non-finite tiles {nan_}", flush=True)
            bad_ = [n for n in range(QN) if torch.corrcoef(torch.stack([qv[L][:, n*TILE:(n+1)*TILE].flatten(), gq_[:, n*TILE:(n+1)*TILE].flatten()]))[0, 1].item() < 0.999]
            print(f"[rqkv] iter {it} layer {L} rows {QRW} qkv pcc {pq_:.5f} max|err| {(qv[L]-gq_).abs().max().item():.4f} |gold|max {gq_.abs().max().item():.3f} bad tiles {bad_[:12]}", flush=True)
            if os.environ.get('MK_NORM_DIAG') == '1': print(f"[normdiag] iter {it} layer {L} qkv cols<64T |dev|/|gold| {(qv[L][:, :64 * TILE].norm() / gq_[:, :64 * TILE].norm()).item():.4f}", flush=True)
            if globals().get('GAMMA_CONS') or globals().get('AR_GAMMA'):   # [norm] q/k/v columns (pre-sigmoid, linear in γ): |dev| / |true RMSNorm · γ_model @ W|
                _r = (qv[L][:, :64 * TILE].norm() / (gq_[:, :64 * TILE].norm() / globals().get('NORM_FIX', 1.0))).item()
                print(f"[norm] iter {it} layer {L} qkv |dev| / |RMSNorm(x)*gamma_model @ W| {_r:.4f} {'PASS' if abs(_r - 1.0) <= 0.03 else 'FAIL'}", flush=True)
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
        if ('MKA_RQPOS', '1') in MKDEF:   # per own user (global user g = ug(u)): PCC, finiteness, the user's kvpos rows
            _pu = []
            for uu in range(NU):
                vu, gu_ = v_[uu], g2[uu]
                _ok = bool(torch.isfinite(vu).all())
                _pu.append(f"g{ug(uu)}:{torch.corrcoef(torch.stack([vu.flatten(), gu_.flatten()]))[0, 1].item():.5f}{'' if _ok else '(NONFINITE)'}")
            print(f"[rqpos] iter {it} layer {L} sdpa per user {' '.join(_pu)}", flush=True)
    if RQNR:
        kv_ = ttnn.to_torch(kd_t).float().reshape(NL, M, -1)
        for L in range(NL):
            if not is_attn_layer(L): continue
            KRW = V_R if (QSK and V_R < 32) else M
            pk_ = torch.corrcoef(torch.stack([kv_[L][:KRW].flatten(), gold_k[L][:KRW].flatten()]))[0, 1].item()
            print(f"[rqnr] iter {it} layer {L} rows {KRW} k head pcc {pk_:.5f} max|err| {(kv_[L][:KRW] - gold_k[L][:KRW]).abs().max().item():.4f}", flush=True)
    if ('MKA_RQPOS', '1') in MKDEF and it == ITERS - 1 and os.environ.get('MKA_RQPOS_PROD', '1') == '1' and not MESH:   # last iter: its ops would be the CSV's last run
        # prod's op-by-op q/k chain on device (attention/tp.py dense verify path): rms_norm -> * bf16(1+w) rows ->
        # rotary_embedding_hf on dims 0-63 with this replay's cos / sin -> concat; vs the engine's k head and the torch golds
        _kv_ = ttnn.to_torch(kd_t).float().reshape(NL, M, -1)
        for L in range(NL):
            if not is_attn_layer(L): continue
            a_ = ATT0 + sum(1 for l_ in range(L) if is_attn_layer(l_))
            gq_ = gold_qkv[L].float()
            qk_ = torch.stack([gq_[:, h_ * 256:(h_ + 1) * 256] for h_ in range(6)] + [gq_[:, 48 * TILE:56 * TILE]])[None]   # [1, 7, 32, 256]
            _d = lambda t_, ly_=ttnn.TILE_LAYOUT: ttnn.from_torch(t_.bfloat16(), dtype=ttnn.bfloat16, layout=ly_, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            qkt = _d(qk_)
            gam = _d(torch.stack([qk_g[a_, 0]] * 6 + [qk_g[a_, 1]])[None, :, None, :].expand(1, 7, TILE, 256).contiguous())
            nrm = ttnn.rms_norm(qkt, epsilon=1e-6)
            qkm = ttnn.multiply(nrm, gam)
            part = ttnn.slice(qkm, (0, 0, 0, 0), (1, 7, M, 64))
            rop = ttnn.experimental.rotary_embedding_hf(part, rq_cos, rq_sin, is_decode_mode=False)
            tail = ttnn.slice(qkm, (0, 0, 0, 64), (1, 7, M, 256))
            ref = ttnn.to_torch(ttnn.concat([rop, tail], dim=-1)).float()[0]   # [7, 32, 256]
            gk_ = gold_k[L][:M].float()
            pk_dev = torch.corrcoef(torch.stack([_kv_[L][:M].flatten(), ref[6].flatten()]))[0, 1].item()
            pk_gold = torch.corrcoef(torch.stack([gk_.flatten(), ref[6].flatten()]))[0, 1].item()
            _qg = torch.stack([norm_rope(gq_[:, h_ * 256:(h_ + 1) * 256], h_, L) * 16.0 for h_ in range(6)])
            pq_gold = torch.corrcoef(torch.stack([_qg.flatten(), ref[:6].flatten()]))[0, 1].item()
            print(f"[rqpos] layer {L} PROD-OP chain: engine k head vs prod {pk_dev:.6f} max|err| {(_kv_[L][:M] - ref[6]).abs().max().item():.4f}; "
                  f"torch golds vs prod: k {pk_gold:.6f} q {pq_gold:.6f} (q max|err| {(_qg - ref[:6]).abs().max().item():.4f})", flush=True)
            # SDPA with prod's q (x 1/16 = SDPA scale), bench KV, prod's causal bound per row: vs the engine's SDPA output
            kq_, vq_ = KVQ[L]; _ar = torch.arange(CTX).float()[None, :]; _pp = []
            for uu in range(NU):
                gu_ = ug(uu); Qu = torch.zeros(NQH * T, DHD)
                for h_ in range(6):
                    for j_ in range(4): Qu[4 * h_ + j_] = ref[h_, 4 * gu_ + j_] / 16.0
                _pj = kvpos_l[gu_ * T:(gu_ + 1) * T]; _pm = max([x_ for x_ in _pj if x_ >= 0] or [0])
                _lim = torch.tensor([float(_pj[r_ % T] if _pj[r_ % T] >= 0 else _pm) for r_ in range(NQH * T)])
                p_ = torch.softmax((Qu.bfloat16().float() @ kq_[uu].T).masked_fill(_ar > _lim[:, None], float('-inf')), -1)
                o_ = p_ @ vq_[uu]
                _pc = lambda a_, b_: torch.corrcoef(torch.stack([a_.flatten(), b_.flatten()]))[0, 1].item()
                _pp.append(f"g{gu_}:{_pc(vs[L][uu, :NQH * T], o_):.6f}/{_pc(golds_s[L][uu, :NQH * T], o_):.6f}")
            print(f"[rqpos] layer {L} PROD-OP SDPA (prod q, softmax with -inf causal mask) per user engine/torch-gold vs prod: {' '.join(_pp)}", flush=True)
            for t_ in (qkt, gam, nrm, qkm, part, rop, tail): ttnn.deallocate(t_)
    if ('MKA_RQPOS', '1') in MKDEF and it == ITERS - 1:   # bit-identity fingerprints of this iteration's SDPA / wo outputs
        import hashlib
        _h = lambda t_: hashlib.md5(ttnn.to_torch(t_).float().numpy().tobytes()).hexdigest()[:12]
        print(f"[rqpos] iter {it} md5 sdpa {_h(outs)}" + (f" wo {_h(wod)}" if RWO else ''), flush=True)
    if os.environ.get('MK_L1P2_MD5', '0') == '1':   # spec §20 phase 2 A/B: bit-identity fingerprints of every live engine output, per iteration
        import hashlib
        _hh = lambda t_: hashlib.md5(ttnn.to_torch(t_).float().numpy().tobytes()).hexdigest()[:12]
        _lv = [('sdpa', outs)] + ([('wo', wod)] if RWO else []) + ([('qkv', qkv_out)] if RQKV else []) + ([('xo', _xo_t)] if (V12 and _xo_t is not None) else []) + \
              ([('res', _res_t)] if REAL_AR else []) + \
              [(n_, globals()[n_]) for n_ in ('gin_dbg', 'go_dbg') if globals().get(n_) is not None] + \
              [(f'gio{i_}', t_) for i_, t_ in enumerate(GDN_ROLE.get('io', []) if GDN24 else []) if t_.memory_config().buffer_type == ttnn.BufferType.DRAM and t_.volume() <= (1 << 24)]
        print(f"[l1p2] iter {it} md5 " + ' '.join(f"{n_} {_hh(t_)}" for n_, t_ in _lv), flush=True)
    if MESH and ('MKA_RQPOS', '1') in MKDEF and os.environ.get('MKA_UPAR', '0') == '1':
        # per device d (own users g = 2u + d % 2): SDPA / wo vs that parity's golds -> every user 0-7 covered on the mesh
        _od = [_to_torch(t_).float().reshape(NL, NU, TILE, DHD) for t_ in ttnn.get_device_tensors(outs)]
        _wd = [_to_torch(t_).float().reshape(NL, M, -1) for t_ in ttnn.get_device_tensors(wod)] if RWO else None
        for L in range(NL):
            if not is_attn_layer(L): continue
            _gp = {p_: rq_gold(L, gold_qkv[L], par=p_) for p_ in (0, 1)}
            _ln = []
            for d_, o_ in enumerate(_od):
                gs_, gw_ = _gp[d_ % 2]
                _pu = ' '.join(f"g{2 * uu + d_ % 2}:{torch.corrcoef(torch.stack([o_[L][uu, :NQH * T].flatten(), gs_[uu, :NQH * T].flatten()]))[0, 1].item():.5f}" for uu in range(NU))
                _pw = f" wo {torch.corrcoef(torch.stack([_wd[d_][L].flatten(), gw_.flatten()]))[0, 1].item():.5f}" if RWO else ''
                _ln.append(f"d{d_}[{_pu}{_pw}]")
            print(f"[rqpos] iter {it} layer {L} MESH per device/user sdpa: {' '.join(_ln)}", flush=True)
        # same-parity identity: md5 of each device's SDPA output over the attention layers' valid rows (NQH * T), by parity
        import hashlib
        _al = [L for L in range(NL) if is_attn_layer(L)]
        _md = [hashlib.md5(torch.stack([o_[L][:, :NQH * T] for L in _al]).numpy().tobytes()).hexdigest()[:12] for o_ in _od]
        _byp = {p_: sorted({_md[d_] for d_ in range(len(_md)) if d_ % 2 == p_}) for p_ in (0, 1)}
        print(f"[rqpos] iter {it} MESH sdpa md5 by parity: p0 {_byp[0]} p1 {_byp[1]} -> same-parity identical: "
              f"{'YES' if all(len(v_) == 1 for v_ in _byp.values()) else 'NO'}", flush=True)
    if RWO:
        wv_ = ttnn.to_torch(wod).float().reshape(NL, M, -1)
        for L in range(NL):
            if not is_attn_layer(L): continue
            rows_ = slice(0, M) if os.environ.get('MKA_UPAR', '0') == '1' else slice(0, 4 * NU)
            pw_ = torch.corrcoef(torch.stack([wv_[L][rows_].flatten(), gold_wo[L][rows_].flatten()]))[0, 1].item()
            print(f"[rwo] iter {it} layer {L} wo pcc {pw_:.5f} max|err| {(wv_[L][rows_] - gold_wo[L][rows_]).abs().max().item():.4f} |gold|max {gold_wo[L].abs().max().item():.3f}", flush=True)
    print(f"[fm11] iter {it} wall {time.time()-t1:.2f}s last-layer pcc {p:.5f} max|err| {e:.4f} |gold|max {gold_last.abs().max().item():.3f}", flush=True)
    if MESH:
        # dead buffers are skipped: out (fm11 output) is never written under V12, ymix (MIXN mixer) never under QSK; the
        # residual identity is the [ar] line (ar_out), qkv / v12 xo / sdpa are compared here when they are live
        _mi = ([] if V12 else [('out', y)]) + ([] if QSK else [('ymix', ymix)]) + ([('qkv', qkv_out)] if RQKV else []) + \
              ([('v12 xo', _xo_t)] if V12 else []) + [('sdpa', outs)]
        def _mid_r(t_):   # R < 32: valid rows only (rows >= R of each 32-row tile row are don't-care, spec v0.8)
            ds = [_to_torch(d).float() for d in ttnn.get_device_tensors(t_)]
            ds = [d_.reshape(-1, TILE, d_.shape[-1])[:, :V_R] for d_ in ds]
            return max((d_ - ds[0]).abs().max().item() for d_ in ds)
        _mif = mesh_identity if V_R == 32 else (lambda t_: _mid_r(t_) if t_ is not outs else mesh_identity(t_))
        print(f"[mesh] iter {it} devices {dev.get_num_devices()}: max |dev d - dev 0|{'' if V_R == 32 else f' (rows < {V_R})'} " + ' '.join(f"{n_} {_mif(t_):.3e}" for n_, t_ in _mi), flush=True)
def _prof_decollide():
    """The device profiler keys zones by a 16-bit hash of 'name,__FILE__,line'. __FILE__ holds the per-run pod dir, so with
    ~300 zone sites a run collides with probability ~0.5 and the host readout throws ("Source location hashes are colliding",
    rc 134, no CSV). Before the readout keep ONE location per colliding hash in the zone logs (the layer-timing zones win);
    the dropped zone's markers are then reported under the kept name. Prints what it dropped."""
    d = os.environ.get('TT_METAL_PROFILER_DIR')
    if not d: return
    keep_pref = ("W-ARSTALL", "W-ARWAIT", "W-MIX", "C-MIX", "W-SDPA", "C-SDPA", "R-KV", "R-RED", "W-WOAR", "C-GU", "C-DN", "W-HWAIT",
                 "W-VX", "W-VMD", "W-OREL", "C-OSUM", "W-AR", "C-AR", "R-ARDLVW", "R-ARDLV", "C-ARG", "W-ARG")
    def h16(s_):
        h_ = 2166136261
        for c_ in s_.encode():
            h_ = ((h_ ^ c_) * 16777619) & 0xFFFFFFFF
        return ((h_ & 0xFFFF) ^ (h_ >> 16)) & 0xFFFF
    dl = "'#pragma message: "
    files = [os.path.join(d, '.logs', f_) for f_ in ('zone_src_locations.log', 'new_zone_src_locations.log')]
    locs = {}
    for f_ in files:
        if not os.path.exists(f_): continue
        for ln in open(f_):
            i_ = ln.find(dl)
            if i_ < 0: continue
            loc = ln[i_ + len(dl):].rstrip('\n')[:-1]
            locs.setdefault(h16(loc), []).append(loc)
    drop = set()
    for h_, ls in locs.items():
        u_ = list(dict.fromkeys(ls))
        if len(u_) < 2: continue
        u_.sort(key=lambda l_: 0 if l_.split(',')[0] in keep_pref else 1)
        drop.update(u_[1:])
        print(f"[prof] zone hash {h_:#06x} collision: keep {u_[0].split(',')[0]}@{u_[0].split(',')[2]}, drop {[l_.split(',')[0] + '@' + l_.split(',')[2] for l_ in u_[1:]]}", flush=True)
    if not drop: return
    for f_ in files:
        if not os.path.exists(f_): continue
        lines = open(f_).readlines()
        open(f_, 'w').writelines(l_ for l_ in lines if not (l_.find(dl) >= 0 and l_[l_.find(dl) + len(dl):].rstrip('\n')[:-1] in drop))


# MK_WALL_N=N: unprofiled wall clock of the engine step, no golds / readback: N back-to-back launches (+ the per-launch
# ENGINE_SEED / ENGINE_DRAIN ops when MK_L1P2_SEED is on), one synchronize; MK_WALL_TRACE=1: the same as a captured trace
# replayed N times (no host dispatch on the path). Positions / kvpos stay at the last iteration's.
_WN = int(os.environ.get('MK_WALL_N', '0'))
if _WN and 'a' in _WALL:
    def _step():
        if ENGINE_SEED is not None: ENGINE_SEED()
        _go_wall0(*_WALL['a'])
        if ENGINE_DRAIN is not None: ENGINE_DRAIN()
    ttnn.synchronize_device(dev)
    for _rep in range(2):
        t_ = time.perf_counter()
        for _ in range(_WN): _step()
        ttnn.synchronize_device(dev)
        print(f"[wall] enqueue rep {_rep}: {_WN} launches {1e6 * (time.perf_counter() - t_) / _WN:.1f} us/launch (incl. host dispatch)", flush=True)
    if os.environ.get('MK_WALL_TRACE', '0') == '1':
        _tid = ttnn.begin_trace_capture(dev, cq_id=0)
        _step()
        ttnn.end_trace_capture(dev, _tid, cq_id=0)
        ttnn.synchronize_device(dev)
        for _rep in range(2):
            t_ = time.perf_counter()
            for _ in range(_WN): ttnn.execute_trace(dev, _tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            print(f"[wall] trace rep {_rep}: {_WN} replays {1e6 * (time.perf_counter() - t_) / _WN:.1f} us/launch", flush=True)
        ttnn.release_trace(dev, _tid)

_prof_decollide()
ttnn.ReadDeviceProfiler(dev)
print('[fm11] profiler read', flush=True)
ttnn.close_device(dev)
print('[fm11] closed', flush=True)
