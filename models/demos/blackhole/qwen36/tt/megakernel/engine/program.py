"""Megakernel engine program setup (integration track; spec §16 merge token, v1.22 -mk-int).

Step 1 of the program.py factoring is a PURE refactor: ``setup_bench`` is attn/bench_mka2.py's setup body moved
verbatim into a function (device placement, weight / KV / activation tensors, kernel descriptors, ``build()``), so the
bench is a thin wrapper and every engine flag behaves exactly as before. Step 2 turns it into the integration API
(``weight_layouts`` / ``build_program`` of integ/DESIGN_integration.md §11): real weights from weights_mk.repack,
the vLLM KV caches / page table / kvpos from per_replay, x_in / x_out instead of the synthetic x.

Step 2 rules (spec v1.25): the runtime-arg region and the CB-config block (sized by the max CB index) are program-wide
(dispatch.cpp), so every core pays for the largest and STREAM is ~40 B from the 70656 B kernel-config limit. Per-core
tables (AR roles, v12 slot tables, GDN page maps, the phase table) therefore travel as per-kind "arg blocks": one
sharded L1 (or DRAM) tensor per kind, its address in a single runtime arg, read once at kernel init. CB ids < 45.

``setup_bench(dev, MESH, mesh_identity)`` returns the setup's names (prog, the io tensors, golds, flags) as a dict.
The caller (bench) must already have installed its mesh shims (ttnn.from_torch / to_torch / generic_op) when MESH.
"""
import math, os, time, torch, ttnn
KD = os.environ.get("MKA_KD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "attn", "kernels"))
TILE, H, I, M = 32, 5120, 2176, 32
KT, HT = H // TILE, I // TILE   # 160, 68
GX, GY = 10, 8; NC = GX * GY; NGU = HT; NDN = KT // NC; XSW = KT // NC
KB = 16; GRP = int(os.environ.get("FM_GRP", "2")); ITERS = int(os.environ.get("FM_ITERS", "2"))


def setup_bench(dev, MESH, mesh_identity):
    try:   # engine/ is on sys.path in the bench (the pushed tree's own module first)
        import cbplan as CBP
    except ImportError:
        from models.demos.blackhole.qwen36.tt.megakernel.engine import cbplan as CBP
    CBP.track(ttnn)   # spec §20 v12cb: the CB plan must know which descriptors are tensor-backed
    banks = dev.dram_grid_size().x
    cores = [ttnn.CoreCoord(i % GX, i // GX) for i in range(NC)]
    rect = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
    def crs(cs):
        # merge into as few rectangles as possible: dispatch writes each kernel binary once per range
        rows = {}
        for c in cs: rows.setdefault(c.y, []).append(c.x)
        runs = []
        for y in sorted(rows):
            xs = sorted(rows[y]); st = pv = xs[0]
            for xx in xs[1:]:
                if xx != pv + 1: runs.append((st, pv, y)); st = xx
                pv = xx
            runs.append((st, pv, y))
        rects = []
        for x0, x1, y in runs:
            if rects and rects[-1][0] == x0 and rects[-1][1] == x1 and rects[-1][3] == y - 1: rects[-1][3] = y
            else: rects.append([x0, x1, y, y])
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, c), ttnn.CoreCoord(b, d)) for a, b, c, d in rects])
    p0 = dev.worker_core_from_logical_core(cores[0])
    # two multicast rectangles: logical columns whose NOC x is contiguous (the DRAM columns split the Tensix block)
    phys_x = [dev.worker_core_from_logical_core(ttnn.CoreCoord(i, 0)).x for i in range(GX)]
    split = next((i for i in range(1, GX) if phys_x[i] != phys_x[i - 1] + 1), GX)
    def prect(lx0, lx1):
        if lx1 < lx0: return [0, 0, 0, 0, 0]
        a = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx0, 0)); b = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx1, GY - 1))
        return [a.x, a.y, b.x, b.y, (lx1 - lx0 + 1) * GY]
    RECTS = prect(0, split - 1) + prect(split, GX - 1)
    print(f"[fm11] noc x {phys_x} split at logical {split} rects {RECTS}", flush=True)
    torch.manual_seed(0)
    x_t = torch.randn(M, H); wg_t = torch.randn(H, I) / math.sqrt(H); wu_t = torch.randn(H, I) / math.sqrt(H); wd_t = torch.randn(I, H) / math.sqrt(I)
    def tl(m, r, c): return m[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
    BANK_LOC = [(c.x, c.y) for c in dev.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
    def assign(n_items, cap_total, caps=None):
        caps = list(caps) if caps else [cap_total // banks + (1 if b < cap_total % banks else 0) for b in range(banks)]
        pairs = sorted(((abs(cores[i].x - bx) + abs(cores[i].y - by), i, b) for i in range(n_items) for b, (bx, by) in enumerate(BANK_LOC)))
        out = {}
        for d, i, b in pairs:
            if i in out or caps[b] == 0: continue
            out[i] = b; caps[b] -= 1
        return out
    PLACE = os.environ.get("FM_PLACE", "near") == "near"
    gu_bank = assign(HT, HT) if PLACE else {q: (q * banks) // HT for q in range(HT)}
    _gu_n = [sum(1 for q in range(HT) if gu_bank[q] == b) for b in range(banks)]
    # FM_BAL=1: give banks with more gate|up slices fewer down slices so every bank streams ~the same bytes
    _dn_caps = None
    if os.environ.get("FM_BAL") == "1":
        _tot = HT * 2 + NC * NDN * HT / KT  # in gate|up-slice units: gu slice = KT*2 tiles, dn slice = HT*NDN tiles
        _per = (HT * KT * 2 + NC * HT * NDN) / banks
        _dn_caps = [max(0, round((_per - n * KT * 2) / (HT * NDN))) for n in _gu_n]
        while sum(_dn_caps) > NC: _dn_caps[_dn_caps.index(max(_dn_caps))] -= 1
        while sum(_dn_caps) < NC: _dn_caps[_dn_caps.index(min(_dn_caps))] += 1
        print(f"[fm11] bal dn caps {_dn_caps} for gu {_gu_n}", flush=True)
    dn_bank = assign(NC, NC, _dn_caps) if PLACE else {c: c % banks for c in range(NC)}
    per_bank = [[] for _ in range(banks)]
    for q in range(HT): per_bank[gu_bank[q]].append(q)
    slot = {q: (b, j * KT * 2) for b, qs in enumerate(per_bank) for j, q in enumerate(qs)}
    gu_pages = torch.zeros(max(map(len, per_bank)) * KT * 2 * banks, TILE, TILE)
    for q in range(HT):
        b, s0 = slot[q]
        for kt in range(KT):
            gu_pages[(s0 + 2 * kt) * banks + b] = tl(wg_t, kt, q); gu_pages[(s0 + 2 * kt + 1) * banks + b] = tl(wu_t, kt, q)
    _dn_rank = {}
    for b in range(banks):
        for j, c in enumerate([c for c in range(NC) if dn_bank[c] == b]): _dn_rank[c] = j
    dn_pages = torch.zeros((max(_dn_rank.values()) + 1) * HT * NDN * banks, TILE, TILE)
    def dn_slot(c): return dn_bank[c], _dn_rank[c] * HT * NDN
    for c in range(NC):
        b, s0 = dn_slot(c)
        for kt in range(HT):
            for j in range(NDN): dn_pages[(s0 + kt * NDN + j) * banks + b] = tl(wd_t, kt, c * NDN + j)
    def pages_tensor(p): return ttnn.from_torch(p.reshape(1, 1, -1, TILE), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    wgu, wd = pages_tensor(gu_pages), pages_tensor(dn_pages)
    gq, dq = ttnn.to_torch(wgu).float().reshape(-1, TILE, TILE), ttnn.to_torch(wd).float().reshape(-1, TILE, TILE)
    wgq, wuq, wdq = torch.zeros(H, I), torch.zeros(H, I), torch.zeros(I, H)
    for q in range(HT):
        b, s0 = slot[q]
        for kt in range(KT):
            wgq[kt*TILE:(kt+1)*TILE, q*TILE:(q+1)*TILE] = gq[(s0 + 2*kt) * banks + b]; wuq[kt*TILE:(kt+1)*TILE, q*TILE:(q+1)*TILE] = gq[(s0 + 2*kt + 1) * banks + b]
    for c in range(NC):
        b, s0 = dn_slot(c)
        for kt in range(HT):
            for j in range(NDN): wdq[kt*TILE:(kt+1)*TILE, (c*NDN+j)*TILE:(c*NDN+j+1)*TILE] = dq[(s0 + kt*NDN + j) * banks + b]
    MIXN = int(os.environ.get("MK_MIXN", "1"))
    wm_t = torch.randn(H, NC * MIXN * TILE) / math.sqrt(H)
    mix_bank = assign(NC, NC)
    _mx_rank = {}
    for b in range(banks):
        for j, c in enumerate([c for c in range(NC) if mix_bank[c] == b]): _mx_rank[c] = j
    mx_pages = torch.zeros((max(_mx_rank.values()) + 1) * KT * MIXN * banks, TILE, TILE)
    def mx_slot(c): return mix_bank[c], _mx_rank[c] * KT * MIXN
    for c in range(NC):
        b, s0 = mx_slot(c)
        for kt in range(KT):
            for j in range(MIXN): mx_pages[(s0 + kt * MIXN + j) * banks + b] = tl(wm_t, kt, c * MIXN + j)
    def ws(r, w): return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(r, (M, w), ttnn.ShardOrientation.ROW_MAJOR))
    def hs(r, w): return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(r, (TILE, w), ttnn.ShardOrientation.ROW_MAJOR))
    x = ttnn.from_torch(x_t.reshape(1, 1, M, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ws(rect, XSW * TILE))
    xb = ttnn.to_torch(x).float().reshape(M, H)
    gold = (torch.nn.functional.silu(xb @ wgq) * (xb @ wuq)).bfloat16().float() @ wdq
    QSK = 'MKA_QSK=1' in os.environ.get('MK_DEFINES', '').split(',')
    # mesh: freed mesh buffers are not reliably reusable before the program runs, so with RQKV (which replaces x with a
    # full-grid copy below) the 80-core copy is never allocated
    _SKIP_XBUF80 = MESH and 'MKA_RQKV=1' in os.environ.get('MK_DEFINES', '')
    xbuf = None if _SKIP_XBUF80 else ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TILE * NC, TILE if QSK else H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, hs(rect, TILE if QSK else H))
    XPRE = 1 if os.environ.get("FM_XPRE", "1") == "1" else 0
    if XPRE and not _SKIP_XBUF80 and not QSK:
        if MESH: ttnn.deallocate(xbuf)  # mesh tensors are not freed on rebinding
        xbuf = ttnn.from_torch(xb.reshape(1, 1, M, H).bfloat16().float().repeat(1, 1, NC, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=hs(rect, H))
    HDT = ttnn.bfloat8_b if os.environ.get('MK_H_BFP8') == '1' else ttnn.bfloat16  # -mk-tail: h broadcast in BFP8 halves its bytes
    print(f'[mk1] h dtype {HDT}', flush=True)
    # MK_V12: this fm11 hbuf is replaced by v12's placeholder below; in L1 it left a 136 KB hole above ymix / rsp / xbuf
    # (engine-ON d27u08: persistent floor 1323328 vs bucket 8's CB end 1357696 on rows 8-9), so it goes to DRAM
    _HB_DRAM = any(d_ in os.environ.get('MK_DEFINES', '').split(',') for d_ in ('MK_V12=1', 'MKA_QSK=1', 'MK_V12_CB=1'))   # (MKA_QSK implies MK_V12)
    hbuf = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TILE * NC, I]), HDT, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG if _HB_DRAM else hs(rect, I))
    WTB = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat4_b); XTB = ttnn.Tile([TILE, TILE]).get_tile_size(HDT)
    def cbd(idx, dt, pages, r):
        page = ttnn.Tile([TILE, TILE]).get_tile_size(dt)
        return ttnn.CBDescriptor(total_size=pages * page, core_ranges=r, format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)])
    acc = ttnn.TensorAccessorArgs(wgu).get_compile_time_args() + ttnn.TensorAccessorArgs(wd).get_compile_time_args()
    print(f"[fm11] xpre {XPRE} place {'near' if PLACE else 'rr'} gu/bank {[len(q) for q in per_bank]} cores {NC} gate_up {NGU} banks {banks} bytes gu {HT*KT*2*WTB/1e6:.2f}MB dn {HT*KT*WTB/1e6:.2f}MB grp {GRP}", flush=True)

    NL = int(os.environ.get("FM_NL", "2")); MHZ = 1350.0
    # NL distinct layers: layer L = layer 0 with up tiles scaled 2^(L%2) and down tiles 2^-(L%3) (exact in BFP4), so a
    # core that mixes h or down weights across layers breaks the PCC.
    PGU, PDN = gu_pages.shape[0], dn_pages.shape[0]
    up_mask = torch.zeros(PGU, 1, 1)
    for q in range(HT):
        b, s0 = slot[q]
        for kt in range(KT): up_mask[(s0 + 2 * kt + 1) * banks + b] = 1
    def gu_L(L): return gq.reshape(PGU, TILE, TILE) * (1 + up_mask * (2.0 ** (L % 2) - 1))  # exact BFP4 values
    def dn_L(L): return dq.reshape(PDN, TILE, TILE) * (2.0 ** -(L % 3))
    wgu_all = ttnn.from_torch(torch.cat([gu_L(L).bfloat16() for L in range(NL)]).reshape(1, 1, -1, TILE), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    wd_all = ttnn.from_torch(torch.cat([dn_L(L).bfloat16() for L in range(NL)]).reshape(1, 1, -1, TILE), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    PWM = mx_pages.shape[0]
    wm_all = ttnn.from_torch(torch.cat([(mx_pages * (2.0 ** (L % 2))).bfloat16() for L in range(NL)]).reshape(1, 1, -1, TILE), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mq = ttnn.to_torch(wm_all).float().reshape(-1, TILE, TILE)[(NL - 1) * PWM:NL * PWM]
    wmq = torch.zeros(H, NC * MIXN * TILE)
    for c in range(NC):
        b, s0 = mx_slot(c)
        for kt in range(KT):
            for j in range(MIXN): wmq[kt*TILE:(kt+1)*TILE, (c*MIXN+j)*TILE:(c*MIXN+j+1)*TILE] = mq[(s0 + kt*MIXN + j) * banks + b]
    gold_mix = xb @ wmq
    ymix = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, NC * MIXN * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ws(rect, MIXN * TILE))
    S_MIX = int(float(os.environ.get("MK_SMIX_US", "67")) * MHZ); S_AR = int(float(os.environ.get("MK_SAR_US", "18")) * MHZ)
    print(f"[mk1] MIXN {MIXN} mixer bytes/layer {NC*KT*MIXN*WTB/1e6:.2f}MB stalls mix {S_MIX} ar {S_AR} cycles", flush=True)
    LL = NL - 1
    gold_last = gold * (2.0 ** (LL % 2)) * (2.0 ** -(LL % 3))
    w2_tiles = sum(min(GRP, KT // KB - b0) * KB * 2 for b0 in range(GRP, KT // KB, 2 * GRP))
    print(f"[fm11] NL {NL} pages/layer gu {PGU} dn {PDN} bytes/layer {(HT*KT*2*WTB + HT*KT*WTB)/1e6:.2f}MB w2_tiles {w2_tiles}", flush=True)
    ROOTS = [NC - 4, NC - 3, NC - 2, NC - 1]
    RXY = []
    for ri in ROOTS:
        pc_ = dev.worker_core_from_logical_core(cores[ri]); RXY += [pc_.x, pc_.y]
    ROLE = {ROOTS[k]: k for k in range(4)}
    HCH = HT // 4
    acc_gu = ttnn.TensorAccessorArgs(wgu_all).get_compile_time_args(); acc_dn = ttnn.TensorAccessorArgs(wd_all).get_compile_time_args()
    acc_wm = ttnn.TensorAccessorArgs(wm_all).get_compile_time_args()
    MKDEF = [tuple(kv.split('=', 1)) for kv in os.environ.get('MK_DEFINES', '').split(',') if kv]  # e.g. MKA_NOMLP=1  # -mk-tail: e.g. MK_DNPIPE=1,MK_FLAG_FLUSH=1
    AMASK = int(os.environ.get('MK_AMASK', '0xFFFFFFFF'), 0)   # mixed groups: bit L = attention-type layer
    S_GDN_US = float(os.environ.get('MK_SGDN_US', '0'))
    QSK = ('MKA_QSK', '1') in MKDEF   # spec §18: qkv split-K on STREAM through v12's unit (implies MK_V12 + MK_RS_QKV)
    if QSK:
        for k_ in ('MK_V12', 'MK_RS_QKV', 'MKA_RQKV'):
            if (k_, '1') not in MKDEF: MKDEF = MKDEF + [(k_, '1')]
        QSK_GK = int(os.environ.get('QSK_GK', '6')); MKDEF = MKDEF + [('QSK_GK', str(QSK_GK))]
    # spec v1.67: MK_V12_CB is the default under MK_V12 (v12's landing buffers as role-local CBs, §20); MK_V12_CB=0 opts out
    if ('MK_V12', '1') in MKDEF and not any(k_ == 'MK_V12_CB' for k_, _ in MKDEF):
        MKDEF = MKDEF + [('MK_V12_CB', '1')]
    MKDEF += [('MKA_AMASK', f'{AMASK & 0xFFFFFFFF}u'), ('MKA_S_GDN', str(int(S_GDN_US * 1350)))]
    if os.environ.get('MKA_EXT_LEANZ', '0') == '1': MKDEF += [('MKA_EXT_LEANZ', '1')]   # profiled runs: EXT DM zones out (mka2_common.hpp MKZ)
    print(f'[mk1] defines {MKDEF}', flush=True)
    def is_attn_layer(L): return (AMASK >> L) & 1

    # ================= MKA2: full grid + SDPA phase setup =================
    gfull = dev.compute_with_storage_grid_size(); GXF, GYF = gfull.x, gfull.y; NCF = GXF * GYF
    fcores = [ttnn.CoreCoord(i % GXF, i // GXF) for i in range(NCF)]
    fgrid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GXF - 1, GYF - 1))])
    fphys = [dev.worker_core_from_logical_core(c) for c in fcores]
    def frect(lx0, lx1):
        if lx1 < lx0: return [0, 0, 0, 0, 0]
        a_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx0, 0)); b_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx1, GYF - 1))
        return [a_.x, a_.y, b_.x, b_.y, (lx1 - lx0 + 1) * GYF]
    pxf = [dev.worker_core_from_logical_core(ttnn.CoreCoord(i, 0)).x for i in range(GXF)]
    fsplit = next((i for i in range(1, GXF) if pxf[i] != pxf[i - 1] + 1), GXF)
    RECTS_F = frect(0, fsplit - 1) + frect(fsplit, GXF - 1) + [0, 0, 0, 0, 0]   # 3 full-grid multicast rects (see below)
    NU, NQH, T, DHD = int(os.environ.get('MKA_NU', '4')), 6, 4, 256
    CTX = int(os.environ.get("MKA_CTX", "32768")); NBK = CTX // TILE; NBB = int(os.environ.get("MKA_NBB", "1"))   # 1 beats 2 by ~6 µs/layer (less NoC/DRAM congestion); 3 overflows STREAM L1 with GUPRE
    KVDT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get("MKA_KVDT", "bfp8")]
    KVTB = ttnn.Tile([TILE, TILE]).get_tile_size(KVDT); QTB = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat16)
    OTB = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.float32); WOTB = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat8_b)
    BUDGET = int(os.environ.get("MKA_BUDGET", "0")); FAN = 6 if NU >= 4 else 11   # 2-level tree: group <= FAN*FAN cores
    MKDEF = MKDEF + [("MKA_NU", str(NU)), ("MKA_FAN", str(FAN))]
    # qkv (wqkvg_pair) is 19.5 MB BFP8 per chip; the MIXN mixer stand-in streams NC*KT*MIXN BFP4 tiles. The rest is read as
    # extra scratch tiles per STREAM core during SDPA (next layer's qkv) so the per-layer DRAM bytes stay faithful.
    EXTRA_T = max(0, int((19.5e6 - NC * KT * MIXN * WTB) / NC / ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat8_b)))
    EXTRA_TX = 0
    if os.environ.get('MKA_EXT', '0') == '1':
        _arc0 = os.environ.get('MKA_ARCORES', '106,107,118,119')   # active-core count (reserved AR corner / GDN24 rows excluded)
        _nact = NCF - (2 * GXF if os.environ.get('MKA_GDN24', '0') == '1' else (0 if _arc0 in ('', 'none') else len(_arc0.split(','))))
        _w = float(os.environ.get('MKA_EXTW', '1')); _d = NC + (_nact - NC) * _w
        EXTRA_T, EXTRA_TX = int(EXTRA_T * NC / _d), int(EXTRA_T * NC * _w / _d)
    S_WO = int(float(os.environ.get("MKA_SWO_US", "10")) * MHZ)
    # MKA_ARCORES="fi,fi,...": AR-only cores (spec: the AR's 4 fabric link-sender cores) - no MKA2 kernel, no KV, no qkv
    _arc = os.environ.get('MKA_ARCORES', '106,107,118,119')   # default ON (spec v0.9.1); 'none' disables
    ARC = set() if _arc in ('', 'none') else set(int(v) for v in _arc.split(','))
    # MKA_GDN24 (spec §17 / §20): rows 8-9 (x 0-11, 24 cores, row-major = 6 heads x 4 value blocks) are the GDN role's
    # cores. They are RESERVED like the AR corner: no MKA2 kernel, CB, sharded L1 tensor, global semaphore, SDPA / KV /
    # qkv / wo share, or multicast destination; their kernels come only from the GDN hook (engine/gdn_role.py). During
    # bring-up the AR corner (x 10-11, rows 8-9) is GDN-role too (MK_REAL_AR + GDN24 later, AR + GDN combined).
    GDN24 = os.environ.get('MKA_GDN24', '0') == '1'
    GDNC = []
    if GDN24:
        assert GXF == GX + 2 and GYF == GY + 2, "MKA_GDN24 assumes the 12 x 10 grid (rows 8-9 = the 24 GDN cores)"
        # MK_REAL_AR + GDN24: the AR corner (x 10-11, rows 8-9) runs the GDN role with the AR sender phases hosted in its
        # reader / writer (see _gdn_host_ar below)
        GDNC = [i for i in range(NCF) if fcores[i].y >= GY]   # row-major: y 8 then 9, x 0..11
        ARC = set()
    RESV = ARC | set(GDNC)
    ACT = [i for i in range(NCF) if i not in RESV]
    grp = [0] * NCF
    for j_, i_ in enumerate(ACT): grp[i_] = j_ * NU // len(ACT)
    members = [[i for i in ACT if grp[i] == u] for u in range(NU)]
    if os.environ.get('MKA_MREV') == '1': members = [m_[::-1] for m_ in members]   # rank 0 / first KV blocks at the far end of each user's rows
    if ('MKA_RMASK', '1') in MKDEF:   # the user's last KV block (causal mask) must sit on a non-STREAM core: move one last
        _sx = lambda f_: fcores[f_].x < GX and fcores[f_].y < GY
        for m_ in members:
            e_ = [f_ for f_ in m_ if not _sx(f_) and f_ != m_[0]]
            assert e_, "a KV group without a non-STREAM core"
            m_.remove(e_[-1]); m_.append(e_[-1])
    rank = {i: members[grp[i]].index(i) for i in ACT}
    kgrid = crs([fcores[i] for i in ACT]) if RESV else fgrid
    GSEM = [ttnn.create_global_semaphore(dev, kgrid, 0) for _ in range(23)]
    GSA = [ttnn.get_global_semaphore_address(g) for g in GSEM]
    if ARC:
        # spec v0.9.1: the AR sender cores are not in ANY mka2 range (kernels, CBs, sharded L1 tensors, global semaphores,
        # multicasts). Full-grid multicasts: left block, right block above the AR corner, right block beside the AR corner.
        _ax = sorted({fcores[i].x for i in ARC}); _ay = sorted({fcores[i].y for i in ARC})
        assert _ax[-1] == GXF - 1 and _ay[-1] == GYF - 1 and len(ARC) == len(_ax) * len(_ay), "AR corner must be the bottom-right block"
        def _prect(lx0, ly0, lx1, ly1):
            if lx1 < lx0 or ly1 < ly0: return [0, 0, 0, 0, 0]
            a_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx0, ly0)); b_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx1, ly1))
            return [a_.x, a_.y, b_.x, b_.y, (lx1 - lx0 + 1) * (ly1 - ly0 + 1)]
        RECTS_F = (_prect(0, 0, fsplit - 1, GYF - 1) + _prect(fsplit, 0, GXF - 1, _ay[0] - 1) + _prect(fsplit, _ay[0], _ax[0] - 1, GYF - 1))
    if GDN24:   # full-grid multicasts cover rows 0-7 only (left / right of the DRAM split), third rect empty
        def _prect2(lx0, ly0, lx1, ly1):
            a_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx0, ly0)); b_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx1, ly1))
            return [a_.x, a_.y, b_.x, b_.y, (lx1 - lx0 + 1) * (ly1 - ly0 + 1)]
        RECTS_F = _prect2(0, 0, fsplit - 1, GY - 1) + _prect2(fsplit, 0, GXF - 1, GY - 1) + [0, 0, 0, 0, 0]
        assert sum(RECTS_F[4::5]) == len(ACT), (RECTS_F, len(ACT))
        print(f'[mka2] GDN24: reserved {len(GDNC)} GDN cores (rows {GY}-{GYF - 1}); {len(ACT)} MKA2 cores; RECTS_F {RECTS_F}', flush=True)
    if ARC: print(f'[mka2] AR-only cores {sorted(ARC)} (phys {[(fphys[i].x, fphys[i].y) for i in sorted(ARC)]}): {len(ACT)} MKA2 cores', flush=True)
    stream_idx = {}   # full-grid core -> STREAM index (mk1 numbering, x + 10*y on the 10x8 rect)
    for i, c in enumerate(fcores):
        if c.x < GX and c.y < GY: stream_idx[i] = c.x + GX * c.y
    torch.manual_seed(1)
    kvs, golds_s, qs_all = [], [], []
    KVQ = {}
    RQNR = ('MKA_RQNR', '1') in MKDEF
    KVPROD = os.environ.get('MKA_KVPROD', '0') == '1'
    NKVH, KVH, KVBS = int(os.environ.get('MKA_NKVH', '4')), int(os.environ.get('MKA_KVH', '1')), int(os.environ.get('MKA_BS', '64'))
    BS32 = KVBS // TILE; BPU = NBK // BS32
    kvvs = []   # paged_v per layer (KVPROD); kvs holds paged_k (KVPROD) or the interleaved K|V tensor
    if KVPROD:
        assert NBK % BS32 == 0 and 0 <= KVH < NKVH
        MKDEF = MKDEF + [('MKA_KVPROD', '1'), ('MKA_NKVH', str(NKVH)), ('MKA_BS32', str(BS32)), ('MKA_KVH', str(KVH)), ('MKA_BPU', str(BPU))]
        print(f"[mka2] KV prod layout: paged_k / paged_v [{NU * BPU}, {NKVH}, {KVBS}, {DHD}] per layer, head {KVH}", flush=True)
        def to_paged(t):   # [NU, CTX, DHD] -> [NU*BPU, NKVH, KVBS, DHD] with head KVH filled, identity page table
            full = torch.zeros(NU * BPU, NKVH, KVBS, DHD, dtype=torch.bfloat16)
            full[:, KVH] = t.reshape(NU * BPU, KVBS, DHD).bfloat16()
            return full
        def from_paged(tt):
            return ttnn.to_torch(tt).float().reshape(NU * BPU, NKVH, KVBS, DHD)[:, KVH].reshape(NU, CTX, DHD)
    for L in range(NL):
        if KVPROD and kvs and not is_attn_layer(L):   # GDN-type layers never read KV: share the previous layer's tensors
            q_t = torch.zeros(NU, TILE, DHD); kvs.append(kvs[-1]); kvvs.append(kvvs[-1]); golds_s.append(golds_s[-1]); qs_all.append(q_t); continue
        q_t = torch.zeros(NU, TILE, DHD); q_t[:, :NQH * T] = torch.randn(NU, NQH * T, DHD) / math.sqrt(DHD)
        k_t = torch.randn(NU, CTX, DHD); v_t = torch.randn(NU, CTX, DHD)
        if KVPROD:
            kv = ttnn.from_torch(to_paged(k_t), dtype=KVDT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kvv_ = ttnn.from_torch(to_paged(v_t), dtype=KVDT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kq_, vq_ = from_paged(kv), from_paged(kvv_); kvvs.append(kvv_); rows = kvq = None
        else:
            rows = torch.stack([k_t.reshape(NU, NBK, TILE, DHD), v_t.reshape(NU, NBK, TILE, DHD)], dim=2).reshape(1, 1, -1, DHD)
            kv = ttnn.from_torch(rows.bfloat16(), dtype=KVDT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kvq = ttnn.to_torch(kv).float().reshape(NU, NBK, 2, TILE, DHD)
            kq_, vq_ = kvq[:, :, 0].reshape(NU, CTX, DHD), kvq[:, :, 1].reshape(NU, CTX, DHD)
        qq_ = q_t.bfloat16().float()
        g_ = torch.zeros(NU, TILE, DHD)
        for u in range(NU):
            p = torch.exp(qq_[u] @ kq_[u].T); g_[u] = (p @ vq_[u]) / p.sum(-1, keepdim=True)
        if RQNR: KVQ[L] = (kq_.clone(), vq_.clone())
        if os.environ.get('MKA_SAMEDATA') == '1' and L > 0: kv, g_, q_t = kvs[0], golds_s[0], qs_all[0]
        kvs.append(kv); golds_s.append(g_); qs_all.append(q_t)
        del rows, kvq, kq_, vq_
    qten = ttnn.from_torch(torch.cat(qs_all).reshape(1, 1, NL * NU * TILE, DHD).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    outs = ttnn.from_torch(torch.zeros(1, 1, NL * NU * TILE, DHD).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    WO_T = 48 * 2   # wo_pair slice per STREAM core: K = 1536 / 32 = 48 tiles x 2 output tiles, BFP8 (8.4 MB per chip)
    # MKA_EXT=1 (spec v0.5 STEP 1 stand-in): the projection bytes not covered by the 80-core MIXN mixer (qkv remainder + wo)
    # are read by all 120 cores (STREAM + STREAM_EXT), each core's share = 80/120 of a STREAM core's v0.4 share
    EXT = os.environ.get('MKA_EXT', '0') == '1'
    EXTW = float(os.environ.get('MKA_EXTW', '1'))   # STREAM_EXT share / STREAM share of the projection bytes
    _den = NC + (len(ACT) - NC) * EXTW
    WO_T120 = int(NC * WO_T / _den) if EXT else WO_T
    WO_TX = int(NC * WO_T * EXTW / _den) if EXT else 0
    wo_dummy = ttnn.from_torch(torch.zeros(1, 1, (NC * WO_T + 1024) * TILE, TILE).bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    QROOT = 0; QRP = fphys[QROOT]
    print(f"[mka2] full grid {GXF}x{GYF} ({NCF} cores, {len(stream_idx)} STREAM) groups {[len(m) for m in members]} ctx {CTX} KV {KVDT} "
          f"({NU*NBK*16*KVTB/1e6:.1f} MB/layer) NBB {NBB} budget {BUDGET} S_WO {S_WO} qkv-extra {EXTRA_T} tiles/core", flush=True)
    # KV share per core: STREAM cores carry the weight stream (~47 MB/layer on their writers + gate|up on their readers),
    # so ATTN-only cores take MKA_WNS x the blocks of a STREAM core (every core >= 1 block)
    WNS = float(os.environ.get("MKA_WNS", "1"))
    GUPRE = ('MKA_GUPRE', '1') in MKDEF
    # per-core kv_mode (0 split K/NOC1 V/NOC0, 1 swapped, 2 K+V on NOC1 (reader), 3 K+V on NOC0 (writer)).
    # MKA_SWAP: 'none' | 'x<thr>' / 'y<thr>' (mode 1 at phys x/y >= thr) | 'stream' | 'attn' (mode 1 on that role)
    # MKA_KVHALF='<lo>,<hi>[,<thr>]': mode <lo> for phys x < thr (default 9), mode <hi> otherwise, e.g. '3,2' or '2,3'
    _sw = os.environ.get('MKA_SWAP', 'none'); _kh = os.environ.get('MKA_KVHALF', '')
    def SWAP(fi):
        if _kh:
            p_ = [int(v) for v in _kh.split(',')]; thr = p_[2] if len(p_) > 2 else 9
            return p_[0] if fphys[fi].x < thr else p_[1]
        if _sw.startswith('x'): return 1 if fphys[fi].x >= int(_sw[1:]) else 0
        if _sw.startswith('y'): return 1 if fphys[fi].y >= int(_sw[1:]) else 0
        if _sw == 'stream': return 1 if fi in stream_idx else 0
        if _sw == 'attn': return 0 if fi in stream_idx else 1
        return 0
    EVEN_T = sum(min(GRP, KT // KB - b0) * KB * 2 for b0 in range(0, KT // KB, 2 * GRP))   # even gate|up half per GU core
    RNG = {}
    for u in range(NU):
        m_ = members[u]
        if os.environ.get("MKA_WMAP") == "1":
            # weight ∝ 1 / measured per-core KV read time (d27u08, 32K, layer 2 of the NOC-balanced build): phys x columns,
            # STREAM rows y<=8 vs ATTN-only rows y>=9 (the read rate falls with x on NOC1)
            RT_S = {1: 125, 2: 135, 3: 158, 4: 203, 5: 232, 6: 252, 7: 285, 10: 285, 12: 305, 13: 305, 14: 370, 15: 370}
            RT_A = {1: 280, 2: 357, 3: 386, 4: 424, 5: 460, 6: 482, 7: 463, 10: 496, 12: 545, 13: 546, 14: 549, 15: 549}
            w_ = [1.0 / ((RT_S if fphys[i].y <= 8 else RT_A).get(fphys[i].x, 300)) ** float(os.environ.get("MKA_WEXP", "1")) for i in m_]
        else:
            w_ = [1.0 if i in stream_idx else WNS for i in m_]
        cum = [0.0]
        for x_ in w_: cum.append(cum[-1] + x_)
        ed = [round(NBK * c_ / cum[-1]) for c_ in cum]
        for k_, i in enumerate(m_):
            RNG[i] = (ed[k_], max(ed[k_ + 1], ed[k_] + 1))
        RNG[m_[-1]] = (RNG[m_[-1]][0], NBK)
    print(f"[mka2] KV share weight non-STREAM x{WNS}: blocks STREAM {RNG[0][1]-RNG[0][0]} / ATTN-only {RNG[max(i for i in ACT if i not in stream_idx)][1]-RNG[max(i for i in ACT if i not in stream_idx)][0]}", flush=True)

    # ================= MKA_RQKV: real qkv PROJ on the full grid (spec v0.4 deferred norm, v0.5 STREAM_EXT) =================
    RQKV = ('MKA_RQKV', '1') in MKDEF
    QN, QH, NOWN, EPS = 112, KT // 2, 2, 1e-6   # qkv output tiles (q 0-47, k 48-55, v 56-63, gate 64-111), K-half, AR owners
    QROLE = {}   # fi -> (role, first tile, n tiles, k0, nk, peer fi)
    QDONE_N = len(stream_idx)
    if RQKV:
        import struct
        f2u = lambda v: struct.unpack('<I', struct.pack('<f', v))[0]
        BF8TB = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat8_b)
        MKDEF = MKDEF + [("MKA_MWTB", str(BF8TB)), ("MKA_NOWN", str(NOWN)), ("MKA_INVH_BITS", f"{f2u(1.0 / H)}u"), ("MKA_EPS_BITS", f"{f2u(EPS)}u")]
        assert not (GDN24 and ('MKA_QSK', '1') not in MKDEF), "MKA_GDN24 with the real qkv needs MKA_QSK (16 ext cores cannot hold the ext qkv tiles); use the MIXN stand-in"
        ext_list = [fi for fi in range(NCF) if fi not in stream_idx and fi not in RESV]
        s_by_i = {i_: fi for fi, i_ in stream_idx.items()}
        n_ext_tiles = 2 * len(ext_list); n_half = QN - n_ext_tiles
        assert QSK or (n_half >= 0 and 2 * n_half <= len(stream_idx)), (n_ext_tiles, n_half)   # K-half split only without QSK
        for e, fi in enumerate(ext_list): QROLE[fi] = (3, 2 * e, 2, 0, KT, fi)
        for i_ in range(0 if QSK else 2 * n_half):
            fi = s_by_i[i_]; h_ = i_ % 2
            QROLE[fi] = (1 if h_ == 0 else 2, n_ext_tiles + i_ // 2, 1, h_ * QH, QH, s_by_i[i_ + 1] if h_ == 0 else s_by_i[i_ - 1])
        QDONE_N = sum(1 for v in QROLE.values() if v[0] in (1, 3))
        if QSK:
            import sys as _sys
            _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
            import layout_attn as LA_
            QSL_ = LA_.qkv_splitk_slices()
            for fi in ext_list: QROLE[fi] = (3, 0, 0, 0, 0, fi)   # ext ring user (Wo); no qkv on the ext cores
            QSKEY = {fi: (fcores[fi].x, fcores[fi].y) for fi in stream_idx}
            for fi, (c_, k_) in QSKEY.items():
                sl_ = QSL_[(c_, k_)]
                QROLE[fi] = (1, sl_.first + k_, 1 + (1 if k_ + 8 < sl_.nt else 0), sl_.nt, QSK_GK, fi)   # owner of items k, k + 8
            QDONE_N = len(stream_idx)
        QSLOT = 2 * KT
        # bank-local slices: each qkv core's slice in its nearest bank (fcores logical coords vs the bank's preferred worker)
        qcores = sorted(f_ for f_ in QROLE if not (QSK and QROLE[f_][0] == 3))
        # byte-balanced: each slice class (ext 348 KB, STREAM half 87 KB) is spread evenly over the banks, nearest bank
        # first within the class (a count-only cap put most ext slices on the banks next to the ext cores -> hot banks)
        q_bank = {}
        for cls in ([] if QSK else ([fi for fi in qcores if QROLE[fi][0] == 3], [fi for fi in qcores if QROLE[fi][0] != 3])):
            caps = [len(cls) // banks + (1 if b_ < len(cls) % banks else 0) for b_ in range(banks)]
            pairs = sorted(((abs(fcores[fi].x - bx) + abs(fcores[fi].y - by), fi, b_) for fi in cls for b_, (bx, by) in enumerate(BANK_LOC)))
            for d_, fi, b_ in pairs:
                if fi in q_bank or caps[b_] == 0: continue
                q_bank[fi] = b_; caps[b_] -= 1
        q_slot = {}   # bank-local page offset of each core's slice (slices packed per bank)
        _tot = [0] * banks
        if QSK:
            _qb, _qo, PQL = LA_.splitk_bank_assign(QSL_, {key: key for key in QSL_}, BANK_LOC, banks)
            for fi, key in QSKEY.items(): q_bank[fi], q_slot[fi] = _qb[key], _qo[key]
        else:
            for fi in qcores:
                b_ = q_bank[fi]; q_slot[fi] = _tot[b_]; _tot[b_] += QROLE[fi][2] * QROLE[fi][4]
            PQL = max(_tot) * banks
        torch.manual_seed(7)
        wq_t = torch.randn(H, QN * TILE) / math.sqrt(H)
        gam = 0.5 + torch.rand(H)
        wf_t = gam[:, None] * wq_t   # diag(γ) W, folded at load (spec v0.4; × NORM_FIX is part of γ here)
        def q_pages(scale):
            if QSK: return LA_.pack_qkv_splitk_layer(wf_t, QSL_, _qb, _qo, banks, PQL, scale)
            pg = torch.zeros(PQL, TILE, TILE)
            for fi in qcores:
                ro, n0, nt, k0, nk, _ = QROLE[fi]; b_, j_ = q_bank[fi], q_slot[fi]
                for kl in range(nk):
                    for jj in range(nt):
                        pg[(j_ + kl * nt + jj) * banks + b_] = tl(wf_t, k0 + kl, n0 + jj) * scale
            return pg
        wq_all = ttnn.from_torch(torch.cat([q_pages(2.0 ** (L % 2)) for L in range(NL)]).reshape(1, 1, -1, TILE).bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        wqq = ttnn.to_torch(wq_all).float().reshape(NL, PQL, TILE, TILE)
        wfq = torch.zeros(NL, H, QN * TILE)   # dequantized folded weights as the device sees them, per layer
        if QSK:
            for key, sl_ in QSL_.items():
                for kl in range(sl_.pages // sl_.nt):
                    for jj in range(sl_.nt):
                        k_, n_ = sl_.k0 + kl, sl_.first + jj
                        wfq[:, k_ * TILE:(k_ + 1) * TILE, n_ * TILE:(n_ + 1) * TILE] = wqq[:, (_qo[key] + sl_.page_of(kl, jj)) * banks + _qb[key]]
        for fi in ([] if QSK else qcores):
            ro, n0, nt, k0, nk, _ = QROLE[fi]; b_, j_ = q_bank[fi], q_slot[fi]
            for kl in range(nk):
                for jj in range(nt):
                    wfq[:, (k0 + kl) * TILE:(k0 + kl + 1) * TILE, (n0 + jj) * TILE:(n0 + jj + 1) * TILE] = wqq[:, (j_ + kl * nt + jj) * banks + b_]
        ss = torch.stack([(xb[:, o * (H // NOWN):(o + 1) * (H // NOWN)].double() ** 2).sum(-1) for o in range(NOWN)])   # [NOWN, M]
        rsp_t = torch.zeros(TILE, NOWN * TILE)
        for o in range(NOWN): rsp_t[:, o * TILE] = ss[o].float()
        rsp = ttnn.from_torch(rsp_t.repeat(len(ACT), 1).reshape(1, 1, len(ACT) * TILE, NOWN * TILE), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=hs(kgrid, NOWN * TILE))
        rstd_g = 1.0 / torch.sqrt(ss.sum(0) / H + EPS)   # [M]
        gold_qkv = [(rstd_g[:, None] * (xb.double() @ wfq[L].double())).float() for L in range(NL)]
        qkv_out = ttnn.from_torch(torch.zeros(1, 1, NL * TILE, QN * TILE).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # x on every qkv core: shard x over the full grid (lockstep L1 address, same footprint as the 80-core shard)
        if not QSK:
            if xbuf is not None: ttnn.deallocate(xbuf)   # free the 80-core shard first so the full-grid one reuses its lockstep L1 range
            xbuf = ttnn.from_torch(xb.reshape(1, 1, M, H).bfloat16().float().repeat(1, 1, len(ACT), 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=hs(kgrid, H))
        elif xbuf is None:   # QSK: CB 0 is a 1-tile placeholder (hw_startup operand)
            xbuf = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TILE * NC, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, hs(rect, TILE))
        EXTRA_T = EXTRA_TX = 0   # the qkv bytes are real now
        ext_crs = crs([fcores[fi] for fi in ext_list])
        if QSK: print(f"[qsk] split-K qkv: 80 STREAM owners, slot M {max(s_.pages for s_ in QSL_.values())} pages, PQL {PQL}, GDN stand-in {QSK_GK} K tiles", flush=True)
        else: print(f"[rqkv] {QN} tiles: ext {len(ext_list)}x2 full-K, STREAM {2 * n_half} K-halves; producers {QDONE_N}; MB/bank {[round(sum(QROLE[f][2] * QROLE[f][4] for f in qcores if q_bank[f] == b_) * BF8TB / 1e6, 2) for b_ in range(banks)]}; {QN * KT * BF8TB / 1e6:.1f} MB/layer", flush=True)
    if not RQKV and ('MK_V12', '1') in MKDEF:   # V12 on the MIXN mixer stand-in (GDN24 before QSK builds without RWO): own CB 43 rstd stand-in
        import struct
        f2u = lambda v: struct.unpack('<I', struct.pack('<f', v))[0]
        MKDEF = MKDEF + [("MKA_NOWN", str(NOWN)), ("MKA_INVH_BITS", f"{f2u(1.0 / H)}u"), ("MKA_EPS_BITS", f"{f2u(EPS)}u")]
        ss = torch.stack([(xb[:, o * (H // NOWN):(o + 1) * (H // NOWN)].double() ** 2).sum(-1) for o in range(NOWN)])   # [NOWN, M]
        rsp_t = torch.zeros(TILE, NOWN * TILE)
        for o in range(NOWN): rsp_t[:, o * TILE] = ss[o].float()
        rsp = ttnn.from_torch(rsp_t.repeat(len(ACT), 1).reshape(1, 1, len(ACT) * TILE, NOWN * TILE), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=hs(kgrid, NOWN * TILE))
        rstd_g = 1.0 / torch.sqrt(ss.sum(0) / H + EPS)   # [M]
        print('[v12] on the MIXN qkv stand-in (no MKA_RQKV): CB 43 rstd stand-in on its own', flush=True)
        if xbuf is not None and not MESH:   # compact L1 like the RQKV path: the re-made x shard lands in the placeholder's hole,
            ttnn.deallocate(xbuf)              # so the v12 L1 tensors (v_pbuf 200 KB) fit above the STREAM CB region
            xbuf = ttnn.from_torch(xb.reshape(1, 1, M, H).bfloat16().float().repeat(1, 1, NC, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=hs(rect, H))
    GDNIN = int(os.environ.get('MKA_GDNIN_CORES', '64'))   # GDN in-proj stand-in on the first 64 K-half cores (5.6 MB, spec 5.9),
    def gdn_on(fi):                                         # independent of how many K-half pairs the qkv split uses
        if QSK: return 1 if fi in stream_idx else 0
        return 1 if (fi in stream_idx and stream_idx[fi] < GDNIN and QROLE.get(fi, (0,))[0] in (1, 2)) else 0
    # ================= MK_V12: fm12 split-K MLP phase (spec v0.6-v0.9) replacing the fm11 MLP =================
    V12 = ('MK_V12', '1') in MKDEF
    V12CB = V12CB_PB = False
    if V12:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
        import layout_mlp as LM
        if not any(k_ == 'MK_OWNER_RESID' for k_, _ in MKDEF): MKDEF = MKDEF + [('MK_OWNER_RESID', '1')]
        # LoFi on the BFP4 MLP matmuls is bit-identical to HiFi2 in the engine (owner xo torch.equal, d30u20) -> default on
        if not any(k_ == 'MK_V12_LOFI' for k_, _ in MKDEF): MKDEF = MKDEF + [('MK_V12_LOFI', '1')]
        V_RESID = ('MK_OWNER_RESID', '1') in MKDEF
        V_R = int(os.environ.get('V12_R', '32')); V_FIRST = int(os.environ.get('V12_FIRST', '1'))
        assert V_R == 32 or 1 <= V_R <= 16
        # spec v1.71 (RC rule): MK_REAL_AR at R < 32 has an open residual blow-up (u1, R=4). The RC engine is the R = 32 (8-user)
        # program only (1-7 users padded onto it). Never inside the model (QWEN36_R16_MEGAKERNEL=1); bench debugging needs the
        # explicit opt-in MK_ALLOW_REAL_AR_R_LT32=1.
        if V_R < 32 and ('MK_REAL_AR', '1') in MKDEF:
            assert os.environ.get('QWEN36_R16_MEGAKERNEL', '0') != '1', \
                f"engine with MK_REAL_AR at R={V_R} < 32 is forbidden in the model (spec v1.71: open R<32 real-AR bug)"
            assert os.environ.get('MK_ALLOW_REAL_AR_R_LT32', '0') == '1', \
                f"MK_REAL_AR at R={V_R} < 32 is numerically wrong (spec v1.71); set MK_ALLOW_REAL_AR_R_LT32=1 to debug it"
        if os.environ.get('MK_V12_RCT', '1') == '1': MKDEF = MKDEF + [('MK_V12_RCT', str(V_R))]   # compile-time R (text budget)
        def q4(w): return ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
        vwg, vwu, vwd = q4(wg_t), q4(wu_t), q4(wd_t)   # BFP4 values are tile-local, so layout independent
        # MK_REAL_AR at R = 32: the MLP input is the evolving residual (x + ring x MLP per layer, with the CB 20 rstd stand-in
        # instead of a real norm), which overflows bf16 within a few layers; 2^-6 on the down weights keeps it bounded
        _vds = 2.0 ** -6 if (('MK_REAL_AR', '1') in MKDEF and V_R == 32) else 1.0
        def vlayer(L): return vwg, vwu * (2.0 ** (L % 2)), vwd * (2.0 ** -(L % 3)) * _vds   # distinct per layer, exact in BFP4
        s_xy = [(i % GX, i // GX) for i in range(NC)]
        v_bank, v_rank, v_nsb = LM.assign_banks(s_xy, BANK_LOC)
        PVL = LM.pages_per_layer(v_nsb, banks)
        t_v = time.time()
        wv_all = ttnn.from_torch(torch.cat([LM.pack_layer(*vlayer(L), v_bank, v_rank, v_nsb, banks) for L in range(NL)]).reshape(1, 1, -1, TILE),
                                 dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # gold of the last layer's owner output (x static in the engine): Σ down partials (+ x while the AR is a stall)
        _g, _u, _d = vlayer(NL - 1)
        _xn = (rstd_g[:, None] * xb.double()).float()
        _h = (torch.nn.functional.silu(_xn @ _g) * (_xn @ _u)).bfloat16().float()
        v_gold = _h @ _d + (xb if V_RESID else 0)
        O_CRS = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(GX, 0), ttnn.CoreCoord(GX + 1, GY - 1))])
        def v_wsh(crs, tiles_per): return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs, [TILE, tiles_per * TILE], ttnn.ShardOrientation.ROW_MAJOR))
        def v_zero(crs, ncores, tiles_per, dt=ttnn.bfloat16):
            return ttnn.from_torch(torch.zeros(TILE, ncores * tiles_per * TILE), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=v_wsh(crs, tiles_per))
        # the fm11 h landing buffer (136 KB, unused by v12) is freed FIRST so the v12 landing buffers reuse its range (the L1
        # allocator is lockstep across cores, so the owners' pbuf also costs STREAM-core address space)
        ttnn.deallocate(hbuf)
        # MK_V12_CB=1 (spec §20, v12cb): the program-internal v12 landing buffers are CBs on their own role's cores (no persistent,
        # lockstep L1 on all 120 cores): v_xk CB 33 / v_rs CB 36 / v_h7 CB 37 (STREAM), v_xpk = CB 16 (STREAM + owners, the owners'
        # packing scratch), v_pbuf CB 3 (owners). Peers address them remotely, so their addresses come from the host replica of
        # tt-metal's CB allocator (engine/cbplan.py) over the final CB list (VADDR, filled by _v12_plan below); MK_V12_CBCHK=1
        # checks every (core, CB) address on device. Stay tensors: v_xo / v_x0 (AR inp / residual = engine x I/O), v_id / v_j (2 KB
        # constants), and v_pbuf under MK_AR_GAMMA without MK_AR_GEARLY (its γ CB 32 aliases pbuf + 0).
        V12CB = ('MK_V12_CB', '1') in MKDEF
        V12CB_PB = V12CB and not (('MK_REAL_AR', '1') in MKDEF and ('MK_AR_GAMMA', '1') in MKDEF and ('MK_AR_GEARLY', '1') not in MKDEF)
        VADDR = {}
        if V12CB:
            v_xk = v_rs = v_h7 = v_xpk = None
        else:
            v_xk = v_zero(rect, NC, 20); v_rs = v_zero(rect, NC, 16); v_h7 = v_zero(rect, NC, 7); v_xpk = v_zero(rect, NC, 10 if V_R < 32 else 1)
        v_xo = v_zero(O_CRS, 16, 10)
        v_x0 = ttnn.from_torch(xb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=v_wsh(O_CRS, 10))   # owner j: x tiles 10j..
        v_pbuf = None if V12CB_PB else v_zero(O_CRS, 16, 100)
        def VA(n_):   # v12 landing-buffer address: the tensor's, or the planned CB address (MK_V12_CB)
            t_ = {'xk': v_xk, 'rs': v_rs, 'h7': v_h7, 'xpk': v_xpk, 'pbuf': v_pbuf}[n_]
            return t_.buffer_address() if t_ is not None else VADDR[n_]
        V_ALL = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1)), ttnn.CoreRange(ttnn.CoreCoord(GX, 0), ttnn.CoreCoord(GX + 1, GY - 1))])
        v_id = ttnn.from_torch(torch.eye(TILE).repeat(1, NC + 16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=v_wsh(V_ALL, 1))
        _J = torch.zeros(TILE, TILE); _J[0, :] = 1.0
        v_j = ttnn.from_torch(_J.repeat(1, NC), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=v_wsh(rect, 1))
        hbuf = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TILE * NC, TILE]), HDT, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG if V12CB else hs(rect, TILE))   # placeholder (unused by v12)
        VGS = [ttnn.create_global_semaphore(dev, kgrid, 0) for _ in range(10)]   # never on reserved (AR / GDN) cores
        VGA = [ttnn.get_global_semaphore_address(g) for g in VGS]
        def v_prect(lx0, lx1, ly0, ly1):
            a_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx0, ly0)); b_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx1, ly1))
            return [a_.x, a_.y, b_.x, b_.y, (lx1 - lx0 + 1) * (ly1 - ly0 + 1)]
        def v_ph(x_, y_): return dev.worker_core_from_logical_core(ttnn.CoreCoord(x_, y_))
        # busy mirror to the rows 8-9 ext PQ readers (x 0-9, never the AR corner); none under GDN24 (rows 8-9 reserved)
        V_BRC = [0] * 10 if GDN24 else v_prect(0, split - 1, GY, GYF - 1) + v_prect(split, GX - 1, GY, GYF - 1)
        print(f"[v12] R {V_R} FIRST {V_FIRST} RESID {int(V_RESID)} pages/layer {PVL} ({NC * LM.SLOT * WTB / 1e6:.1f} MB/layer, built {time.time() - t_v:.1f}s) vgs {VGA}", flush=True)
        def v_args(fi, c):
            """(writer, reader, compute) v12 arg tails for full-grid core fi"""
            z_own = [0] * 3 + [0] * 10 + [0] * 10 + [0, 0] + [0] * 10
            if fi in stream_idx:
                n, k = c.x, c.y; i_ = stream_idx[fi]
                fp, nt, nn = LM.core_fields(i_, v_bank, v_rank, banks)
                me = v_ph(n, k); cy = [v_ph(n, r_).y for r_ in range(GY)]
                hi = (2 * k + 1, 2 * k) if V_FIRST else (2 * k, 2 * k + 1)
                ob = v_ph(GX + hi[0] % 2, k); oa = v_ph(GX + hi[1] % 2, k)
                g0, g1 = v_ph(GX, k), v_ph(GX + 1, k)
                w_ = [1, k, nn, n] + v_prect(n, n, 0, GY - 1) + [me.x] + cy + [VA('h7'), VA('rs'), oa.x, oa.y, VA('pbuf'),
                      V_R, VA('xpk'), me.x, me.y, 1 if n == 0 else 0, g0.x, g0.y, g1.x, g1.y, wv_all.buffer_address(), fp, PVL, nt] + VGA + z_own
                r_ = [1, k, nn, n, me.x] + cy + [VA('rs'), VA('h7'), ob.x, ob.y, VA('pbuf'), V_R] + VGA + [0] * 15
                return w_, r_, [1, nn, k, V_FIRST]
            if GX <= c.x <= GX + 1 and c.y < GY:   # owner j = 2k + slot
                k, slot = c.y, c.x - GX; me = v_ph(c.x, c.y)
                w_ = [2, k, 0, 0] + [0] * 5 + [0] + [0] * 8 + [0, 0, 0, 0, VA('pbuf'), V_R, 0, me.x, me.y, 0, 0, 0, 0, 0, 0, 0, 0, 0] + VGA + \
                     [slot, VA('xk') + slot * 10 * 2048, VA('xpk') + slot * 10240] + v_prect(0, split - 1, k, k) + v_prect(split, GX - 1, k, k) + \
                     [v_ph(n_, k).x for n_ in range(GX)] + [v_ph(0, k).y, 1 if (k == 0 and slot == 0) else 0] + V_BRC
                r_ = [2, k, 0, 0, 0] + [0] * 8 + [0, 0, 0, 0, 0, V_R] + VGA + [slot, VA('xk') + slot * 10 * 2048, VA('xpk') + slot * 10240] + v_prect(0, split - 1, k, k) + v_prect(split, GX - 1, k, k) + [me.x, me.y]
                return w_, r_, [2, 0, k, 0]
            w_ = [0] * 4 + [0] * 5 + [0] + [0] * 8 + [0] * 18 + VGA + z_own
            r_ = [0] * 5 + [0] * 8 + [0] * 6 + VGA + [0] * 15
            return w_, r_, [0, 0, 0, 0]

    # MK_REAL_AR=1 (AR integration v1, spec v0.9.2 / §18 option a): the real MLP all-reduce on the 1x8 ring replaces the MLP
    # AR stall. Owners (10..11, 0..7) run AR_RECV (mkar::Recv) on the writer after the v12 release and mkar::Reduce on the
    # compute after C-OSUM; the 4 AR-only cores run AR_SEND. x of layer L + 1 lands in the K-rows' v_xk / v_xpk slot with its
    # own flag pair (MK_AR_XF0 / XF1: the stand-in v12_ox keeps feeding the MLP after the attention / GDN AR stall); each
    # STREAM core waits its row's two flags instead of the stall, the Q root waits all 16 deliveries (MK_AR_XCNT) before XRDY.
    REAL_AR = ('MK_REAL_AR', '1') in MKDEF
    if REAL_AR:
        assert MESH and V12, "MK_REAL_AR needs MK_MESH=1 and MK_V12=1"
        assert ('MK_OWNER_RESID', '0') in MKDEF, "MK_REAL_AR: the AR adds the residual (MK_OWNER_RESID=0)"
        from models.demos.blackhole.qwen36.tt.megakernel.ar_phase import ArPhase
        AR_XF = [ttnn.create_global_semaphore(dev, fgrid, 0) for _ in range(2)]
        AR_XCNT = ttnn.create_global_semaphore(dev, kgrid, 0)
        # MK_AR_DLVR=1 (default, spec v1.21): x delivery on the owner READER (NOC1) via a local handoff word
        AR_DLVR = os.environ.get('MK_AR_DLVR', '1') == '1'
        AR_DLV = ttnn.create_global_semaphore(dev, O_CRS, 0) if AR_DLVR else None
        # MK_REAL_AAR=1 (spec §19): the real attention AR after each attention layer's wo (input: the wo unit's output CB
        # 23 on the owners, which also spans the 4 sender cores so its local address is the owners'); it delivers the
        # MLP's x (X2 flag 2L + 2) with γ_post_attn. ARs per pass: NL + attention layers, in program order.
        REAL_AAR = ('MK_REAL_AAR', '1') in MKDEF
        AR_NAR, AR_ATYPE = NL, 0
        if REAL_AAR:
            assert ('MKA_RWO', '1') in MKDEF and AR_DLVR, "MK_REAL_AAR needs MKA_RWO (the wo unit's CB_XO) and the reader-side delivery"
            _k = 0
            for L_ in range(NL):
                if is_attn_layer(L_): AR_ATYPE |= 1 << _k; _k += 1
                _k += 1
            AR_NAR = _k
            AR_ADLV = ttnn.create_global_semaphore(dev, O_CRS, 0); AR_ASRC = ttnn.create_global_semaphore(dev, O_CRS, 0)
            MKDEF = MKDEF + [('MK_AR_ADLV', str(ttnn.get_global_semaphore_address(AR_ADLV))), ('MK_AR_ASRC', str(ttnn.get_global_semaphore_address(AR_ASRC))),
                             ('MK_AR_CB_AIN', '23')]
            print(f"[ar] attention AR on: {AR_NAR} ARs / pass, attention mask {AR_ATYPE:#x}", flush=True)
        # MK_AR_DBG=1 (validation, R = 32): per-AR input / residual chunks -> DRAM (see mka2_writer.cpp ar_dbg)
        AR_DBG = os.environ.get('MK_AR_DBG', '0') == '1' and V_R == 32
        if AR_DBG:
            ar_dbgt = ttnn.from_torch(torch.zeros(1, 1, TILE, AR_NAR * 2 * H).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            MKDEF = MKDEF + [('MK_AR_DBGADDR', str(ar_dbgt.buffer_address()))]
        if AR_DLVR: MKDEF = MKDEF + [('MK_AR_DLV', str(ttnn.get_global_semaphore_address(AR_DLV)))]
        MKDEF = MKDEF + [('MK_AR_XF0', str(ttnn.get_global_semaphore_address(AR_XF[0]))), ('MK_AR_XF1', str(ttnn.get_global_semaphore_address(AR_XF[1]))),
                         ('MK_AR_XCNT', str(ttnn.get_global_semaphore_address(AR_XCNT)))]
        # GDN24 (AR sender hosted in the GDN role on the corner, spec v1.44): the corner cores do NOT span the owners' RS slot
        # CB (CB_RED, 140-160 KB) or the wo output CB 23 (their GDN-role CB stack would be lifted above the owners' whole CB
        # region). Owners publish both addresses at init into 2 words on the 4 corner cores (MK_AR_PUB / MK_AR_PUBA); the
        # hosted SendWriter / SendReader wait for them non-zero; the reader zeroes them at fini (device-local).
        AR_PUB = None
        if GDN24:
            _corner_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(GX, GY), ttnn.CoreCoord(GX + 1, GY + 1))])
            AR_PUB = [ttnn.create_global_semaphore(dev, _corner_crs, 0) for _ in range(2)]
            _pxy = [dev.worker_core_from_logical_core(ttnn.CoreCoord(*xy_)) for xy_ in ((GX, GY), (GX + 1, GY), (GX, GY + 1), (GX + 1, GY + 1))]
            AR_PUB_D = [('MK_AR_PUB', str(ttnn.get_global_semaphore_address(AR_PUB[0]))), ('MK_AR_PUBA', str(ttnn.get_global_semaphore_address(AR_PUB[1])))]
            MKDEF = MKDEF + AR_PUB_D + [(f'MK_AR_PUBXY{i_}', f'{(c_.x << 16) | c_.y}u') for i_, c_ in enumerate(_pxy)]
        # own_cb (R = 32): no own slot in the AR's RS slot CB; the reduce reads this device's partial from CB_VXO (CB 4)
        # AR CB ids: v12 STREAM-only ids (free on the owners and the AR-only senders) so the program's max CB index stays 44
        # (each index above it costs every core 16 B of kernel config buffer); MK_AR_CBHI=1: the 50..60 block
        _arcb = None if os.environ.get('MK_AR_CBHI', '0') == '1' else [34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 30]
        # x_targets is read lazily (engine_parts); under MK_V12_CB the v_xk / v_xpk addresses come from the CB plan, after build()'s list
        AR_XT = dict(xk=None if V12CB else VA('xk'), xpk=None if V12CB else VA('xpk'), xflag=[ttnn.get_global_semaphore_address(g) for g in AR_XF], rstd=0)
        AR = ArPhase(dev, n_ar=AR_NAR, rows=0 if V_R == 32 else V_R, stats=False, own_cb=4 if os.environ.get('MK_AR_NOOWN', '1') == '1' else None, cb_ids=_arcb, deliver_ext=AR_DLVR,
                     x_targets=AR_XT)
        AR_PT = AR.pt
        _ow = AR.owners_grid()
        def _arw(tiles): return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(_ow, [TILE, tiles * TILE], ttnn.ShardOrientation.ROW_MAJOR))
        # the RS slot region (ring x PT tiles per owner, 160 KB at R = 32) is a plain program CB on owners + senders, not an L1
        # tensor: the L1 allocator is lockstep over every core, and 160 KB below the tensors would overrun the STREAM cores'
        # CB region (the senders address their own copy via get_write_ptr; v_xo is only the descriptor's placeholder)
        ar_buf = v_xo
        # residual stream (AR output = next AR's residual, in place). R = 32: it IS v_x0 (owner j: x tiles 10j.., unused by the
        # owner sum with MK_OWNER_RESID=0), so the stand-in x delivery feeds each MLP the real residual stream and no extra
        # lockstep L1 is spent (the owners' CB region is ~50 KB short otherwise); R < 32: a packed residual of its own
        ar_out = v_x0 if AR_PT == 10 else ttnn.from_torch(torch.zeros(TILE, 16 * AR_PT * TILE), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=_arw(AR_PT))
        if AR_DLVR and AR_PT != 10: MKDEF = MKDEF + [('MK_AR_OUT', str(ar_out.buffer_address()))]   # reader's packed source
        # MK_AR_GAMMA=1 (spec §19): x_deliver = residual ⊙ γ_next on every owner; γ table in DRAM [NL x 2 (input_ln,
        # post_attn_ln) x 16 chunks x 10 tiles] bf16, γ in row 0, page (L*2+n)*160 + chunk*10 + t (√1.25 host-folded)
        AR_GAMMA = ('MK_AR_GAMMA', '1') in MKDEF
        if AR_GAMMA:
            assert AR_DLVR, "MK_AR_GAMMA delivers from the owner reader (MK_AR_DLVR=1)"
            _gg = torch.Generator().manual_seed(19)
            AR_GTAB = (0.5 + torch.rand(NL, 2, H, generator=_gg)) * (1.25 ** 0.5)
            AR_GTAB = AR_GTAB.bfloat16().float()   # as the device sees it
            AR_GT = 10 if AR_PT == 10 else AR_PT
            AR_GEARLY = ('MK_AR_GEARLY', '1') in MKDEF
            if AR_PT == 10:
                _gt = torch.zeros(TILE, NL * 2 * H); _gt[0] = AR_GTAB.reshape(-1)
                ar_gam = ttnn.from_torch(_gt.reshape(1, 1, TILE, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:   # R < 32: packed, row-replicated: page (L*2+n)*16 + chunk = AR_GT pseudo-tiles, element ((2t+f)R + r)16 + c
                _R = V_R; _pk = torch.zeros(NL, 2, 16, AR_GT * 1024)
                _g = AR_GTAB.reshape(NL, 2, 16, 10, 2, 16)                     # [L, n, chunk, t, f, c]
                _rep = _g[:, :, :, :, :, None, :].expand(NL, 2, 16, 10, 2, _R, 16)   # [.., t, f, r, c]
                _pk[..., :10 * 2 * _R * 16] = _rep.reshape(NL, 2, 16, -1)
                ar_gam = ttnn.from_torch(_pk.reshape(1, 1, NL * 2 * 16, AR_GT * 1024), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            MKDEF = MKDEF + [('MK_AR_CBG', '32'), ('MK_AR_GT', str(AR_GT)),
                             ('MK_AR_GROWS', '0' if AR_PT == 10 else str(V_R))]
            print(f"[ar] gamma: table {NL}x2x16x10 tiles at DRAM {ar_gam.buffer_address()}, CB 32 = pbuf+0 ({AR_GT} tiles)", flush=True)
        AR_SEQ = AR_PT == 10 and not REAL_AAR   # (attention ARs: no host gold of the chain; MK_AR_DBG checks each AR) the MLP input is the evolving residual: sequential golds (x_{L+1} = x_L + ring * mlp_L(x_L))
        _xres = xb.clone()
        _snd = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c_), ttnn.CoreCoord(*c_)) for c_ in ((10, 8), (11, 8), (10, 9), (11, 9))])
        ar_sem = ttnn.create_global_semaphore(dev, _ow.merge(_snd), 0)
        # per-device AR parts up front (the owners' L1 arg table needs every device's args); build() reuses them
        def _ar_parts():   # (MK_V12_CB: called after the CB plan filled AR_XT)
            ARP_ = {c_: AR.engine_parts(ttnn.MeshCoordinate(0, c_), v_xo, ar_buf, ar_out, ar_out, ar_sem) for c_ in range(AR.ring)}
            blk_ = AR.arg_table(ARP_, gamma_addr=ar_gam.buffer_address() if AR_GAMMA else 0)
            print(f"[ar] arg table at L1 {blk_.buffer_address()} ({AR.BLK_NW} words / owner)", flush=True)
            print(f"[ar] real MLP AR: ring {AR.ring} rows {AR.rows} PT {AR_PT} buf {ar_buf.buffer_address()} out {ar_out.buffer_address()} xf {[ttnn.get_global_semaphore_address(g) for g in AR_XF]} xcnt {ttnn.get_global_semaphore_address(AR_XCNT)} xk {AR_XT['xk']} xpk {AR_XT['xpk']}", flush=True)
            return ARP_, blk_, [('MK_AR_BLK', str(blk_.buffer_address())), ('MK_AR_BLK_COMP', str(AR.BLK_COMP)), ('MK_AR_BLK_RECV', str(AR.BLK_RECV)),
                                ('MK_AR_BLK_GAMMA', str(AR.BLK_GAMMA))]
        if not V12CB:
            ARP, ar_blk, _ardefs = _ar_parts()
            MKDEF = MKDEF + _ardefs

    def q_wargs(fi):
        if not RQKV: return []
        ro, n0, nt, k0, nk, peer = QROLE.get(fi, (0, 0, 0, 0, 0, fi))
        pp_ = fphys[peer]
        return [ro, n0, nt, pp_.x, pp_.y, qkv_out.buffer_address(), gdn_on(fi)] + wo_wargs(fi) + head_args_w(fi)
    TREEFLAG = ('MKA_TREEFLAG', '1') in MKDEF
    def tfk(fi):   # index of this core in its parent's child list (level-0 list or the root's level-1 list)
        r_ = rank[fi]
        return (r_ % FAN - 1) if r_ % FAN else (r_ // FAN - 1 if r_ else 0)
    def q_rargs(fi):
        if not RQKV: return []
        ro = QROLE.get(fi, (0,))[0]
        return [ro, wq_all.buffer_address(), (q_slot.get(fi, 0) * banks + q_bank.get(fi, 0)) if ro else 0, PQL] + wo_rargs(fi) + head_args_r(fi)
    RWO = ('MKA_RWO', '1') in MKDEF
    assert not (('MKA_WORING', '1') in MKDEF and ('MKA_TREEFLAG', '1') in MKDEF), 'CB 31: WORING xor TREEFLAG'
    # GDN24 bring-up: real wo on the 16 owners (10 tiles each, ring 480 pages); the 36-core split (owners + GDN cores) comes
    # with the GDN role's include of the wo unit (spec v1.5)
    WO_MAXT_ = int(os.environ.get('MKA_WO_MAXT', '10' if GDN24 else '5'))
    RING_P_ = max(WO_MAXT_ * 48, 2 * KT)
    if RWO: MKDEF = MKDEF + [('MKA_WO_MAXT', str(WO_MAXT_))]
    # MK_REAL_AR with the 50..60 CB ids (MK_AR_CBHI=1): they grow every core's CB-config block (16 B per index up to the
    # max); the u8 STREAM program was then 192 B past 70656 B with the compute zones in (default: compact ids, no LEAN_C)
    if (RWO or (('MK_REAL_AR', '1') in MKDEF and os.environ.get('MK_AR_CBHI', '0') == '1')) and ('MKA_LEAN_C', '1') not in MKDEF: MKDEF = MKDEF + [('MKA_LEAN_C', '1')]
    WOT, WOP, WOB = {}, {}, {}
    if RWO:
        assert RQKV and ('MKA_PQGDN', '1') in MKDEF, "MKA_RWO needs MKA_RQKV and MKA_PQGDN (the ext ring holds Wo after qkv)"
        assert not any(is_attn_layer(L_) and is_attn_layer(L_ + 1) for L_ in range(NL - 1)), "MKA_RWO: no back-to-back attention layers"
        WK_, WN_ = 48, 160
        wo_cores = list(ext_list)
        # rot chunk order for device 0: owning device o = j // 2, farthest around the 8-ring first, own (o = 0) last
        rot = sorted(range(16), key=lambda j_: (-min(j_ // 2, 8 - j_ // 2), j_))
        flat = [n_ for j_ in rot for n_ in range(10 * j_, 10 * j_ + 10)]
        for c_, fi in enumerate(wo_cores): WOT[fi] = flat[c_::len(wo_cores)]
        assert max(len(v) for v in WOT.values()) <= WO_MAXT_, (max(len(v) for v in WOT.values()), WO_MAXT_)
        torch.manual_seed(11)
        wo_t = torch.randn(WK_ * TILE, WN_ * TILE) / math.sqrt(WK_ * TILE)
        # byte-balanced bank-local slices, k-major pages (k * nt + j)
        caps_ = [len(wo_cores) // banks + (1 if b_ < len(wo_cores) % banks else 0) for b_ in range(banks)]
        pairs_ = sorted(((abs(fcores[fi].x - bx) + abs(fcores[fi].y - by), fi, b_) for fi in wo_cores for b_, (bx, by) in enumerate(BANK_LOC)))
        for d_, fi, b_ in pairs_:
            if fi in WOB or caps_[b_] == 0: continue
            WOB[fi] = b_; caps_[b_] -= 1
        tot_ = [0] * banks
        for fi in wo_cores: WOP[fi] = tot_[WOB[fi]]; tot_[WOB[fi]] += len(WOT[fi]) * WK_
        WOPL = max(tot_) * banks
        def wo_pages(scale):
            pg = torch.zeros(WOPL, TILE, TILE)
            for fi in wo_cores:
                for kl in range(WK_):
                    for jj, n_ in enumerate(WOT[fi]):
                        pg[(WOP[fi] + kl * len(WOT[fi]) + jj) * banks + WOB[fi]] = tl(wo_t, kl, n_) * scale
            return pg
        wo_all = ttnn.from_torch(torch.cat([wo_pages(2.0 ** -(L % 2)) for L in range(NL)]).reshape(1, 1, -1, TILE).bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        woq_ = ttnn.to_torch(wo_all).float().reshape(NL, WOPL, TILE, TILE)
        woq = torch.zeros(NL, WK_ * TILE, WN_ * TILE)
        for fi in wo_cores:
            for kl in range(WK_):
                for jj, n_ in enumerate(WOT[fi]):
                    woq[:, kl * TILE:(kl + 1) * TILE, n_ * TILE:(n_ + 1) * TILE] = woq_[:, (WOP[fi] + kl * len(WOT[fi]) + jj) * banks + WOB[fi]]
        wod = ttnn.from_torch(torch.zeros(1, 1, NL * TILE, WN_ * TILE).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        WCC = wo_cores[0]   # concat core
        owner_fi = [fcores.index(next(c_ for c_ in fcores if c_.x == 10 + j_ % 2 and c_.y == j_ // 2)) for j_ in range(16)]
        assert all(f_ in WOT for f_ in owner_fi), "owners must be wo cores"
        assert WCC == owner_fi[0], (WCC, owner_fi[0])   # writers reach the concat core through the owner table
        # wo-core multicast rects: ext columns (x 10-11, rows 0..y_ar-1) + bottom rows left / right of the DRAM split (x < x_ar)
        _yb = min([fcores[f_].y for f_ in ext_list if fcores[f_].x < 10] or [GY])   # first ext-only row (8; GDN24: none -> rows 0-7 only)
        def _pr(lx0, ly0, lx1, ly1):
            if lx1 < lx0 or ly1 < ly0: return [0, 0, 0, 0, 0]
            a_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx0, ly0)); b_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx1, ly1))
            return [a_.x, a_.y, b_.x, b_.y, (lx1 - lx0 + 1) * (ly1 - ly0 + 1)]
        _xr = min(fcores[f_].x for f_ in ARC) if ARC else GXF   # AR corner starts here (bottom rows)
        _ybe = GYF if _yb == GY and not any(fcores[f_].x < 10 for f_ in ext_list) else _yb   # GDN24: no wo cores in rows 8-9
        RECTS_E = _pr(10, 0, 11, _yb - 1) + _pr(0, _ybe, fsplit - 1, GYF - 1) + _pr(fsplit, _ybe, min(_xr, GXF) - 1, GYF - 1)
        def self_re(fi):
            c_ = fcores[fi]
            return 0 if c_.x >= 10 and c_.y < _yb else (1 if c_.x < fsplit else 2)
        assert sum(RECTS_E[4::5]) == len(wo_cores), (RECTS_E, len(wo_cores))
        gold_wo = []
        for L in range(NL):
            g = torch.zeros(M, WN_ * TILE)
            if is_attn_layer(L):
                A_ = torch.zeros(M, WK_ * TILE)
                sg = torch.sigmoid(gold_qkv[L][:, 64 * TILE:112 * TILE])
                for u_ in range(NU):
                    for h_ in range(6):
                        for j_ in range(4):
                            A_[4 * u_ + j_, h_ * 256:(h_ + 1) * 256] = golds_s[L][u_, 4 * h_ + j_, :]
                g = (A_ * sg) @ woq[L]
            gold_wo.append(g)
        print(f"[rwo] wo split-N on {len(wo_cores)} ext cores ({min(len(v) for v in WOT.values())}-{max(len(v) for v in WOT.values())} tiles), concat {WCC}, rot {rot}, rects {RECTS_E}", flush=True)
    HC = []
    if RQNR:
        assert RQKV and ('MKA_WORING', '1') not in MKDEF and ('MKA_TREEFLAG', '1') not in MKDEF   # RWO optional (GDN24 decouple)
        QT_ = 8 * NU
        MKDEF = MKDEF + [('MKA_INVHD_BITS', f"{f2u(1.0 / 256)}u")]
        # head cores: owners (10 + s, k) except the concat core (its CB_A must keep zero rows for the A multicast)
        HC = [fcores.index(next(c_ for c_ in fcores if c_.x == 10 + (j_ % 2) and c_.y == j_ // 2)) for j_ in range(1, 8)]
        if RWO: assert WCC not in HC and all(h_ in WOT for h_ in HC)
        torch.manual_seed(13)
        g_q = (0.5 + torch.rand(6, 256)) / 16.0   # q head-norm gamma with the 1/sqrt(256) softmax scale folded in
        g_k = 0.5 + torch.rand(256)
        pos = torch.tensor([float(CTX - 4 + (r_ % 4)) for r_ in range(M)])   # verify row 4u + j at position pos_u + j
        inv = 1.0 / (1.0e7 ** (torch.arange(32).float() / 32.0))
        ang = pos[:, None] * inv[None, :]
        cos_t, sin_t = torch.cos(ang), torch.sin(ang)
        hg = torch.zeros(NHEAD_ := 7, 10, TILE, TILE)
        for h_ in range(7):
            g_ = g_q[h_] if h_ < 6 else g_k
            for t_ in range(8): hg[h_, t_] = g_[t_ * 32:(t_ + 1) * 32][None, :].repeat(TILE, 1)
            hg[h_, 8], hg[h_, 9] = cos_t, sin_t
        hg_t = ttnn.from_torch(hg.reshape(1, 1, -1, TILE).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hgq = ttnn.to_torch(hg_t).float().reshape(7, 10, TILE, TILE)
        kd_t = ttnn.from_torch(torch.zeros(1, 1, NL * TILE, 8 * TILE).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        def norm_rope(X, h_):   # X [M, 256] -> gamma * X / rms, RoPE on dims 0-63 (bf16 constants as on device)
            gq_ = torch.cat([hgq[h_, t_, 0] for t_ in range(8)])
            Y = X / torch.sqrt((X * X).mean(-1, keepdim=True) + 1e-6) * gq_
            c_, s_ = hgq[h_, 8], hgq[h_, 9]
            y0, y1 = Y[:, :32].clone(), Y[:, 32:64].clone()
            Y[:, :32], Y[:, 32:64] = y0 * c_ - y1 * s_, y1 * c_ + y0 * s_
            return Y
        gold_k = {}
        for L in range(NL):
            if not is_attn_layer(L): continue
            Xq = gold_qkv[L].double()
            qh = [norm_rope(Xq[:, h_ * 256:(h_ + 1) * 256].float(), h_) for h_ in range(6)]
            gold_k[L] = norm_rope(Xq[:, 48 * TILE:56 * TILE].float(), 6)
            g_ = torch.zeros(NU, TILE, DHD)
            kq_, vq_ = KVQ[L]
            for u_ in range(NU):
                Qu = torch.zeros(TILE, DHD)
                for h_ in range(6):
                    for j_ in range(4): Qu[4 * h_ + j_] = qh[h_][4 * u_ + j_]
                Qu = Qu.bfloat16().float()
                p_ = torch.exp(Qu[:NQH * T] @ kq_[u_].T)
                g_[u_, :NQH * T] = (p_ @ vq_[u_]) / p_.sum(-1, keepdim=True)
            golds_s[L] = g_
            # wo golden from the new attention output (gate unchanged)
            A_ = torch.zeros(M, 48 * TILE)
            sg = torch.sigmoid(gold_qkv[L][:, 64 * TILE:112 * TILE])
            for u_ in range(NU):
                for h_ in range(6):
                    for j_ in range(4):
                        A_[4 * u_ + j_, h_ * 256:(h_ + 1) * 256] = g_[u_, 4 * h_ + j_, :]
            if RWO: gold_wo[L] = (A_ * sg) @ woq[L]
        print(f"[rqnr] head cores {HC} (q 0-5, k 6), Q from the real qkv; SDPA golden recomputed", flush=True)
    def head_args_w(fi):
        if not RQNR: return []
        return [HC.index(fi) if fi in HC else 255] + sum(([fphys[h_].x, fphys[h_].y] for h_ in HC), []) + [kd_t.buffer_address()]
    RMASK = ('MKA_RMASK', '1') in MKDEF
    MASKC = {m_[-1]: u_ for u_, m_ in enumerate(members)} if RMASK else {}   # last-block core -> user
    if RMASK:
        assert RQNR
        mk = torch.ones(NU, TILE, TILE)
        GOLD_NOMASK = {}
        for u_ in range(NU):
            for r_ in range(NQH * T):
                j_ = r_ % 4
                mk[u_, r_, 28 + j_ + 1:] = 0.0   # bench: pos_u = CTX - 4, last block = positions CTX-32 .. CTX-1
        mk_t = ttnn.from_torch(mk.reshape(1, 1, -1, TILE).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for L in range(NL):   # SDPA (and wo) golden with the causal mask
            if not is_attn_layer(L): continue
            Xq = gold_qkv[L].double()
            qh = [norm_rope(Xq[:, h_ * 256:(h_ + 1) * 256].float(), h_) for h_ in range(6)]
            g_ = torch.zeros(NU, TILE, DHD)
            kq_, vq_ = KVQ[L]
            for u_ in range(NU):
                Qu = torch.zeros(TILE, DHD)
                for h_ in range(6):
                    for j_ in range(4): Qu[4 * h_ + j_] = qh[h_][4 * u_ + j_]
                Qu = Qu.bfloat16().float()
                p_ = torch.exp(Qu[:NQH * T] @ kq_[u_].T)
                g_nm = (p_ @ vq_[u_]) / p_.sum(-1, keepdim=True)
                p_[:, CTX - TILE:] *= mk[u_, :NQH * T]
                g_[u_, :NQH * T] = (p_ @ vq_[u_]) / p_.sum(-1, keepdim=True)
                GOLD_NOMASK.setdefault(L, torch.zeros(NU, TILE, DHD))[u_, :NQH * T] = g_nm
            golds_s[L] = g_
            A_ = torch.zeros(M, 48 * TILE)
            sg = torch.sigmoid(gold_qkv[L][:, 64 * TILE:112 * TILE])
            for u_ in range(NU):
                for h_ in range(6):
                    for j_ in range(4): A_[4 * u_ + j_, h_ * 256:(h_ + 1) * 256] = g_[u_, 4 * h_ + j_, :]
            if RWO: gold_wo[L] = (A_ * sg) @ woq[L]
        print(f"[rmask] causal mask on last-block cores {sorted(MASKC)}", flush=True)
    def attn_regold(L, gq):   # MK_REAL_AR checker: attention layer L's qkv / k / SDPA / wo golds from the x the AR delivered
        """gq: the raw qkv gold [M, QN*TILE] = rstd (CB 20 stand-in) * x_L @ Wqkv, x_L = the residual the previous AR delivered."""
        gold_qkv[L] = gq
        if not RQNR: return
        Xq = gq.double()
        qh = [norm_rope(Xq[:, h_ * 256:(h_ + 1) * 256].float(), h_) for h_ in range(6)]
        gold_k[L] = norm_rope(Xq[:, 48 * TILE:56 * TILE].float(), 6)
        g_ = torch.zeros(NU, TILE, DHD)
        kq_, vq_ = KVQ[L]
        for u_ in range(NU):
            Qu = torch.zeros(TILE, DHD)
            for h_ in range(6):
                for j_ in range(4): Qu[4 * h_ + j_] = qh[h_][4 * u_ + j_]
            Qu = Qu.bfloat16().float()
            p_ = torch.exp(Qu[:NQH * T] @ kq_[u_].T)
            if RMASK:
                GOLD_NOMASK.setdefault(L, torch.zeros(NU, TILE, DHD))[u_, :NQH * T] = (p_ @ vq_[u_]) / p_.sum(-1, keepdim=True)
                p_[:, CTX - TILE:] *= mk[u_, :NQH * T]
            g_[u_, :NQH * T] = (p_ @ vq_[u_]) / p_.sum(-1, keepdim=True)
        golds_s[L] = g_
        if RWO:
            A_ = torch.zeros(M, 48 * TILE)
            sg = torch.sigmoid(gq[:, 64 * TILE:112 * TILE])
            for u_ in range(NU):
                for h_ in range(6):
                    for j_ in range(4): A_[4 * u_ + j_, h_ * 256:(h_ + 1) * 256] = g_[u_, 4 * h_ + j_, :]
            gold_wo[L] = (A_ * sg) @ woq[L]
    def head_args_r(fi):
        if not RQNR: return []
        return [hg_t.buffer_address() if fi in HC else 0, HC.index(fi) * 10 if fi in HC else 0,
                mk_t.buffer_address() if fi in MASKC else 0, MASKC.get(fi, 0)]
    def wo_wargs(fi):
        if not RWO: return []
        t_ = WOT.get(fi, [])
        oc = owner_fi.index(fi) if fi in owner_fi else 255
        return [len(t_)] + (t_ + [0] * WO_MAXT_)[:WO_MAXT_] + [1 if fi == WCC else 0, oc] + RECTS_E + [self_re(fi) if fi in WOT else 3] + \
            sum(([fphys[f_].x, fphys[f_].y] for f_ in owner_fi), []) + [wod.buffer_address()]
    WORING = ('MKA_WORING', '1') in MKDEF
    WOHELP = os.environ.get('MKA_WOHELP', '1') == '1' and not WORING and not QSK   # QSK: Wo fills the ext ring in the GDN layers
    HLP = {}   # helper fi -> (ext fi, page offset, pages);  OWN[ext fi] = own pages
    OWN = {}
    if RWO:
        _sl = sorted(stream_idx, key=lambda f_: stream_idx[f_])
        for e_, fe in enumerate(wo_cores):
            np_ = len(WOT[fe]) * 48
            hs_ = [fs for k_, fs in enumerate(_sl) if WOHELP and k_ % len(wo_cores) == e_]
            parts = len(hs_) + 1
            cut = [round(np_ * i_ / parts) for i_ in range(parts + 1)]
            OWN[fe] = cut[1]
            for j_, fs in enumerate(hs_): HLP[fs] = (fe, cut[j_ + 1], cut[j_ + 2] - cut[j_ + 1])
    def wo_rargs(fi):
        if not RWO: return []
        cc = fphys[WCC]
        own = OWN.get(fi, 0); hl = (len(WOT[fi]) * 48 - own) if fi in WOT else 0
        he, ho, hn = HLP.get(fi, (None, 0, 0))
        hp = fphys[he] if he is not None else fphys[fi]
        hpage = (WOP[he] * banks + WOB[he] + ho * banks) if he is not None else 0
        return [len(WOT.get(fi, [])), (WOP[fi] * banks + WOB[fi]) if fi in WOT else 0, WOPL, wo_all.buffer_address(), cc.x, cc.y,
                own, hl, hp.x, hp.y, hpage, hn, ho]
    # ---- MKA_GDN24: GDN-role sems + hook (engine/gdn_role.py, the GDN agent's; default / absent = no-op: cores idle) ----
    GDN_ROLE = dict(kernels=[], cbs=[], io=[])
    if GDN24:
        gdn_crs = crs([fcores[i] for i in GDNC])
        # device-local global sems (spec v1.33: the waiter drains to its final target and resets at program end)
        GDN_SEMS = dict(in_ready=ttnn.create_global_semaphore(dev, gdn_crs, 0), attn_kv=ttnn.create_global_semaphore(dev, gdn_crs, 0),
                        vb0=ttnn.create_global_semaphore(dev, gdn_crs, 0), vb1=ttnn.create_global_semaphore(dev, gdn_crs, 0),
                        out_ready=ttnn.create_global_semaphore(dev, kgrid, 0))   # out_ready: target = the Q root (kgrid)
        GDN_SA = {k_: ttnn.get_global_semaphore_address(v_) for k_, v_ in GDN_SEMS.items()}
        def _gpr(lx0, lx1):
            a_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx0, GY)); b_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx1, GYF - 1))
            return [a_.x, a_.y, b_.x, b_.y, (lx1 - lx0 + 1) * (GYF - GY)]
        _gr = _gpr(0, fsplit - 1) + _gpr(fsplit, GXF - 1)
        assert _gr[4] + _gr[9] == len(GDNC), _gr
        MKDEF = MKDEF + [('MKA_GDN24', '1'), ('MKA_GDN_INREADY', f"{GDN_SA['in_ready']}u"), ('MKA_GDN_ATTNKV', f"{GDN_SA['attn_kv']}u"),
                         ('MKA_GDN_RECTS', '{{' + ','.join(map(str, _gr[:5])) + '},{' + ','.join(map(str, _gr[5:])) + '}}')]
        try:   # engine/ is on sys.path (bench_mka2 imports `program` from it): the pushed tree's own gdn_role first
            from gdn_role import gdn_role as _gdn_hook
        except ModuleNotFoundError:
            try:
                from models.demos.blackhole.qwen36.tt.megakernel.engine.gdn_role import gdn_role as _gdn_hook
            except ModuleNotFoundError:
                _gdn_hook = None
        # MKA_GDN_WAIT=1 (G2): the GDN-type layer's stall ends when all 24 GDN cores signalled their output rows (each
        # multicasts out_ready += 1 to every engine core, which waits >= 24 * (g + 1) and resets at program end) instead
        # of the MK_SGDN_US spin. Needs the hook.
        GDN_WAIT = os.environ.get('MKA_GDN_WAIT', '0') == '1'
        if _gdn_hook is not None and os.environ.get('MKA_GDN_HOOK', '1') == '1':
            import types
            if REAL_AR:   # the hosted AR sender owns CB ids RS / RED / HDR / AG on the corner cores: the GDN step's DFBs skip
                # them (e1_shim build_engine skip_cb_ids default (34, 35, 37, 39))
                assert sorted(AR.cbids[i] for i in (0, 1, 3, 5)) == [34, 35, 37, 39], AR.cbids
            b_ = types.SimpleNamespace(dev=dev, mesh=dev if MESH else None, NL=NL, AMASK=AMASK, is_attn=is_attn_layer, NU=NU, CTX=CTX,
                                       gdn_layers=[L for L in range(NL) if not is_attn_layer(L)], fphys=lambda c_: dev.worker_core_from_logical_core(c_),
                                       cbd=cbd, crs=crs, sems=GDN_SA, out_ready_target=(QRP.x, QRP.y), out_rects=(RECTS_F[:10] if GDN_WAIT else None), per_replay={},
                                       free_cb_ids=list(range(45)))
            GDN_ROLE = dict(_gdn_hook(b_, [fcores[i] for i in GDNC]))
            GDN_ROLE.setdefault('io', [])
            print(f"[gdn24] hook: {len(GDN_ROLE.get('kernels', []))} kernels, {len(GDN_ROLE.get('cbs', []))} CBs, {len(GDN_ROLE['io'])} io tensors", flush=True)
            if GDN_WAIT:
                MKDEF = MKDEF + [('MKA_GDN_WAIT', '1'), ('MKA_GDN_OUTREADY', f"{GDN_SA['out_ready']}u"), ('MKA_GDN_NC', str(len(GDNC)))]
                print(f"[gdn24] GDN stall = out_ready wait ({len(GDNC)} per GDN layer, multicast to RECTS_F), not the MK_SGDN_US spin", flush=True)
        else:
            assert not GDN_WAIT, "MKA_GDN_WAIT needs the GDN hook"
            print('[gdn24] no GDN hook: the 24 GDN cores stay idle', flush=True)
    # ---- GDN24 + MK_REAL_AR: AR_SEND hosted in the GDN role's reader / writer on the 4 AR corner cores ----
    def _gdn_host_ar(gks, sp):
        """gks: the GDN role's [reader, writer, compute] over the 24 GDN cores; sp: AR.sender_parts(this device's parts).
        The 20 other GDN cores keep the plain role kernels; the 4 corner cores get reader / writer sources generated from
        the role's (same step code): the loop runs over ALL layers of the pass, GDN step on GDN-type layers, then that
        layer's AR iterations (attention AR + MLP AR in program order); SendWriter is a kernel_main local (the fabric
        connection is open for the whole pass). The sender's [common, role] words are the corner cores' runtime args
        (the role has none)."""
        import re as _re
        corner = [ttnn.CoreCoord(*xy) for xy in ((10, 8), (11, 8), (10, 9), (11, 9))]
        ck = {(c.x, c.y) for c in corner}
        from models.demos.blackhole.qwen36.tt.megakernel.oneshot_ar import MKDIR as MKDIR_AR
        _aar = ('MK_REAL_AAR', '1') in MKDEF   # the attention AR (its own branch): two ARs on attention layers
        nar = [1 + (1 if _aar and is_attn_layer(L_) else 0) for L_ in range(NL)]
        gmask = sum(1 << L_ for L_ in range(NL) if not is_attn_layer(L_))
        a2 = sum(1 << L_ for L_ in range(NL) if nar[L_] == 2)
        out = []
        for kd in gks:
            src = str(kd.kernel_source)
            role = 'r' if src.endswith('g2_reader.cpp') else ('w' if src.endswith('g2_writer.cpp') else None)
            if role is None:
                out.append(kd); continue
            cores = [c for c in ttnn.corerange_to_cores(kd.core_ranges) if (c.x, c.y) not in ck]
            rest = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
            base = dict(compile_time_args=list(kd.compile_time_args), common_runtime_args=list(kd.common_runtime_args), config=kd.config)
            out.append(ttnn.KernelDescriptor(kernel_source=src, core_ranges=rest, defines=list(kd.defines), runtime_args=ttnn.RuntimeArgs(), **base))
            txt = open(src).read()
            hdr = 'mk_ar_send.hpp' if role == 'r' else 'mk_ar_sendw.hpp'
            obj = 'static mkar::SendReader ar_s;' if role == 'r' else 'mkar::SendWriter ar_s;   // kernel_main local: the fabric connection'
            loop = _re.search(r'\n  for \(uint32_t L = 0; L < (\d+); \+\+L\) \{\n', txt)
            assert loop, src
            end = txt.index('\n  }\n', loop.end())
            body = txt[loop.end():end]
            nc = sp['rd_ncommon'] if role == 'r' else sp['wr_ncommon']
            new_loop = (f'\n  {obj}\n  ar_s.init(0, {nc});\n  uint32_t ar_it = 0, gL = 0;\n'
                        f'  for (uint32_t LL = 0; LL < {NL}; ++LL) {{\n'
                        f'   if (({gmask}u >> LL) & 1u) {{\n    const uint32_t L = gL++;\n' + body + '\n   }\n'
                        f'   for (uint32_t k = 0; k < 1u + (({a2}u >> LL) & 1u); ++k) ar_s.iter(ar_it++);   // this layer\'s ARs\n'
                        '  }\n' + ('  ar_s.fini(MKAR_ITERS);\n' if role == 'r' else '  ar_s.fini();\n'))
            txt = txt[:loop.start()] + new_loop + txt[end + len('\n  }\n'):]
            # the AR phase headers right after the first include (the role's DM_OPT pragma, if any, stays first)
            inc = f'#include "{os.path.join(MKDIR_AR, hdr)}"\n'
            i_ = txt.index('#include')
            txt = txt[:i_] + inc + txt[i_:]
            hsrc = src.replace('.cpp', '_ar.cpp')
            open(hsrc, 'w').write(txt)
            rt = ttnn.RuntimeArgs()
            for c in corner:
                rt[c.x][c.y] = (sp['rd_args'] if role == 'r' else sp['wr_args'])[(c.x, c.y)]
            dfs = list(kd.defines) + list(sp['rd_defines'] if role == 'r' else sp['wr_defines'])
            dfs = dfs + [('MKAR_S_PUB', dict(AR_PUB_D)['MK_AR_PUB']), ('MKAR_S_PUBA', dict(AR_PUB_D)['MK_AR_PUBA'])]
            if os.environ.get('MKAR_SZONE', '0') == '1': dfs = dfs + [('MKAR_SZONE', '1')]   # hosted sender iteration zones (S-RIT / S-WIT)
            out.append(ttnn.KernelDescriptor(kernel_source=hsrc, core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in corner]),
                                             defines=dfs, runtime_args=rt, **base))
        return out
    def build(out, coord=None, plan_only=False):
        cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, xbuf), cbd(1, ttnn.bfloat4_b, (EVEN_T if GUPRE else 2 * GRP * KB * 2), rect), cbd(2, HDT, 1, rect),
               (cbd(3, HDT, 1, rect) if V12CB else ttnn.cb_descriptor_from_sharded_tensor(3, hbuf)), cbd(4, ttnn.bfloat4_b, 2 * HT * NDN, rect), cbd(5, ttnn.bfloat4_b, 2 * w2_tiles, rect),
               (cbd(16, ttnn.bfloat16, 1, rect) if V12 else ttnn.cb_descriptor_from_sharded_tensor(16, out)), (cbd(6, ttnn.bfloat8_b, 240 if QSK else QH, rect) if RQKV else cbd(6, ttnn.bfloat4_b, KT * MIXN, rect)),
               ttnn.cb_descriptor_from_sharded_tensor(7, ymix), cbd(8, ttnn.bfloat16, 1, rect),
               # ATTN working set (spec CBs 9-12 + scratch 24-29), full grid
               cbd(9, ttnn.bfloat16, QT_ if RQNR else 32, kgrid), cbd(10, KVDT, 8 * 2 * NBB, kgrid), cbd(11, KVDT, 8 * 2 * NBB, kgrid), cbd(12, ttnn.float32, 9, kgrid),
               cbd(24, ttnn.bfloat16, 1, kgrid), cbd(25, ttnn.bfloat16, 2 * NBB, kgrid), *([cbd(26, ttnn.float32, 18, crs([fcores[i] for i in ACT if i in stream_idx])), cbd(26, ttnn.float32, 9, ext_crs)] if RWO else [cbd(26, ttnn.float32, 18, kgrid)]), cbd(27, ttnn.float32, 1, kgrid),
               cbd(28, ttnn.bfloat16, 8, kgrid), cbd(29, ttnn.bfloat8_b, 8, kgrid)]
        if RQKV:   # STREAM_EXT weight ring FIRST (lowest CB addresses on the ext cores), then the full-grid qkv CBs
            cbs = ([] if (QSK and not RWO) else [cbd(19, ttnn.bfloat8_b, RING_P_ if RWO else 2 * KT, ext_crs)]) + cbs + [cbd(13, ttnn.float32, 1, kgrid), cbd(14, ttnn.float32, 2, kgrid),
                   cbd(17, ttnn.bfloat16, 2, kgrid), cbd(18, ttnn.float32, 1, kgrid), ttnn.cb_descriptor_from_sharded_tensor(20, rsp)]
        if TREEFLAG: cbs = cbs + [cbd(31, ttnn.bfloat16, 1, kgrid)]
        if RWO:   # ext-only: A, sig(gate) S with the in-place A (.) S view (same memory), owner CB_XO; CB_QO grows to 5
            _p = 2048
            # (CB 26, the tree receive CB, keeps 2 slots on STREAM and 1 slot on the ext cores: see the ATTN set above)
            cbs = [c_ for c_ in cbs if not any(f_.buffer_index == 17 for f_ in c_.format_descriptors)]
            assert sum(1 for c_ in cbs for f_ in c_.format_descriptors if f_.buffer_index == 17) == 0
            cbs = cbs + [cbd(17, ttnn.bfloat16, WO_MAXT_, kgrid), cbd(15, ttnn.bfloat16, 48, ext_crs), cbd(23, ttnn.bfloat16, 10, ext_crs.merge(crs([fcores[i] for i in sorted(ARC)])) if (REAL_AR and REAL_AAR and ARC) else ext_crs),
                         *([cbd(31, ttnn.bfloat8_b, 5 * 48, ext_crs)] if WORING else []),
                         *([cbd(30, ttnn.bfloat16, 11, ext_crs)] if RQNR else []),
                         # S / A(.)S (bf16) and, under RQNR, the head-norm fp32 scratch CB 31 share one 96 KB region: the head
                         # cores use CB 31 at qkv time, S arrives by multicast after the concat core's V loop (the concat core,
                         # which holds S from qkv time on, is never a head core). Saves 32 KB per ext core (owner L1, GDN24 wo)
                         ttnn.CBDescriptor(total_size=48 * _p, core_ranges=ext_crs, format_descriptors=[
                             ttnn.CBFormatDescriptor(buffer_index=21, data_format=ttnn.bfloat16, page_size=_p),
                             ttnn.CBFormatDescriptor(buffer_index=22, data_format=ttnn.bfloat16, page_size=_p)] +
                             ([ttnn.CBFormatDescriptor(buffer_index=31, data_format=ttnn.float32, page_size=2 * _p)] if RQNR else []))]
        if RQNR and not RWO:   # head cores without the wo set: CB_A (head tiles, full 48-page slot), CB_XO (head output), scratch, constants
            cbs = cbs + [cbd(15, ttnn.bfloat16, 48, ext_crs), cbd(23, ttnn.bfloat16, 10, ext_crs), cbd(31, ttnn.float32, 8, ext_crs), cbd(30, ttnn.bfloat16, 11, ext_crs)]
        if V12:   # drop the fm11 MLP CBs (keep 1-tile CB 1 / 2 for the hw_startup calls), add the v12 set
            keep = []
            for d_ in cbs:
                idx_ = d_.format_descriptors[0].buffer_index
                if idx_ in (3, 4, 5, 16): continue
                if idx_ == 1: d_ = cbd(1, ttnn.bfloat4_b, 1, rect)
                if idx_ == 2: d_ = cbd(2, HDT, 1, rect)
                keep.append(d_)
            cbs = [cbd(32, ttnn.bfloat4_b, LM.SLOT, rect)] + keep + [
                cbd(33, ttnn.bfloat16, 20, rect) if V12CB else ttnn.cb_descriptor_from_sharded_tensor(33, v_xk), cbd(34, ttnn.bfloat16, 14, rect), cbd(35, ttnn.bfloat16, 2, rect),
                cbd(36, ttnn.bfloat16, 16, rect) if V12CB else ttnn.cb_descriptor_from_sharded_tensor(36, v_rs),
                cbd(37, ttnn.bfloat16, 7, rect) if V12CB else ttnn.cb_descriptor_from_sharded_tensor(37, v_h7), cbd(38, ttnn.bfloat16, 1, rect),
                cbd(39, ttnn.bfloat16, 1, rect), cbd(40, ttnn.bfloat16, 10, rect), cbd(41, ttnn.bfloat16, 10, rect), ttnn.cb_descriptor_from_sharded_tensor(42, v_id),
                ttnn.cb_descriptor_from_sharded_tensor(43, rsp), cbd(44, ttnn.float32, 1, rect), ttnn.cb_descriptor_from_sharded_tensor(30, v_j),
                cbd(3, ttnn.bfloat16, 100, O_CRS) if v_pbuf is None else ttnn.cb_descriptor_from_sharded_tensor(3, v_pbuf),
                ttnn.cb_descriptor_from_sharded_tensor(4, v_xo), ttnn.cb_descriptor_from_sharded_tensor(5, v_x0),
                # CB_VPK: the stand-in's packing scratch (R < 32 only); MK_V12_CB: the same descriptor is v_xpk on STREAM (both 10 tiles)
                cbd(16, ttnn.bfloat16, 10 if V_R < 32 else 1, rect.merge(O_CRS) if V12CB else O_CRS)]
        if plan_only: return _order(cbs)   # MK_V12_CB: the list prefix that fixes the v12 addresses (AR / GDN-role CBs follow it)
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=rect, initial_value=0) for i in range(12)]
        if REAL_AR:   # this device's AR parts: sender kernels / CBs / fabric semaphores, the shared RS slot CB, owner CBs
            arp = ARP[coord[1]]
            assert dict(arp['recv_defines']).get('MKAR_DELIVER', '0') != '0', "MK_REAL_AR: the AR must deliver x (STREAM waits its flags)"
            sems = sems + arp['semaphores']
            _red = [CBP.mark_static(ttnn.CBDescriptor(total_size=d_.total_size, core_ranges=(O_CRS if GDN24 else d_.core_ranges), format_descriptors=list(d_.format_descriptors))) for d_ in arp['shared_cbs']]
            cbs = cbs + arp['sender_cbs'] + _red + arp['owner_cbs']
            if AR_GAMMA and AR_GEARLY:   # MK_AR_GEARLY: a dedicated γ CB (owners' CB region), filled from the MLP start
                cbs = cbs + [cbd(32, ttnn.bfloat16, AR_GT, O_CRS)]
            elif AR_GAMMA:   # CB 32 on the owners = pbuf+0 (dead from C-OSUM(L) to the next MLP's partials; spec v1.25 note)
                _g = ttnn.cb_descriptor_from_sharded_tensor(32, v_pbuf)
                _g.total_size = AR_GT * 2048
                _g.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=32, data_format=ttnn.bfloat16, page_size=2048)]
                cbs = cbs + [_g]
        rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        RS = os.environ.get('MKA_ROLESPLIT', '1') == '1'   # per-role binaries (spec v0.9.17)
        rta = {k_: (ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()) for k_ in ('s', 'e')}
        z10 = [0] * 10
        for fi, c in enumerate(fcores):
            if fi in RESV: continue
            u, r, n = grp[fi], rank[fi], len(members[grp[fi]])
            sb0, sb1 = RNG[fi]
            m = members[u]
            if r == 0: has_p, par, plvl = 0, 0, 0
            elif r % FAN != 0: has_p, par, plvl = 1, m[r - r % FAN], 0
            else: has_p, par, plvl = 1, m[0], 1
            ch0 = [m[k] for k in range(r + 1, min(r + FAN, n))] if r % FAN == 0 else []
            ch1 = [m[k] for k in range(FAN, n, FAN)] if r == 0 else []
            def xy(lst):
                o = []
                for k in range(FAN - 1): o += [fphys[lst[k]].x, fphys[lst[k]].y] if k < len(lst) else [0, 0]
                return o
            pp = fphys[par]
            sdpa_r = [kv.buffer_address() for kv in kvs] + [kv.buffer_address() for kv in kvvs] + [u, sb0, sb1, NBK, has_p, pp.x, pp.y, plvl, len(ch0)] + xy(ch0) + [len(ch1)] + xy(ch1) + [1 if r == 0 else 0, outs.buffer_address(), QRP.x, QRP.y]
            isS = fi in stream_idx
            WO_NOC0 = ('MKA_WO_NOC0', '1') in MKDEF
            if isS:
                i = stream_idx[fi]
                g_ = 1 if i < NGU else 0; rk = ROLE.get(i, 255)
                if V12: g_, rk = 0, 255   # fm11 gate|up / h-root roles off (v12 MLP)
                gp = (slot[i][1] * banks + slot[i][0]) if g_ else 0
                b, s0 = dn_slot(i); selfr = 0 if c.x < split else 1
                mb, ms0 = mx_slot(i)
                r_mk = [wgu_all.buffer_address(), gp] + RECTS + [selfr, hbuf.buffer_address(), rk if rk >= 2 else 255, g_, PGU]
                w_mk = [i] + RECTS + [selfr] + RXY + [hbuf.buffer_address(), wd_all.buffer_address(), s0 * banks + b, wgu_all.buffer_address(), gp, rk if rk < 2 else 255, g_, PGU, PDN, *((wq_all.buffer_address(), (q_slot.get(fi, 0) * banks + q_bank.get(fi, 0)) if fi in QROLE else 0, PQL) if RQKV else (wm_all.buffer_address(), ms0 * banks + mb, PWM))]
            else:
                g_ = 0; i = 0
                r_mk = [0, 0] + z10 + [0, 0, 255, 0, 0]
                w_mk = [0] + z10 + [0] + [0] * 8 + [0] * 5 + [255] + [0] * 3 + ([wq_all.buffer_address(), q_slot.get(fi, 0) * banks + q_bank.get(fi, 0), PQL] if (RQKV and fi in QROLE) else [0] * 3)
            selff = 0 if c.x < fsplit else (2 if (ARC and c.y >= _ay[0]) else 1)
            i_s = stream_idx.get(fi, 0)
            rd_w = (isS or EXT) and not WO_NOC0 and not RWO   # RWO: the wo bytes are real (ext ring)
            # stand-in bank spread: the reader reads 8-tile runs from page wo_page + 8k, which all sit in bank wo_page % 8.
            # With wo_page = i * 96 every core hammered bank 0 (and cores near it won). MKA_WOBANK=1 (default) puts core i on bank i % 8.
            wob = (fi % 8) if os.environ.get('MKA_WOBANK', '1') == '1' else 0
            _ra = rrt[c.x][c.y] = [1 if isS else 0] + r_mk + GSA + sdpa_r + [wo_dummy.buffer_address(), ((fi * WO_T120) if EXT else (i_s * WO_T)) + wob, (WO_T120 if isS else WO_TX) if rd_w else 0, (EXTRA_T if isS else EXTRA_TX) if rd_w else 0, SWAP(fi)] + ([tfk(fi)] if TREEFLAG else []) + q_rargs(fi)
            _wa = wrt[c.x][c.y] = [1 if isS else 0] + w_mk + GSA + [1 if fi == QROOT else 0, qten.buffer_address()] + RECTS_F + [selff] + RECTS + \
                [kv.buffer_address() for kv in kvs] + [kv.buffer_address() for kv in kvvs] + [u, sb0, sb1, NBK, wo_dummy.buffer_address(), (i * WO_T) if isS else 0, WO_T if isS else 0, BUDGET, QDONE_N, QRP.x, QRP.y, EXTRA_T if isS else 0, SWAP(fi)] + q_wargs(fi)
            nrecv = len(ch0) + len(ch1)
            _ca = crt[c.x][c.y] = [1 if isS else 0, g_, u, sb1 - sb0, nrecv, 1 if r == 0 else 0] + (list(QROLE.get(fi, (0, 0, 0, 0, 0, 0))[i_] for i_ in (0, 3, 4, 2)) + [gdn_on(fi)] + ([QROLE.get(fi, (0, 0))[1], len(WOT.get(fi, []))] if (RWO or QSK) else []) + ([1 if fi in HC else 0, 1 if (fi in HC or fi in MASKC) else 0] + ([(RNG[fi][1] - RNG[fi][0] - 1) if fi in MASKC else 0xFFFFFFFF] if RMASK else []) if RQNR else []) if RQKV else [])
            if V12:   # v12 arg tails (appended after every other arg)
                vw_, vr_, vc_ = v_args(fi, c)
                _wa = wrt[c.x][c.y] = list(_wa) + vw_; _ra = rrt[c.x][c.y] = list(_ra) + vr_; _ca = crt[c.x][c.y] = list(_ca) + vc_
            _r3 = rta['s' if isS else 'e']; _r3[0][c.x][c.y] = _ra; _r3[1][c.x][c.y] = _wa; _r3[2][c.x][c.y] = _ca
        acc_kv = ttnn.TensorAccessorArgs(kvs[0]).get_compile_time_args(); acc_o = ttnn.TensorAccessorArgs(outs).get_compile_time_args()
        acc_q = ttnn.TensorAccessorArgs(qten).get_compile_time_args(); acc_wo = ttnn.TensorAccessorArgs(wo_dummy).get_compile_time_args()
        acc_wq = ttnn.TensorAccessorArgs(wq_all).get_compile_time_args() if RQKV else []; acc_qo = ttnn.TensorAccessorArgs(qkv_out).get_compile_time_args() if RQKV else []
        def _ks(rng, defs, rr, ww, cc, wdefs=(), cdefs=(), dmopt=None):
            return [ttnn.KernelDescriptor(kernel_source=f"{KD}/mka2_reader.cpp", core_ranges=rng, defines=MKDEF + defs, opt_level=dmopt, compile_time_args=[KT, KB, WTB, GRP, HCH, XTB, NL, NC, KVTB, OTB, QTB, NBB, WOTB] + acc_gu + acc_kv + acc_o + acc_wo + (acc_wq if RQKV else []) + (acc_qo if RQNR else []),
                                    runtime_args=rr, config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
              ttnn.KernelDescriptor(kernel_source=f"{KD}/mka2_writer.cpp", core_ranges=rng, defines=MKDEF + defs + list(wdefs), opt_level=dmopt, compile_time_args=[XTB, NGU, NDN, KT, HT, WTB, KB, GRP, HCH, NL, NC, QH if RQKV else KT * MIXN, S_WO, S_AR, KVTB, QTB, NBB, WOTB] + acc_dn + acc_gu + (acc_wq if RQKV else acc_wm) + acc_q + acc_kv + acc_wo + (acc_qo if RQKV else []) + (ttnn.TensorAccessorArgs(wv_all).get_compile_time_args() if V12 else []) + (ttnn.TensorAccessorArgs(ar_gam).get_compile_time_args() if (REAL_AR and AR_GAMMA) else []),
                                    runtime_args=ww, config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
              ttnn.KernelDescriptor(kernel_source=f"{KD}/mka2_compute.cpp", core_ranges=rng, defines=MKDEF + defs + list(cdefs), compile_time_args=[KT, KB, HT, NDN, GRP, NL, MIXN, NBB], runtime_args=cc,
                                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False))]
        if RS:
            _srng = crs([fcores[i] for i in ACT if i in stream_idx]); _erng = crs([fcores[i] for i in ACT if i not in stream_idx])
            _aar_d = [('MKAR_ATYPE', f'{AR_ATYPE}ull'), ('MKAR_CB_AIN', '23')] if (REAL_AR and REAL_AAR) else []
            _ard = (arp['recv_defines'] + [('MKAR_NCOMMON', str(arp['recv_ncommon'])), ('MKAR_ARGS_L1', str(ar_blk.buffer_address())), ('MKAR_TRID', '15')] + ([('MKAR_LEANZ', '1')] if ('MKA_LEAN_C', '1') in MKDEF else []) + ([('MKAR_DLV_NONE', '1')] if AR_DLVR else []) + _aar_d, arp['comp_defines']) if REAL_AR else ((), ())
            # MK_EXT_DM_OS=1 (spec v1.5a: DM-only Os allowed after a timing check): the EXT (owner) reader / writer at -Os,
            # the text-sizing group once QSK + MK_REAL_AR land (compute stays O2)
            _dmo = ttnn._ttnn.operations.experimental.fabric_mux.KernelBuildOptLevel.Os if os.environ.get('MK_EXT_DM_OS', '0') == '1' else None
            ks = _ks(_srng, [('MKA_ROLE_STREAM', '1')], *rta['s']) + _ks(_erng, [('MKA_ROLE_EXT', '1')], *rta['e'], *_ard, dmopt=_dmo)
            if REAL_AR:
                for kd_ in arp['sender_kernels']:   # attention ARs read the wo output CB (bit it of MKAR_ATYPE)
                    if REAL_AAR and not any(k_ == 'MKAR_ATYPE' for k_, _ in kd_.defines): kd_.defines = list(kd_.defines) + _aar_d
                if not GDN24: ks = ks + arp['sender_kernels']   # GDN24: hosted in the GDN role (corner cores, sender_parts)
        else:
            assert not REAL_AR, "MK_REAL_AR needs the per-role binaries (MKA_ROLESPLIT=1)"
            ks = _ks(kgrid, [], rrt, wrt, crt)
        cbs = _order(cbs)
        print('[cbs] ' + ' '.join(f'{f_.buffer_index}:{c_.total_size // 1024}K' for c_ in cbs for f_ in c_.format_descriptors), flush=True)
        if GDN24:
            _gk = list(GDN_ROLE.get('kernels', []))
            if REAL_AR: _gk = _gdn_host_ar(_gk, AR.sender_parts(arp))
            ks = ks + _gk; cbs = cbs + list(GDN_ROLE.get('cbs', []))
            def _cbsum(lx, ly):
                n_ = 0
                for d_ in cbs:
                    try:
                        if d_.core_ranges.contains(ttnn.CoreCoord(lx, ly)) and getattr(d_, 'buffer', None) is None: n_ += d_.total_size   # static CBs only
                    except Exception:
                        return None
                return n_
            print(f"[gdn24] CB bytes per core: STREAM (0,0) {_cbsum(0, 0)}  ext (10,0) {_cbsum(10, 0)}  GDN (0,{GY}) {_cbsum(0, GY)}", flush=True)
        if V12CB:   # the full list (AR + GDN-role CBs appended after the v12 prefix) must leave the planned v12 addresses unchanged
            _vchk = _v12_addrs(cbs)
            assert _vchk == {k_: VADDR[k_] for k_ in _vchk}, f"[v12cb] v12 CB addresses moved in the full list: {_vchk} vs plan {VADDR}"
            _e = CBP.end_per_core(CBP.plan(cbs))
            print(f"[v12cb] CB region end: STREAM (0,0) {_e.get((0, 0))}  owner (10,0) {_e.get((GX, 0))}  GDN (0,{GY}) {_e.get((0, GY))}", flush=True)
        if os.environ.get('MKA_CBDUMP') == '1':
            for d_ in cbs:
                print(f"[cbdump] ids {[f_.buffer_index for f_ in d_.format_descriptors]} size {d_.total_size} ranges {[(r_.start.x, r_.start.y, r_.end.x, r_.end.y) for r_ in d_.core_ranges.ranges()]}", flush=True)
        return ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=cbs)
    def _order(cbs):
        if RWO and GDN24:   # GDN24: kgrid is ONE core range (x 0-11, rows 0-7) intersecting both the STREAM rect and the ext
            # (owner) range. tt-metal places a CB at the end of its own range's allocator and raises every intersecting allocator, so
            # an ext-only CB listed before a kgrid CB lifts that kgrid CB (on STREAM too) above it: the 522 KB Wo ring put STREAM
            # 71 KB into the L1 buffers. Order: CBs on both roles first, then the role-disjoint ones (each lands on its own top).
            _both = lambda c_: any(r_.start.x < GX for r_ in c_.core_ranges.ranges()) and any(r_.end.x >= GX for r_ in c_.core_ranges.ranges())
            cbs = [c_ for c_ in cbs if _both(c_)] + [c_ for c_ in cbs if not _both(c_)]
        elif RWO:   # placement: a CB on kgrid sits above BOTH the STREAM-rect and the ext CBs; ext-only CBs listed FIRST land
            # at low addresses on the ext cores (beside the STREAM rect's own CBs, which they never share a core with)
            _ext_only = lambda c_: c_.core_ranges == ext_crs
            cbs = [c_ for c_ in cbs if _ext_only(c_)] + [c_ for c_ in cbs if not _ext_only(c_)]
        if REAL_AR and REAL_AAR:   # CB 23 (ext + the AR corner) FIRST: a multi-range CB lands at the max of its ranges' current
            # tops, and the corner's AR CBs (RS / AG staging, RS slots) would otherwise push it ~100 KB up on the owners
            _w23 = lambda c_: any(f_.buffer_index == 23 for f_ in c_.format_descriptors)
            cbs = [c_ for c_ in cbs if _w23(c_)] + [c_ for c_ in cbs if not _w23(c_)]
        return cbs
    def _v12_addrs(cbs):   # planned v12 landing-buffer addresses (MK_V12_CB)
        p_ = CBP.plan(cbs)
        a_ = dict(xk=CBP.address_of(p_, 33, (0, 0)), rs=CBP.address_of(p_, 36, (0, 0)), h7=CBP.address_of(p_, 37, (0, 0)), xpk=CBP.address_of(p_, 16, (0, 0)))
        if v_pbuf is None: a_['pbuf'] = CBP.address_of(p_, 3, (GX, 0))
        return a_
    # V12 drops the fm11 output CB 16 (re-used as CB_VPK on the owners): keep the unused fm11 output out of L1 (it was the L1 floor)
    out_t = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG if V12 else ws(rect, NDN * TILE))
    if os.environ.get("MK_L1DUMP") == "1":  # debug: every live L1 tensor's address
        for _n, _v in sorted(locals().items()):
            for _i, _t in enumerate(_v if isinstance(_v, list) else [_v]):
                if isinstance(_t, ttnn.Tensor) and _t.storage_type() == ttnn.StorageType.DEVICE and _t.memory_config().buffer_type == ttnn.BufferType.L1:
                    print(f"[l1dump] {_n}[{_i}] addr {_t.buffer_address()} shape {list(_t.shape)}", flush=True)
    if V12CB:   # spec §20 v12cb: plan the v12 CB addresses from the list prefix, then the AR parts that deliver into them
        VADDR.update(_v12_addrs(build(out_t, None, plan_only=True)))
        print(f"[v12cb] planned (CB base {CBP.CB_BASE}): {VADDR}; tensors kept: v_xo v_x0 v_id v_j{' v_pbuf (AR gamma alias)' if v_pbuf is not None else ''}", flush=True)
        if REAL_AR:
            AR_XT['xk'], AR_XT['xpk'] = VADDR['xk'], VADDR['xpk']
            ARP, ar_blk, _ardefs = _ar_parts()
            MKDEF = MKDEF + _ardefs
    if REAL_AR:   # per-device programs (fabric connections, owner / gather roles differ by ring position)
        _pcache = {}
        def PROG_PER_DEV(co):
            k_ = (co[0], co[1])
            if k_ not in _pcache: _pcache[k_] = build(out_t, co)
            return _pcache[k_]
        prog = PROG_PER_DEV(ttnn.MeshCoordinate(0, 0))
    else:
        prog = build(out_t)
    if ('MK_V12_CBCHK', '1') in MKDEF:   # every (core, CB) address of the program on device vs the host plan; raises on a mismatch
        # per-device programs (MK_REAL_AR ring roles): each device's probe runs its own CB list
        CBP.verify(dev, [PROG_PER_DEV(ttnn.MeshCoordinate(0, c_)) for c_ in range(AR.ring)] if REAL_AR else prog, tag=' v12cb')
    if GDN24:   # GDN-core CB headroom (spec rule): every L1 tensor reserves its range on ALL cores, CBs grow from the bottom
        _l1 = []
        for _n, _v in list(locals().items()):
            for _t in (_v if isinstance(_v, (list, tuple)) else [_v]):
                try:
                    if isinstance(_t, ttnn.Tensor) and _t.is_allocated() and _t.memory_config().buffer_type == ttnn.BufferType.L1:
                        _l1.append((_t.buffer_address(), _n))
                except Exception:
                    pass
        _lo = min(_l1) if _l1 else (None, None)
        try:
            _cbb = dev.allocator().get_base_allocator_addr(ttnn.HalMemType.L1)
        except Exception:
            _cbb = None
        print(f"[gdn24] L1 tensors (addr, name) {sorted(set(_l1))}", flush=True)
        _cbb = _cbb or int(os.environ.get('MKA_L1_CB_BASE', '112000'))   # BH CB base (from the clash message; no Python allocator binding)
        print(f"[gdn24] L1 tensors {len(_l1)}; lowest {_lo[1]} @ {_lo[0]}; CB base {_cbb}; GDN-core CB space {(_lo[0] - _cbb) if _lo[0] else '?'} B", flush=True)
    return dict(locals())


# ================================================================================================================== #
# Engine-ON integration API (integ/DESIGN_integration.md §11; spec step 2 plumbing). setup_bench stays the bench's body
# untouched; the functions below run it on the model's mesh with the engine-config env, collect its io list and
# per-device programs, and add the integration arg blocks. With today's stand-ins the program reads its own synthetic
# weights / KV / x: the model's repacked weights, KV caches, page table and GDN ring addresses are RESIDENT and their
# addresses are delivered (arg blocks + common args), but no engine kernel reads them yet. When M3 lands the engine
# kernels switch to them; this file is where the swap happens.
# ================================================================================================================== #
import contextlib

# The engine configuration used inside vLLM (the §16 v1.22 baseline of run_si_mixed.sh + the attention defines).
# Override per key with QWEN36_R16_MK_ENV="KEY=VAL KEY=VAL ...". MKA_NU / FM_NL / MK_AMASK are set per bucket.
ENGINE_ENV = dict(
    # spec owner, canonical vLLM 1x8-mesh env (batch 3b, -mk-int b1d67e77745): config B + real wo + real attention AR
    # (u8 1470.6 / u1 1186.3 us per 8 layers, d26u02 mesh). GDN = the MK_SGDN_US stand-in (MKA_GDN24 off until gdnar). MKA_GDN_L1 must stay off (per bucket).
    MK_SGDN_US="64", MK_SAR_US="18", MKA_SWO_US="10", MKA_CTX="32768", MKA_KVDT="bfp8", MKA_NBB="1",
    MKA_BUDGET="16384", MKA_EXT="1", MK_MIXN="1",
    MK_DEFINES="MKA_GUPRE=1,MK_DNPIPE=1,MK_FLAG_FLUSH=1,MKA_MIXTRID=1,MKA_RQKV=1,MKA_PQGDN=1,MKA_PQDELAY=27000,"
               "MKA_RQNR=1,MKA_RMASK=1,MKA_QSK=1,MK_V12=1,MK_REAL_AR=1,MK_OWNER_RESID=0,MKA_RWO=1,MK_REAL_AAR=1",  # MK_V12_CB default-on under MK_V12 since -mk-int 9565e5e6b1d
    MK_EXT_DM_OS="1",  # batch 3b: needed to fit the EXT text with MK_REAL_AAR (without AAR the attention out is never reduced)
)
# the 'replay' arg block payload (written per launch inside the trace): the in-trace tensor addresses, in this order
PER_REPLAY_ARGS = ("kvpos", "ctrl", "cos", "sin", "pt_rows", "pt_users", "pt_groups")


def engine_env(n_users, n_layers):
    env = dict(ENGINE_ENV)
    for kv in os.environ.get("QWEN36_R16_MK_ENV", "").split():
        k, v = kv.split("=", 1)
        env[k] = v
    nl = min(int(env.pop("FM_NL", os.environ.get("QWEN36_R16_MK_NL", "8"))), n_layers, 32)  # MKA_AMASK is a u32
    env["FM_NL"] = str(nl)
    env["MKA_NU"] = str(max(1, -(-int(n_users) // 2)))  # own users per chip (u8 -> 4, u1 -> 1)
    env.setdefault("V12_R", str(min(32, 4 * int(n_users))))  # rows of the bucket (u8 32, u1 4)
    env.setdefault("MK_AMASK", hex(sum(1 << L for L in range(nl) if L % 4 == 3)))  # Qwen3.8: every 4th is attention
    return env


# Cross-bucket L1 pool (spec §20 bridge, until every engine L1 tensor is a CB): each verify bucket is its own engine
# build, and every L1 tensor reserves its address range on EVERY core, so two buckets' tensors stacked below the L1 top
# pushed the owners' lowest buffer to 506112, under the static CB end 884608 ("clash"; one bucket alone: 993280). Bucket
# programs never run concurrently (one verify replay at a time), so the k-th L1 tensor of the same (shape, dtype, layout,
# memory config, initial bytes) in a later bucket's setup IS the earlier bucket's tensor: identical start state, one
# address range. Tensors whose initial bytes differ get their own range (and may still clash: then the bucket fails to
# build and stays op-by-op). QWEN36_R16_MK_L1POOL=0 turns the pool off.
_L1POOL = {}  # id(mesh) -> {key: [tensor, ...]}
_ABLOCKS = {}  # id(mesh) -> the shared integration ArgBlocks


def _l1_key(shape, dtype, layout, mc, data=None):
    import hashlib

    h = None
    if data is not None:
        import torch

        t = data if isinstance(data, torch.Tensor) else None
        h = hashlib.sha1(t.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest() if t is not None else "opaque"
    return (tuple(int(x) for x in shape), str(dtype), str(layout), repr(mc), h)


def _is_l1(mc):
    return mc is not None and mc.buffer_type == ttnn.BufferType.L1


@contextlib.contextmanager
def _setup_context(mesh, env):
    """setup_bench's mesh contract (bench_mka2's MESH shims), ONLY while it runs: from_torch replicates (host tilize,
    then to_device with the memory config), to_torch reads device 0. Plus the engine-config env, and the cross-bucket
    L1 pool (from_torch / allocate_tensor_on_device into L1). All restored after."""
    saved_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    _ft0, _tt0, _at0 = ttnn.from_torch, ttnn.to_torch, ttnn.allocate_tensor_on_device
    pool = _L1POOL.setdefault(id(mesh), {}) if os.environ.get("QWEN36_R16_MK_L1POOL", "1") == "1" else None
    seen = {}  # key -> occurrences in THIS setup
    stats = dict(reused=0, new=0, bytes_reused=0)

    def _pooled(key, make):
        if pool is None:
            return make()
        k = seen[key] = seen.get(key, -1) + 1
        lst = pool.setdefault(key, [])
        if k < len(lst) and lst[k].is_allocated():
            stats["reused"] += 1
            return lst[k]
        t = make()
        stats["new"] += 1
        if k < len(lst):
            lst[k] = t
        else:
            lst.append(t)
        return t

    def _ft(*a, **k):
        d = k.pop("device", None)
        mc = k.pop("memory_config", None)
        if d is None:
            return _ft0(*a, **k)
        k.setdefault("mesh_mapper", ttnn.ReplicateTensorToMesh(mesh))

        def make():
            h = _ft0(*a, **k)
            return ttnn.to_device(h, d, memory_config=mc) if mc is not None else ttnn.to_device(h, d)

        if not _is_l1(mc):
            return make()
        src = a[0] if a else k.get("tensor")
        shp = getattr(src, "shape", ())
        return _pooled(_l1_key(shp, k.get("dtype"), k.get("layout"), mc, src), make)

    def _at(shape, dtype, layout, device, memory_config=None, *a, **k):
        make = lambda: _at0(shape, dtype, layout, device, memory_config, *a, **k)
        if not _is_l1(memory_config):
            return make()
        return _pooled(_l1_key(list(shape), dtype, layout, memory_config), make)

    def _tt(t, *a, **k):
        if t.storage_type() == ttnn.StorageType.DEVICE:
            return _tt0(ttnn.get_device_tensors(t)[0], *a, **k)
        return _tt0(t, *a, **k)

    ttnn.from_torch, ttnn.to_torch, ttnn.allocate_tensor_on_device = _ft, _tt, _at
    try:
        yield stats
    finally:
        ttnn.from_torch, ttnn.to_torch, ttnn.allocate_tensor_on_device = _ft0, _tt0, _at0
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _io_list(S):
    """The engine generic_op io list (bench_mka2's run-loop expression, from setup_bench's names)."""
    g = S.get
    io = [S["x"], S["wgu_all"], S["wd_all"], S["wm_all"], S["xbuf"], S["hbuf"], S["ymix"], S["qten"], S["outs"], S["wo_dummy"]]
    io += list(S["kvs"]) + list(S["kvvs"])
    io += [S["wq_all"], S["rsp"], S["qkv_out"]] if g("RQKV") else ([S["rsp"]] if g("V12") else [])
    io += [S["wo_all"], S["wod"]] if g("RWO") else []
    io += [S["hg_t"], S["kd_t"]] if g("RQNR") else []
    io += [S["mk_t"]] if g("RMASK") else []
    io += [S[n] for n in ("wv_all", "v_xk", "v_rs", "v_h7", "v_xpk", "v_pbuf", "v_xo", "v_x0", "v_id", "v_j")] if g("V12") else []
    io += [S["ar_out"], S["ar_blk"]] if g("REAL_AR") else []
    io += list(S["GDN_ROLE"].get("io", [])) if g("GDN24") else []
    # MK_V12_CB (spec §20): v12 scratch moved to CBs -> setup_bench leaves those names None (bench_mka2 skips them too)
    return [t for t in io if t is not None] + [S["out_t"]]


def weight_layouts(mesh):
    """The six weight kinds the engine is fed from weights_mk.repack (spec §19): mlp via v12's layout_mlp (the real
    STREAM layout), gamma (fixed layout), and qkv / wo / gdn_in / gdn_out in tile order until their track packers
    (layout_attn split-K qkv, the wo unit, GDN) take over -- resident and addressable, not yet read by the kernels."""
    from models.demos.blackhole.qwen36.tt.megakernel.integ import weights_mk as wm

    bank_loc = [(c.x, c.y) for c in mesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
    return {
        "mlp": wm.MlpLayout(bank_loc),
        "gamma": None,
        "qkv": wm.TileOrderLayout("qkv", "bfloat8_b"),
        "wo": wm.TileOrderLayout("wo", "bfloat8_b"),
        "gdn_in": wm.TileOrderLayout("gdn_in", "bfloat4_b"),
        "gdn_out": wm.TileOrderLayout("gdn_out", "bfloat4_b"),
    }


def build_program(mesh, n_users, T, layer_types, weights, kv_desc, gdn_desc):
    """-> program(x_in, x_out, per_replay) -> (io, MeshProgramDescriptor) for one verify bucket (engine kind mk).

    Builds the engine once (setup_bench on the mesh, engine_env), plus two integration arg blocks over the engine grid:
      'attn': [page_table addr, B, nb, n_attn_layers, (k addr, v addr) per attention layer, block_size, n_kv_heads]
    The per-replay in-trace tensors (kvpos, ctrl, cos, sin, page-table slices): the 'replay' arg block, rewritten by a
    one-core writer op right before the engine op in every replay (their addresses are fixed per capture). x_in / x_out: the stand-in engine runs on its own x; x_out is left to the caller."""
    from models.demos.blackhole.qwen36.tt.megakernel.engine import argblock as AB

    env = engine_env(n_users, len(layer_types))
    # HARD RULE (spec owner 11-10, batch 3d): real AR has a residual blow-up at R < 32 (u1 R=4, post-RC fix): with
    # MK_REAL_AR the engine is only ever built at R = 32 rows (8 users x T = 4); fewer live users are padded onto it.
    if "MK_REAL_AR=1" in env.get("MK_DEFINES", "").split(","):
        assert int(env.get("V12_R", "32")) == 32 and int(n_users) * int(T) == 32, (
            f"MK_REAL_AR engine at R={env.get('V12_R')} (n_users {n_users} x T {T}): only R = 32 is allowed; pad onto bucket 8")
    with _setup_context(mesh, env) as pool_stats:
        S = setup_bench(mesh, True, lambda t: 0.0)
    io0 = _io_list(S)
    per_dev = S.get("PROG_PER_DEV")
    try:  # v12's CB plan check (engine/cbplan.py, on -mk-v12cb): the in-model CB base / L1 high-water may differ from 112000
        from models.demos.blackhole.qwen36.tt.megakernel.engine import cbplan as CBP
    except ImportError:
        CBP = None
    if CBP is not None and os.environ.get("QWEN36_R16_MK_CBPLAN_VERIFY", "1") == "1":
        CBP.verify(mesh, [per_dev(ttnn.MeshCoordinate(r_, c_)) if per_dev else S["prog"]
                          for r_ in range(mesh.shape[0]) for c_ in range(mesh.shape[1])])
    grid = mesh.compute_with_storage_grid_size()
    full = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    # the integration arg blocks are model-global (KV bases, page table) or rewritten per launch (replay): ONE set per
    # mesh, shared by every bucket (each L1 tensor reserves its range on every core; see _L1POOL)
    blocks = _ABLOCKS.get(id(mesh))
    if blocks is None:
        blocks = _ABLOCKS[id(mesh)] = AB.ArgBlocks(mesh)
        attn = []
        for d in kv_desc or ():
            attn += [d.get("k_addr", 0), d.get("v_addr", 0)]
        kv0 = (kv_desc or [{}])[0]
        blocks.add("attn", full, [{(c.x, c.y): [kv0.get("pt_addr", 0), kv0.get("pt_B", 0), kv0.get("pt_nb", 0),
                                                len(kv_desc or ())] + attn + [kv0.get("block_size", 0), kv0.get("n_kv_heads", 0)]
                                   for c in AB._cores(full)}])
        blocks.add("replay", full, [{}], min_payload=len(PER_REPLAY_ARGS))
    # kind 3 ('gdn') is the GDN role's own block (gdn/u1/g2_role.py); the model's GDN state goes into the role's per-layer
    # table (gdn_desc[i]['tensors'], keyed like GdnRole.tl) when the role adopts it, not into a second block here.
    wb = {k: ((weights or {}).get(k) or {}).get("tensor") for k in ("mlp", "qkv", "wo", "gdn_in", "gdn_out", "gamma")}
    descs = {}
    for r in range(mesh.shape[0]):
        for c_ in range(mesh.shape[1]):
            co = ttnn.MeshCoordinate(r, c_)
            descs[(r, c_)] = per_dev(co) if per_dev else S["prog"]
            AB.check_cb_ids(descs[(r, c_)])
    info = dict(env=env, nl=int(env["FM_NL"]), nu=int(env["MKA_NU"]), n_io=len(io0), per_device=per_dev is not None,
                common_args=blocks.common_args(), l1pool=dict(pool_stats), n_gdn=len(gdn_desc or ()), n_attn=len(kv_desc or ()),
                weights={k: (t.buffer_address() if t is not None else 0) for k, t in wb.items()})

    def program(x_in, x_out, per_replay):
        """Inside the trace: (1) the 'replay' block := this replay's in-trace tensor addresses (one-core writer op),
        (2) -> the engine's io list (+ the arg-block tensors) and mesh program. The engine kernels read the arg blocks
        through the common-arg slots once they adopt them (info['common_args']); today's stand-ins ignore them."""
        blocks.write_op("replay", [(per_replay.get(n).buffer_address() if per_replay.get(n) is not None else 0)
                                   for n in PER_REPLAY_ARGS])
        mp = ttnn.MeshProgramDescriptor()
        for (r, c_), d in descs.items():
            co = ttnn.MeshCoordinate(r, c_)
            mp[ttnn.MeshCoordinateRange(co, co)] = d
        return io0 + blocks.tensors(), mp

    program.info = info
    program.setup = S
    program.blocks = blocks
    return program
