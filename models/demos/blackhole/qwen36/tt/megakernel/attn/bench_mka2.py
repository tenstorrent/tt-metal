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
dev = ttnn.open_device(device_id=int(os.environ.get('FM_DEV', '0')), trace_region_size=64 << 20)
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
xbuf = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TILE * NC, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, hs(rect, H))
XPRE = 1 if os.environ.get("FM_XPRE", "1") == "1" else 0
if XPRE:
    xbuf = ttnn.from_torch(xb.reshape(1, 1, M, H).bfloat16().float().repeat(1, 1, NC, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=hs(rect, H))
HDT = ttnn.bfloat8_b if os.environ.get('MK_H_BFP8') == '1' else ttnn.bfloat16  # -mk-tail: h broadcast in BFP8 halves its bytes
print(f'[mk1] h dtype {HDT}', flush=True)
hbuf = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TILE * NC, I]), HDT, ttnn.TILE_LAYOUT, dev, hs(rect, I))
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
MKDEF += [('MKA_AMASK', f'{AMASK & 0xFFFFFFFF}u'), ('MKA_S_GDN', str(int(S_GDN_US * 1350)))]
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
    _w = float(os.environ.get('MKA_EXTW', '1')); _d = NC + (NCF - NC) * _w
    EXTRA_T, EXTRA_TX = int(EXTRA_T * NC / _d), int(EXTRA_T * NC * _w / _d)
S_WO = int(float(os.environ.get("MKA_SWO_US", "10")) * MHZ)
# MKA_ARCORES="fi,fi,...": AR-only cores (spec: the AR's 4 fabric link-sender cores) - no MKA2 kernel, no KV, no qkv
_arc = os.environ.get('MKA_ARCORES', '106,107,118,119')   # default ON (spec v0.9.1); 'none' disables
ARC = set() if _arc in ('', 'none') else set(int(v) for v in _arc.split(','))
ACT = [i for i in range(NCF) if i not in ARC]
grp = [0] * NCF
for j_, i_ in enumerate(ACT): grp[i_] = j_ * NU // len(ACT)
members = [[i for i in ACT if grp[i] == u] for u in range(NU)]
if os.environ.get('MKA_MREV') == '1': members = [m_[::-1] for m_ in members]   # rank 0 / first KV blocks at the far end of each user's rows
rank = {i: members[grp[i]].index(i) for i in ACT}
kgrid = crs([fcores[i] for i in ACT]) if ARC else fgrid
GSEM = [ttnn.create_global_semaphore(dev, kgrid, 0) for _ in range(21)]
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
if ARC: print(f'[mka2] AR-only cores {sorted(ARC)} (phys {[(fphys[i].x, fphys[i].y) for i in sorted(ARC)]}): {len(ACT)} MKA2 cores', flush=True)
stream_idx = {}   # full-grid core -> STREAM index (mk1 numbering, x + 10*y on the 10x8 rect)
for i, c in enumerate(fcores):
    if c.x < GX and c.y < GY: stream_idx[i] = c.x + GX * c.y
torch.manual_seed(1)
kvs, golds_s, qs_all = [], [], []
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
_den = NC + (NCF - NC) * EXTW
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
    ext_list = [fi for fi in range(NCF) if fi not in stream_idx and fi not in ARC]
    s_by_i = {i_: fi for fi, i_ in stream_idx.items()}
    n_ext_tiles = 2 * len(ext_list); n_half = QN - n_ext_tiles
    assert n_half >= 0 and 2 * n_half <= len(stream_idx), (n_ext_tiles, n_half)
    for e, fi in enumerate(ext_list): QROLE[fi] = (3, 2 * e, 2, 0, KT, fi)
    for i_ in range(2 * n_half):
        fi = s_by_i[i_]; h_ = i_ % 2
        QROLE[fi] = (1 if h_ == 0 else 2, n_ext_tiles + i_ // 2, 1, h_ * QH, QH, s_by_i[i_ + 1] if h_ == 0 else s_by_i[i_ - 1])
    QDONE_N = sum(1 for v in QROLE.values() if v[0] in (1, 3))
    QSLOT = 2 * KT
    # bank-local slices: each qkv core's slice in its nearest bank (fcores logical coords vs the bank's preferred worker)
    qcores = sorted(QROLE)
    # byte-balanced: each slice class (ext 348 KB, STREAM half 87 KB) is spread evenly over the banks, nearest bank
    # first within the class (a count-only cap put most ext slices on the banks next to the ext cores -> hot banks)
    q_bank = {}
    for cls in ([fi for fi in qcores if QROLE[fi][0] == 3], [fi for fi in qcores if QROLE[fi][0] != 3]):
        caps = [len(cls) // banks + (1 if b_ < len(cls) % banks else 0) for b_ in range(banks)]
        pairs = sorted(((abs(fcores[fi].x - bx) + abs(fcores[fi].y - by), fi, b_) for fi in cls for b_, (bx, by) in enumerate(BANK_LOC)))
        for d_, fi, b_ in pairs:
            if fi in q_bank or caps[b_] == 0: continue
            q_bank[fi] = b_; caps[b_] -= 1
    q_slot = {}   # bank-local page offset of each core's slice (slices packed per bank)
    _tot = [0] * banks
    for fi in qcores:
        b_ = q_bank[fi]; q_slot[fi] = _tot[b_]; _tot[b_] += QROLE[fi][2] * QROLE[fi][4]
    PQL = max(_tot) * banks
    torch.manual_seed(7)
    wq_t = torch.randn(H, QN * TILE) / math.sqrt(H)
    gam = 0.5 + torch.rand(H)
    wf_t = gam[:, None] * wq_t   # diag(γ) W, folded at load (spec v0.4; × NORM_FIX is part of γ here)
    def q_pages(scale):
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
    for fi in qcores:
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
    ttnn.deallocate(xbuf)   # free the 80-core shard first so the full-grid one reuses its lockstep L1 range
    xbuf = ttnn.from_torch(xb.reshape(1, 1, M, H).bfloat16().float().repeat(1, 1, len(ACT), 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=hs(kgrid, H))
    EXTRA_T = EXTRA_TX = 0   # the qkv bytes are real now
    ext_crs = crs([fcores[fi] for fi in ext_list])
    print(f"[rqkv] {QN} tiles: ext {len(ext_list)}x2 full-K, STREAM {2 * n_half} K-halves; producers {QDONE_N}; MB/bank {[round(sum(QROLE[f][2] * QROLE[f][4] for f in qcores if q_bank[f] == b_) * BF8TB / 1e6, 2) for b_ in range(banks)]}; {QN * KT * BF8TB / 1e6:.1f} MB/layer", flush=True)
GDNIN = int(os.environ.get('MKA_GDNIN_CORES', '64'))   # GDN in-proj stand-in on the first 64 K-half cores (5.6 MB, spec 5.9),
def gdn_on(fi):                                         # independent of how many K-half pairs the qkv split uses
    return 1 if (fi in stream_idx and stream_idx[fi] < GDNIN and QROLE.get(fi, (0,))[0] in (1, 2)) else 0
def q_wargs(fi):
    if not RQKV: return []
    ro, n0, nt, k0, nk, peer = QROLE.get(fi, (0, 0, 0, 0, 0, fi))
    pp_ = fphys[peer]
    return [ro, n0, nt, pp_.x, pp_.y, qkv_out.buffer_address(), gdn_on(fi)] + wo_wargs(fi)
TREEFLAG = ('MKA_TREEFLAG', '1') in MKDEF
def tfk(fi):   # index of this core in its parent's child list (level-0 list or the root's level-1 list)
    r_ = rank[fi]
    return (r_ % FAN - 1) if r_ % FAN else (r_ // FAN - 1 if r_ else 0)
def q_rargs(fi):
    if not RQKV: return []
    ro = QROLE.get(fi, (0,))[0]
    return [ro, wq_all.buffer_address(), (q_slot[fi] * banks + q_bank[fi]) if ro else 0, PQL] + wo_rargs(fi)
RWO = ('MKA_RWO', '1') in MKDEF
if RWO and ('MKA_LEAN_C', '1') not in MKDEF: MKDEF = MKDEF + [('MKA_LEAN_C', '1')]
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
    assert max(len(v) for v in WOT.values()) <= 5
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
    _yb = min(fcores[f_].y for f_ in ext_list if fcores[f_].x < 10)   # first ext-only row (8)
    def _pr(lx0, ly0, lx1, ly1):
        if lx1 < lx0 or ly1 < ly0: return [0, 0, 0, 0, 0]
        a_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx0, ly0)); b_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx1, ly1))
        return [a_.x, a_.y, b_.x, b_.y, (lx1 - lx0 + 1) * (ly1 - ly0 + 1)]
    _xr = min(fcores[f_].x for f_ in ARC) if ARC else GXF   # AR corner starts here (bottom rows)
    RECTS_E = _pr(10, 0, 11, _yb - 1) + _pr(0, _yb, fsplit - 1, GYF - 1) + _pr(fsplit, _yb, min(_xr, GXF) - 1, GYF - 1)
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
def wo_wargs(fi):
    if not RWO: return []
    t_ = WOT.get(fi, [])
    oc = owner_fi.index(fi) if fi in owner_fi else 255
    return [len(t_)] + (t_ + [0] * 5)[:5] + [1 if fi == WCC else 0, oc] + RECTS_E + [self_re(fi) if fi in WOT else 3] + \
        sum(([fphys[f_].x, fphys[f_].y] for f_ in owner_fi), []) + [wod.buffer_address()]
WOHELP = os.environ.get('MKA_WOHELP', '1') == '1'
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
def build(out):
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, xbuf), cbd(1, ttnn.bfloat4_b, (EVEN_T if GUPRE else 2 * GRP * KB * 2), rect), cbd(2, HDT, 1, rect),
           ttnn.cb_descriptor_from_sharded_tensor(3, hbuf), cbd(4, ttnn.bfloat4_b, 2 * HT * NDN, rect), cbd(5, ttnn.bfloat4_b, 2 * w2_tiles, rect),
           ttnn.cb_descriptor_from_sharded_tensor(16, out), (cbd(6, ttnn.bfloat8_b, QH, rect) if RQKV else cbd(6, ttnn.bfloat4_b, KT * MIXN, rect)),
           ttnn.cb_descriptor_from_sharded_tensor(7, ymix), cbd(8, ttnn.bfloat16, 1, rect),
           # ATTN working set (spec CBs 9-12 + scratch 24-29), full grid
           cbd(9, ttnn.bfloat16, 32, kgrid), cbd(10, KVDT, 8 * 2 * NBB, kgrid), cbd(11, KVDT, 8 * 2 * NBB, kgrid), cbd(12, ttnn.float32, 9, kgrid),
           cbd(24, ttnn.bfloat16, 1, kgrid), cbd(25, ttnn.bfloat16, 2 * NBB, kgrid), *([cbd(26, ttnn.float32, 18, crs([fcores[i] for i in ACT if i in stream_idx])), cbd(26, ttnn.float32, 9, ext_crs)] if RWO else [cbd(26, ttnn.float32, 18, kgrid)]), cbd(27, ttnn.float32, 1, kgrid),
           cbd(28, ttnn.bfloat16, 8, kgrid), cbd(29, ttnn.bfloat8_b, 8, kgrid)]
    if RQKV:   # STREAM_EXT weight ring FIRST (lowest CB addresses on the ext cores), then the full-grid qkv CBs
        cbs = [cbd(19, ttnn.bfloat8_b, 2 * KT, ext_crs)] + cbs + [cbd(13, ttnn.float32, 1, kgrid), cbd(14, ttnn.float32, 2, kgrid),
               cbd(17, ttnn.bfloat16, 2, kgrid), cbd(18, ttnn.float32, 1, kgrid), ttnn.cb_descriptor_from_sharded_tensor(20, rsp)]
    if TREEFLAG: cbs = cbs + [cbd(31, ttnn.bfloat16, 1, kgrid)]
    if RWO:   # ext-only: A, sig(gate) S with the in-place A (.) S view (same memory), owner CB_XO; CB_QO grows to 5
        _p = 2048
        # (CB 26, the tree receive CB, keeps 2 slots on STREAM and 1 slot on the ext cores: see the ATTN set above)
        cbs = [c_ for c_ in cbs if not any(f_.buffer_index == 17 for f_ in c_.format_descriptors)]
        assert sum(1 for c_ in cbs for f_ in c_.format_descriptors if f_.buffer_index == 17) == 0
        cbs = cbs + [cbd(17, ttnn.bfloat16, 5, kgrid), cbd(15, ttnn.bfloat16, 48, ext_crs), cbd(23, ttnn.bfloat16, 10, ext_crs),
                     ttnn.CBDescriptor(total_size=48 * _p, core_ranges=ext_crs, format_descriptors=[
                         ttnn.CBFormatDescriptor(buffer_index=21, data_format=ttnn.bfloat16, page_size=_p),
                         ttnn.CBFormatDescriptor(buffer_index=22, data_format=ttnn.bfloat16, page_size=_p)])]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=rect, initial_value=0) for i in range(12)]
    rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    RS = os.environ.get('MKA_ROLESPLIT', '1') == '1'   # per-role binaries (spec v0.9.17)
    rta = {k_: (ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()) for k_ in ('s', 'e')}
    z10 = [0] * 10
    for fi, c in enumerate(fcores):
        if fi in ARC: continue
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
            gp = (slot[i][1] * banks + slot[i][0]) if g_ else 0
            b, s0 = dn_slot(i); selfr = 0 if c.x < split else 1
            mb, ms0 = mx_slot(i)
            r_mk = [wgu_all.buffer_address(), gp] + RECTS + [selfr, hbuf.buffer_address(), rk if rk >= 2 else 255, g_, PGU]
            w_mk = [i] + RECTS + [selfr] + RXY + [hbuf.buffer_address(), wd_all.buffer_address(), s0 * banks + b, wgu_all.buffer_address(), gp, rk if rk < 2 else 255, g_, PGU, PDN, *((wq_all.buffer_address(), (q_slot[fi] * banks + q_bank[fi]) if fi in QROLE else 0, PQL) if RQKV else (wm_all.buffer_address(), ms0 * banks + mb, PWM))]
        else:
            g_ = 0; i = 0
            r_mk = [0, 0] + z10 + [0, 0, 255, 0, 0]
            w_mk = [0] + z10 + [0] + [0] * 8 + [0] * 5 + [255] + [0] * 3 + ([wq_all.buffer_address(), q_slot[fi] * banks + q_bank[fi], PQL] if (RQKV and fi in QROLE) else [0] * 3)
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
        _ca = crt[c.x][c.y] = [1 if isS else 0, g_, u, sb1 - sb0, nrecv, 1 if r == 0 else 0] + (list(QROLE.get(fi, (0, 0, 0, 0, 0, 0))[i_] for i_ in (0, 3, 4, 2)) + [gdn_on(fi)] + ([QROLE.get(fi, (0, 0))[1], len(WOT.get(fi, []))] if RWO else []) if RQKV else [])
        _r3 = rta['s' if isS else 'e']; _r3[0][c.x][c.y] = _ra; _r3[1][c.x][c.y] = _wa; _r3[2][c.x][c.y] = _ca
    acc_kv = ttnn.TensorAccessorArgs(kvs[0]).get_compile_time_args(); acc_o = ttnn.TensorAccessorArgs(outs).get_compile_time_args()
    acc_q = ttnn.TensorAccessorArgs(qten).get_compile_time_args(); acc_wo = ttnn.TensorAccessorArgs(wo_dummy).get_compile_time_args()
    acc_wq = ttnn.TensorAccessorArgs(wq_all).get_compile_time_args() if RQKV else []; acc_qo = ttnn.TensorAccessorArgs(qkv_out).get_compile_time_args() if RQKV else []
    def _ks(rng, defs, rr, ww, cc):
        return [ttnn.KernelDescriptor(kernel_source=f"{KD}/mka2_reader.cpp", core_ranges=rng, defines=MKDEF + defs, compile_time_args=[KT, KB, WTB, GRP, HCH, XTB, NL, NC, KVTB, OTB, QTB, NBB, WOTB] + acc_gu + acc_kv + acc_o + acc_wo + (acc_wq if RQKV else []),
                                runtime_args=rr, config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
          ttnn.KernelDescriptor(kernel_source=f"{KD}/mka2_writer.cpp", core_ranges=rng, defines=MKDEF + defs, compile_time_args=[XTB, NGU, NDN, KT, HT, WTB, KB, GRP, HCH, NL, NC, QH if RQKV else KT * MIXN, S_WO, S_AR, KVTB, QTB, NBB, WOTB] + acc_dn + acc_gu + (acc_wq if RQKV else acc_wm) + acc_q + acc_kv + acc_wo + (acc_qo if RQKV else []),
                                runtime_args=ww, config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
          ttnn.KernelDescriptor(kernel_source=f"{KD}/mka2_compute.cpp", core_ranges=rng, defines=MKDEF + defs, compile_time_args=[KT, KB, HT, NDN, GRP, NL, MIXN, NBB], runtime_args=cc,
                                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False))]
    if RS:
        _srng = crs([fcores[i] for i in ACT if i in stream_idx]); _erng = crs([fcores[i] for i in ACT if i not in stream_idx])
        ks = _ks(_srng, [('MKA_ROLE_STREAM', '1')], *rta['s']) + _ks(_erng, [('MKA_ROLE_EXT', '1')], *rta['e'])
    else:
        ks = _ks(kgrid, [], rrt, wrt, crt)
    if RWO:   # placement: a CB on kgrid sits above BOTH the STREAM-rect and the ext CBs; ext-only CBs listed FIRST land
        # at low addresses on the ext cores (beside the STREAM rect's own CBs, which they never share a core with)
        _ext_only = lambda c_: c_.core_ranges == ext_crs
        cbs = [c_ for c_ in cbs if _ext_only(c_)] + [c_ for c_ in cbs if not _ext_only(c_)]
    print('[cbs] ' + ' '.join(f'{f_.buffer_index}:{c_.total_size // 1024}K' for c_ in cbs for f_ in c_.format_descriptors), flush=True)
    return ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=cbs)
out_t = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ws(rect, NDN * TILE))
prog = build(out_t)
def pcc_of(t, g): v = ttnn.to_torch(t).float().reshape(M, H); return torch.corrcoef(torch.stack([v.flatten(), g.flatten()]))[0, 1].item(), (v - g).abs().max().item()
for it in range(ITERS):
    t1 = time.time(); y = ttnn.generic_op([x, wgu_all, wd_all, wm_all, xbuf, hbuf, ymix, qten, outs, wo_dummy] + kvs + kvvs + ([wq_all, rsp, qkv_out] if RQKV else []) + ([wo_all, wod] if RWO else []) + [out_t], prog); ttnn.synchronize_device(dev)
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
        for L in range(NL):
            if not is_attn_layer(L): continue
            if RWO: gold_qkv[L][:, 64 * TILE:112 * TILE] = torch.sigmoid(gold_qkv[L][:, 64 * TILE:112 * TILE]) if it == 0 else gold_qkv[L][:, 64 * TILE:112 * TILE]
            pq_ = torch.corrcoef(torch.stack([qv[L].flatten(), gold_qkv[L].flatten()]))[0, 1].item()
            nan_ = [n for n in range(QN) if not torch.isfinite(qv[L][:, n*TILE:(n+1)*TILE]).all()]
            if nan_: print(f"[rqkv] iter {it} layer {L} non-finite tiles {nan_}", flush=True)
            bad_ = [n for n in range(QN) if torch.corrcoef(torch.stack([qv[L][:, n*TILE:(n+1)*TILE].flatten(), gold_qkv[L][:, n*TILE:(n+1)*TILE].flatten()]))[0, 1].item() < 0.999]
            print(f"[rqkv] iter {it} layer {L} qkv pcc {pq_:.5f} max|err| {(qv[L]-gold_qkv[L]).abs().max().item():.4f} |gold|max {gold_qkv[L].abs().max().item():.3f} bad tiles {bad_[:12]}", flush=True)
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
        print(f"[mka2] iter {it} layer {L} sdpa pcc {torch.corrcoef(torch.stack([v_.flatten(), g2.flatten()]))[0, 1].item():.5f} max|err| {(v_-g2).abs().max().item():.4f}", flush=True)
    if RWO:
        wv_ = ttnn.to_torch(wod).float().reshape(NL, M, -1)
        for L in range(NL):
            if not is_attn_layer(L): continue
            rows_ = slice(0, 4 * NU)
            pw_ = torch.corrcoef(torch.stack([wv_[L][rows_].flatten(), gold_wo[L][rows_].flatten()]))[0, 1].item()
            print(f"[rwo] iter {it} layer {L} wo pcc {pw_:.5f} max|err| {(wv_[L][rows_] - gold_wo[L][rows_]).abs().max().item():.4f} |gold|max {gold_wo[L].abs().max().item():.3f}", flush=True)
    print(f"[fm11] iter {it} wall {time.time()-t1:.2f}s last-layer pcc {p:.5f} max|err| {e:.4f} |gold|max {gold_last.abs().max().item():.3f}", flush=True)
ttnn.ReadDeviceProfiler(dev)
print('[fm11] profiler read', flush=True)
ttnn.close_device(dev)
print('[fm11] closed', flush=True)
