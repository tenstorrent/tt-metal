"""Fused Qwen3.8 MLP decode block v2 (one chip of TP8) as ONE generic_op on a 10x8 core rectangle:
68 gate|up cores (one h column each), 80 down cores (2 output tiles each, whole down slice prefetched into L1),
x [32, 5120] bf16 width-sharded over the 80 cores (2 tiles each) and multicast to all; h multicast to all.
Weights BFP4, every core's slice contiguous in one DRAM bank. env FM_ITERS, FM_GRP (gate|up blocks per read)."""
import math, os, time, torch, ttnn
KD = os.environ.get("MK_KD", "/work/mk1/kernels") or os.environ.get("FM_KD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels"))
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
hbuf = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TILE * NC, I]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, hs(rect, I))
WTB = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat4_b); XTB = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat16)
def cbd(idx, dt, pages, r):
    page = ttnn.Tile([TILE, TILE]).get_tile_size(dt)
    return ttnn.CBDescriptor(total_size=pages * page, core_ranges=r, format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)])
acc = ttnn.TensorAccessorArgs(wgu).get_compile_time_args() + ttnn.TensorAccessorArgs(wd).get_compile_time_args()
print(f"[fm11] xpre {XPRE} place {'near' if PLACE else 'rr'} gu/bank {[len(q) for q in per_bank]} cores {NC} gate_up {NGU} banks {banks} bytes gu {HT*KT*2*WTB/1e6:.2f}MB dn {HT*KT*WTB/1e6:.2f}MB grp {GRP}", flush=True)

NL = int(os.environ.get("FM_NL", "8"))
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
MHZ = 1350.0
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
def build(out):
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, xbuf), cbd(1, ttnn.bfloat4_b, 2 * GRP * KB * 2, rect), cbd(2, ttnn.bfloat16, 1, rect),
           ttnn.cb_descriptor_from_sharded_tensor(3, hbuf), cbd(4, ttnn.bfloat4_b, 2 * HT * NDN, rect), cbd(5, ttnn.bfloat4_b, 2 * w2_tiles, rect),
           ttnn.cb_descriptor_from_sharded_tensor(16, out), cbd(6, ttnn.bfloat4_b, 2 * KT * MIXN, rect),
           ttnn.cb_descriptor_from_sharded_tensor(7, ymix), cbd(8, ttnn.bfloat16, 1, rect)]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=rect, initial_value=0) for i in range(12)]
    rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        g_ = 1 if i < NGU else 0; rk = ROLE.get(i, 255)
        gp = (slot[i][1] * banks + slot[i][0]) if g_ else 0
        b, s0 = dn_slot(i); selfr = 0 if c.x < split else 1
        crt[c.x][c.y] = [g_]
        mb, ms0 = mx_slot(i)
        rrt[c.x][c.y] = [wgu_all.buffer_address(), gp] + RECTS + [selfr, hbuf.buffer_address(), rk if rk >= 2 else 255, g_, PGU]
        wrt[c.x][c.y] = [i] + RECTS + [selfr] + RXY + [hbuf.buffer_address(), wd_all.buffer_address(), s0 * banks + b, wgu_all.buffer_address(), gp, rk if rk < 2 else 255, g_, PGU, PDN, wm_all.buffer_address(), ms0 * banks + mb, PWM]
    ks = [ttnn.KernelDescriptor(kernel_source=f"{KD}/mk1_reader.cpp", core_ranges=rect, compile_time_args=[KT, KB, WTB, GRP, HCH, XTB, NL, NC] + acc_gu,
                                runtime_args=rrt, config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
          ttnn.KernelDescriptor(kernel_source=f"{KD}/mk1_writer.cpp", core_ranges=rect, compile_time_args=[XTB, NGU, NDN, KT, HT, WTB, KB, GRP, HCH, NL, NC, KT * MIXN, S_MIX, S_AR] + acc_dn + acc_gu + acc_wm,
                                runtime_args=wrt, config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
          ttnn.KernelDescriptor(kernel_source=f"{KD}/mk1_compute.cpp", core_ranges=rect, compile_time_args=[KT, KB, HT, NDN, GRP, NL, MIXN], runtime_args=crt,
                                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False))]
    return ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=cbs)
out_t = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ws(rect, NDN * TILE))
prog = build(out_t)
def pcc_of(t, g): v = ttnn.to_torch(t).float().reshape(M, H); return torch.corrcoef(torch.stack([v.flatten(), g.flatten()]))[0, 1].item(), (v - g).abs().max().item()
for it in range(ITERS):
    t1 = time.time(); y = ttnn.generic_op([x, wgu_all, wd_all, wm_all, xbuf, hbuf, ymix, out_t], prog); ttnn.synchronize_device(dev)
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
    print(f"[fm11] iter {it} wall {time.time()-t1:.2f}s last-layer pcc {p:.5f} max|err| {e:.4f} |gold|max {gold_last.abs().max().item():.3f}", flush=True)
ttnn.ReadDeviceProfiler(dev)
print('[fm11] profiler read', flush=True)
ttnn.close_device(dev)
print('[fm11] closed', flush=True)
