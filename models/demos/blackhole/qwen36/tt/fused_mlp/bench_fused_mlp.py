"""Fused Qwen3.8 MLP decode block v2 (one chip of TP8) as ONE generic_op on a 10x8 core rectangle:
68 gate|up cores (one h column each), 80 down cores (2 output tiles each, whole down slice prefetched into L1),
x [32, 5120] bf16 width-sharded over the 80 cores (2 tiles each) and multicast to all; h multicast to all.
Weights BFP4, every core's slice contiguous in one DRAM bank. env FM_ITERS, FM_GRP (gate|up blocks per read)."""
import math, os, time, torch, ttnn
KD = os.environ.get("FM_KD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels"))
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
print(f"[fm2] noc x {phys_x} split at logical {split} rects {RECTS}", flush=True)
torch.manual_seed(0)
x_t = torch.randn(M, H); wg_t = torch.randn(H, I) / math.sqrt(H); wu_t = torch.randn(H, I) / math.sqrt(H); wd_t = torch.randn(I, H) / math.sqrt(I)
def tl(m, r, c): return m[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
BANK_LOC = [(c.x, c.y) for c in dev.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
def assign(n_items, cap_total):
    caps = [cap_total // banks + (1 if b < cap_total % banks else 0) for b in range(banks)]
    pairs = sorted(((abs(cores[i].x - bx) + abs(cores[i].y - by), i, b) for i in range(n_items) for b, (bx, by) in enumerate(BANK_LOC)))
    out = {}
    for d, i, b in pairs:
        if i in out or caps[b] == 0: continue
        out[i] = b; caps[b] -= 1
    return out
PLACE = os.environ.get("FM_PLACE", "near") == "near"
gu_bank = assign(HT, HT) if PLACE else {q: (q * banks) // HT for q in range(HT)}
dn_bank = assign(NC, NC) if PLACE else {c: c % banks for c in range(NC)}
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
def ws(r, w): return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(r, (M, w), ttnn.ShardOrientation.ROW_MAJOR))
def hs(r, w): return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(r, (TILE, w), ttnn.ShardOrientation.ROW_MAJOR))
x = ttnn.from_torch(x_t.reshape(1, 1, M, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ws(rect, XSW * TILE))
xb = ttnn.to_torch(x).float().reshape(M, H)
gold = (torch.nn.functional.silu(xb @ wgq) * (xb @ wuq)).bfloat16().float() @ wdq
xbuf = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TILE * NC, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, hs(rect, H))
hbuf = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TILE * NC, I]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, hs(rect, I))
WTB = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat4_b); XTB = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat16)
def cbd(idx, dt, pages, r):
    page = ttnn.Tile([TILE, TILE]).get_tile_size(dt)
    return ttnn.CBDescriptor(total_size=pages * page, core_ranges=r, format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)])
acc = ttnn.TensorAccessorArgs(wgu).get_compile_time_args() + ttnn.TensorAccessorArgs(wd).get_compile_time_args()
print(f"[fm2] place {'near' if PLACE else 'rr'} gu/bank {[len(q) for q in per_bank]} cores {NC} gate_up {NGU} banks {banks} bytes gu {HT*KT*2*WTB/1e6:.2f}MB dn {HT*KT*WTB/1e6:.2f}MB grp {GRP}", flush=True)
for it in range(ITERS):
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ws(rect, NDN * TILE))
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, xbuf), cbd(1, ttnn.bfloat4_b, 2 * GRP * KB * 2, rect), cbd(2, ttnn.bfloat16, 1, rect),
           ttnn.cb_descriptor_from_sharded_tensor(3, hbuf), cbd(4, ttnn.bfloat4_b, HT * NDN, rect), cbd(5, ttnn.bfloat4_b, 2 * GRP * KB * 2, rect), ttnn.cb_descriptor_from_sharded_tensor(16, out)]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=rect, initial_value=0) for i in range(4)]
    ks = []
    groups = [("gu", [c for i, c in enumerate(cores) if 0 < i < NGU]), ("coord", [cores[0]]), ("dn", [c for i, c in enumerate(cores) if i >= NGU])]
    for name, cl in groups:
        is_gu = 0 if name == "dn" else 1; is_coord = 1 if name == "coord" else 0
        rrt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c in cl:
            i = cores.index(c)
            gp = (slot[i][1] * banks + slot[i][0]) if is_gu else 0
            b, s0 = dn_slot(i)
            rrt[c.x][c.y] = [wgu.buffer_address(), gp]
            wrt[c.x][c.y] = [i, x.buffer_address()] + RECTS + [0 if c.x < split else 1, p0.x, p0.y, xbuf.buffer_address(), hbuf.buffer_address(), wd.buffer_address(), s0 * banks + b, wgu.buffer_address(), gp]
        r = crs(cl)
        if is_gu: ks.append(ttnn.KernelDescriptor(kernel_source=f"{KD}/fm4_reader.cpp", core_ranges=r, compile_time_args=[KT, KB, WTB, GRP] + ttnn.TensorAccessorArgs(wgu).get_compile_time_args(), runtime_args=rrt, config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)))
        ks.append(ttnn.KernelDescriptor(kernel_source=f"{KD}/fm7_writer.cpp", core_ranges=r, defines=([("DRAIN_FIRST", "1")] if os.environ.get("FM_DRAIN") == "1" else []), compile_time_args=[XSW, XTB, NC, NGU, NDN, is_gu, is_coord, KT, HT, WTB, KB, GRP] + ttnn.TensorAccessorArgs(wd).get_compile_time_args() + ttnn.TensorAccessorArgs(wgu).get_compile_time_args(), runtime_args=wrt, config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)))
        ks.append(ttnn.KernelDescriptor(kernel_source=f"{KD}/fm4_compute.cpp", core_ranges=r, compile_time_args=[KT, KB, HT, NDN, is_gu, GRP],
                  config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False)))
    prog = ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=cbs); PROG_LAST = prog
    t1 = time.time()
    y = ttnn.generic_op([x, wgu, wd, xbuf, hbuf, out], prog)
    ttnn.synchronize_device(dev)
    yv = ttnn.to_torch(y).float().reshape(M, H)
    pcc = torch.corrcoef(torch.stack([yv.flatten(), gold.flatten()]))[0, 1].item()
    # DEBUG_INTERMEDIATES: gathered x and h landing zones vs golden
    xv = ttnn.to_torch(xbuf).float().reshape(NC, M, H); hv = ttnn.to_torch(hbuf).float().reshape(NC, M, I)
    hg = (torch.nn.functional.silu(xb @ wgq) * (xb @ wuq))
    def pc(a, b): return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
    print(f"[fm2]   x landing pcc core0 {pc(xv[0], xb):.5f} core79 {pc(xv[-1], xb):.5f}; h landing pcc core0 {pc(hv[0], hg):.5f} core79 {pc(hv[-1], hg):.5f} "
          f"h col0 {pc(hv[0][:, :32], hg[:, :32]):.4f} nan_h {torch.isnan(hv[0]).sum().item()} h|max| {hv[0].abs().max().item():.3f} vs {hg.abs().max().item():.3f}", flush=True)
    print(f"[fm2] iter {it} wall {time.time()-t1:.1f}s pcc {pcc:.5f} max|err| {(yv-gold).abs().max().item():.4f} |gold|max {gold.abs().max().item():.3f}", flush=True)
REPS = int(os.environ.get("FM_REPS", "200"))
def timed(fn, label, nbytes):
    fn(); ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0); fn(); ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t1 = time.time()
    for _ in range(REPS): ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev); us = (time.time() - t1) / REPS * 1e6
    ttnn.release_trace(dev, tid)
    print(f"[sil] {label}: {us:.1f} us per call over {REPS} trace replays  ({nbytes/us/1e3:.0f} GB/s)", flush=True)
    return us
# fused: rebuild the program once more and replay it
out_t = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ws(rect, NDN * TILE))
_last = {}
def fused_call():
    _last['y'] = ttnn.generic_op([x, wgu, wd, xbuf, hbuf, out_t], prog)
tot = HT * KT * 2 * WTB + HT * KT * WTB
prog = PROG_LAST
cbs_fix = None
timed(fused_call, "fused MLP block (1 generic_op)", tot)
# baseline: DSMC gate_up + down (prod-like, 4 workers per bank), same shapes/dtypes
def rm(n): return ttnn.num_cores_to_corerangeset(n, dev.compute_with_storage_grid_size(), row_wise=True)
def wsl1(grid, wt): return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(grid, (M, wt * 32), ttnn.ShardOrientation.ROW_MAJOR))
def dsw(K_, N_, wt_):
    pn = math.ceil(N_ / (32 * banks)) * 32 * banks
    dg = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, ttnn.ShardSpec(dg, (K_, pn // banks), ttnn.ShardOrientation.ROW_MAJOR))
    return ttnn.from_torch(wt_.reshape(1, 1, K_, N_), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
wgu_ds = dsw(H, 2 * I, torch.cat([wgq, wuq], 1)); wd_ds = dsw(I, H, wdq)
xg = ttnn.from_torch(xb.reshape(1, 1, M, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=wsl1(rm(32), 5))
hh = ttnn.from_torch(torch.zeros(1, 1, M, I), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=wsl1(rm(17), 4))
ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)
pcg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=2, per_core_M=1, per_core_N=4, fused_activation=None, num_workers_per_dram_bank=4, use_multicore_pipeline=True)
pcd = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=2, per_core_M=1, per_core_N=4, fused_activation=None, num_workers_per_dram_bank=4, use_multicore_pipeline=True)
def base_call():
    _last['gu'] = ttnn.linear(xg, wgu_ds, program_config=pcg, memory_config=wsl1(rm(34), 4), dtype=ttnn.bfloat16, compute_kernel_config=ck)
    _last['dn'] = ttnn.linear(hh, wd_ds, program_config=pcd, memory_config=wsl1(rm(40), 4), dtype=ttnn.bfloat16, compute_kernel_config=ck)
timed(base_call, "DSMC gate_up + down (2 ops, no SwiGLU/glue)", tot)
ttnn.close_device(dev)
