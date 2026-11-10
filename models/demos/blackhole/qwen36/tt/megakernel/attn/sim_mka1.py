"""MKA1: megakernel attention block, verify-SDPA phase, one P150 (TP8 slice), KV-split "user" mode at u8:
4 own users, 6 q heads over 1 KV head (rows (h, j) = 4h + j, 24 of 32 used), BFP4 KV, context MKA_CTX.
All 110 cores do SDPA (4 groups of 27/28, one per own user; time-multiplexed with STREAM in the megakernel), while
the 80 STREAM cores' writers stream MKA_PF_KB of weights each (wo slice + next layer's MLP: ~340 KB) as contention.
env: MKA_CTX (32768), MKA_PF_KB (340, 0 = off), MKA_ITERS (2), MKA_KD (kernel dir)."""
import math, os, time, torch, ttnn
KD = os.environ.get("MKA_KD", "/work/mk-attn/kernels")
TILE, DH, NU, NQH, T = 32, 256, 4, 6, 4
CTX = int(os.environ.get("MKA_CTX", "32768")); NB = CTX // TILE
PF_KB = float(os.environ.get("MKA_PF_KB", "340")); ITERS = int(os.environ.get("MKA_ITERS", "2"))
dev = ttnn.open_device(device_id=0)
g = dev.compute_with_storage_grid_size(); GX, GY = g.x, g.y; NC = GX * GY
cores = [ttnn.CoreCoord(i % GX, i // GX) for i in range(NC)]
grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
phys = [dev.worker_core_from_logical_core(c) for c in cores]
px = [dev.worker_core_from_logical_core(ttnn.CoreCoord(i, 0)).x for i in range(GX)]
split = next((i for i in range(1, GX) if px[i] != px[i - 1] + 1), GX)
def prect(lx0, lx1):
    if lx1 < lx0: return [0, 0, 0, 0, 0]
    a_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx0, 0)); b_ = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx1, GY - 1))
    return [a_.x, a_.y, b_.x, b_.y, (lx1 - lx0 + 1) * GY]
RECTS = prect(0, split - 1) + prect(split, GX - 1)
# groups: core i (row-major) -> own user g = i * NU // NC; rank within group
grp = [i * NU // NC for i in range(NC)]
members = [[i for i in range(NC) if grp[i] == u] for u in range(NU)]
rank = {i: members[grp[i]].index(i) for i in range(NC)}
FAN = 6
print(f"[mka1] grid {GX}x{GY} split {split} groups {[len(m) for m in members]} ctx {CTX} NB {NB} pf {PF_KB}KB/stream core", flush=True)
torch.manual_seed(0)
q_t = torch.zeros(NU, TILE, DH)
q_t[:, :NQH * T] = torch.randn(NU, NQH * T, DH) / math.sqrt(DH)
k_t = torch.randn(NU, CTX, DH); v_t = torch.randn(NU, CTX, DH)
# KV pages: row-tile ((u * NB + b) * 2 + kv), 8 col tiles each
PLACE = 1 if os.environ.get("MKA_PLACE", "il") == "bank" else 0
KVDT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get("MKA_KVDT", "bfp8")]   # §8: BFP8 default, BFP4 optional
NBANK = dev.dram_grid_size().x
blk = torch.stack([k_t.reshape(NU, NB, TILE, DH), v_t.reshape(NU, NB, TILE, DH)], dim=2)   # [u, b, kv, 32, 256]
BLOC = [(c.x, c.y) for c in dev.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_1)]
core_bank = {}
for u in range(NU):   # each group: cores spread evenly over the banks, nearest first
    m = members[u]; caps = [len(m) // NBANK + (1 if (b - u * (len(m) % NBANK)) % NBANK < len(m) % NBANK else 0) for b in range(NBANK)]   # rotate the remainder so banks end up balanced
    pairs = sorted((abs(cores[i].x - bx) + abs(cores[i].y - by), i, b) for i in m for b, (bx, by) in enumerate(BLOC))
    for d, i, b in pairs:
        if i in core_bank or caps[b] == 0: continue
        core_bank[i] = b; caps[b] -= 1
print(f"[mka1] place {'bank-local' if PLACE else 'interleaved'} banks {NBANK} bank workers {BLOC}", flush=True)
# every core: equal contiguous block range; its blocks live in its bank, slots counted per bank
rng, slotb = {}, {}
cnt = [0] * NBANK
# MKA_WEIGHT=col: share of the user's blocks ∝ 1 / (measured interleaved KV-read time of the core's NoC column), from the
# 32K silicon map (P150, NOC1 reads): the read rate falls ~3x from the left to the right columns
COLT = {1: 58, 2: 58, 3: 67, 4: 90, 5: 110, 6: 119, 10: 116, 11: 140, 12: 157, 13: 168, 14: 179, 15: 181}
WMODE = os.environ.get("MKA_WEIGHT", "none")
for u in range(NU):
    m = members[u]
    if WMODE == "col": w = [(1.0 / COLT.get(phys[i].x, 120)) ** float(os.environ.get("MKA_WEXP", "1")) for i in m]
    elif WMODE == "role": w = [1.0 if (cores[i].x < 10 and cores[i].y < 8) else float(os.environ.get("MKA_WNS", "2")) for i in m]   # non-STREAM cores take more KV
    else: w = [1.0 for i in m]
    cum = [0.0]
    for x in w: cum.append(cum[-1] + x)
    edges = [round(NB * c / cum[-1]) for c in cum]
    for k, i in enumerate(m):
        b0_, b1_ = edges[k], edges[k + 1]
        if b1_ <= b0_: b1_ = b0_ + 1 if b0_ < NB else b0_
        rng[i] = (b0_, b1_)
for i in range(NC):
    b0_, b1_ = rng[i]; slotb[i] = cnt[core_bank[i]]; cnt[core_bank[i]] += b1_ - b0_
blk_bank = torch.zeros(NU, NB, dtype=torch.long); blk_slot = torch.zeros(NU, NB, dtype=torch.long)
for i in range(NC):
    b0_, b1_ = rng[i]
    blk_bank[grp[i], b0_:b1_] = core_bank[i]; blk_slot[grp[i], b0_:b1_] = slotb[i] + torch.arange(b1_ - b0_)
bank = blk_bank.reshape(-1); slot0 = blk_slot.reshape(-1) * 16
if PLACE:
    tiles = blk.reshape(NU, NB, 2, TILE, DH // TILE, TILE).permute(0, 1, 2, 4, 3, 5).reshape(NU * NB, 2 * (DH // TILE), TILE, TILE)
    pages = torch.zeros(max(cnt) * 16 * NBANK, TILE, TILE)
    for j in range(16):
        pages[(slot0 + j) * NBANK + bank] = tiles[:, j]
    kv_rows = pages
else:
    kv_rows = blk.reshape(-1, DH)
t0 = time.time()
if PLACE:
    kv = ttnn.from_torch(kv_rows.reshape(1, 1, -1, TILE).bfloat16(), dtype=KVDT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pq = ttnn.to_torch(kv).float().reshape(-1, TILE, TILE)
    tq = torch.zeros(NU * NB, 16, TILE, TILE)
    for j in range(16): tq[:, j] = pq[(slot0 + j) * NBANK + bank]
    kvq = tq.reshape(NU, NB, 2, DH // TILE, TILE, TILE).permute(0, 1, 2, 4, 3, 5).reshape(NU, NB, 2, TILE, DH)
else:
    kv = ttnn.from_torch(kv_rows.reshape(1, 1, -1, DH).bfloat16(), dtype=KVDT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kvq = ttnn.to_torch(kv).float().reshape(NU, NB, 2, TILE, DH)
kq, vq = kvq[:, :, 0].reshape(NU, CTX, DH), kvq[:, :, 1].reshape(NU, CTX, DH)
del kv_rows
q = ttnn.from_torch(q_t.reshape(1, 1, NU * TILE, DH).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
qq = ttnn.to_torch(q).float().reshape(NU, TILE, DH)
gold = torch.zeros(NU, TILE, DH)
for u in range(NU):
    p = torch.exp(qq[u] @ kq[u].T)
    gold[u] = (p @ vq[u]) / p.sum(-1, keepdim=True)
print(f"[mka1] tensors + golden {time.time()-t0:.1f}s, KV {kv.buffer().size()/1e6 if hasattr(kv,'buffer') else NU*NB*16*576/1e6:.1f}MB", flush=True)
KVTB = ttnn.Tile([TILE, TILE]).get_tile_size(KVDT); PFTB = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat4_b); QTB = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat16)
OTB = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.float32)
is_stream = [1 if (c.x < 10 and c.y < 8) else 0 for c in cores]
NSTREAM = sum(is_stream)
PFT = int(PF_KB * 1024 // PFTB)
pf = ttnn.from_torch(torch.zeros(1, 1, max(1, NSTREAM * PFT) * TILE, TILE).bfloat16(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
out = ttnn.from_torch(torch.zeros(1, 1, NU * TILE, DH).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
def cbd(idx, dt, pages):
    page = ttnn.Tile([TILE, TILE]).get_tile_size(dt)
    return ttnn.CBDescriptor(total_size=pages * page, core_ranges=grid, format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)])
NBB = int(os.environ.get("MKA_NBB", "4")); DEPTH = 2 * NBB; VNOC0 = int(os.environ.get("MKA_VNOC0", "1")); RSLOT = int(os.environ.get("MKA_RSLOT", "2"))
cbs = [cbd(9, ttnn.bfloat16, 32), cbd(10, KVDT, 8 * DEPTH), cbd(11, KVDT, 8 * DEPTH), cbd(12, ttnn.float32, 9),
       cbd(24, ttnn.bfloat16, 1), cbd(25, ttnn.bfloat16, 2 * NBB), cbd(26, ttnn.float32, 9 * RSLOT), cbd(27, ttnn.float32, 1),
       cbd(28, ttnn.bfloat16, 8), cbd(29, ttnn.bfloat4_b, 32)]
l1_kb = (32 * QTB + 2 * 8 * DEPTH * KVTB + 9 * OTB + QTB + 2 * NBB * QTB + 9 * RSLOT * OTB + OTB + 8 * QTB + 32 * PFTB) / 1024
print(f"[mka1] KV {KVDT} NBB {NBB} rslot {RSLOT} vnoc0 {VNOC0}: ATTN working set {l1_kb:.0f} KB/core (budget 300)", flush=True)
sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=0) for i in range(12, 16)]
rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
si = 0
for i, c in enumerate(cores):
    u, r, n = grp[i], rank[i], len(members[grp[i]])
    b0, b1 = rng[i]; beta = core_bank[i]
    m = members[u]
    lvl0_parent = r - r % FAN
    if r == 0: has_p, par, plvl = 0, 0, 0
    elif r % FAN != 0: has_p, par, plvl = 1, m[lvl0_parent], 0
    else: has_p, par, plvl = 1, m[0], 1
    ch0 = [m[k] for k in range(r + 1, min(r + FAN, n))] if r % FAN == 0 else []
    ch1 = [m[k] for k in range(FAN, n, FAN)] if r == 0 else []
    nrecv = len(ch0) + len(ch1)
    def xy(lst):
        o = []
        for k in range(FAN - 1):
            if k < len(lst): o += [phys[lst[k]].x, phys[lst[k]].y]
            else: o += [0, 0]
        return o
    self_rect = 0 if c.x < split else 1
    pp = phys[par]
    tree = [has_p, pp.x, pp.y, plvl, len(ch0)] + xy(ch0) + [len(ch1)] + xy(ch1) + [1 if r == 0 else 0, out.buffer_address(), u]
    rrt[c.x][c.y] = [kv.buffer_address(), u, b0, b1, NB, beta, slotb[i]] + tree
    wrt[c.x][c.y] = [1 if i == 0 else 0, q.buffer_address()] + RECTS + [self_rect, is_stream[i], pf.buffer_address(), si * PFT, PFT if is_stream[i] else 0,
                     has_p, pp.x, pp.y, plvl, len(ch0)] + xy(ch0) + [len(ch1)] + xy(ch1) + [1 if r == 0 else 0, out.buffer_address(), u, kv.buffer_address(), b0, b1, NB]
    crt[c.x][c.y] = [u, b1 - b0, nrecv, 1 if r == 0 else 0]
    si += is_stream[i]
acc_kv = ttnn.TensorAccessorArgs(kv).get_compile_time_args()
acc_q = ttnn.TensorAccessorArgs(q).get_compile_time_args(); acc_pf = ttnn.TensorAccessorArgs(pf).get_compile_time_args(); acc_o = ttnn.TensorAccessorArgs(out).get_compile_time_args()
ks = [ttnn.KernelDescriptor(kernel_source=f"{KD}/mka1_reader.cpp", core_ranges=grid, compile_time_args=[KVTB, OTB, QTB, NBB, PLACE, NBANK, VNOC0] + acc_kv + acc_o, runtime_args=rrt,
                            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
      ttnn.KernelDescriptor(kernel_source=f"{KD}/mka1_writer.cpp", core_ranges=grid, compile_time_args=[QTB, PFTB, OTB, QTB, VNOC0, NBB, KVTB] + acc_q + acc_pf + acc_o + acc_kv,
                            runtime_args=wrt, config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
      ttnn.KernelDescriptor(kernel_source=f"{KD}/mka1_compute.cpp", core_ranges=grid, compile_time_args=[NBB], runtime_args=crt,
                            config=ttnn.ComputeConfigDescriptor(math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("MKA_FID", "HiFi2")), fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False))]
prog = ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=cbs)
for it in range(ITERS):
    t1 = time.time()
    y = ttnn.generic_op([kv, q, pf, out], prog)
    ttnn.synchronize_device(dev)
    v = ttnn.to_torch(out).float().reshape(NU, TILE, DH)[:, :NQH * T]
    gg = gold[:, :NQH * T]
    pcc = torch.corrcoef(torch.stack([v.flatten(), gg.flatten()]))[0, 1].item()
    print(f"[mka1] iter {it} wall {time.time()-t1:.1f}s pcc {pcc:.5f} max|err| {(v-gg).abs().max().item():.4f} |gold|max {gg.abs().max().item():.3f}", flush=True)
ttnn.ReadDeviceProfiler(dev)
ttnn.close_device(dev)
print("[mka1] closed", flush=True)
