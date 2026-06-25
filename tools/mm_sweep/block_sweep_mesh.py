"""
Parallel block sweep across N submeshes (e.g. 32 chips on a Wormhole Galaxy).

WHY THIS SHAPE: minimal_matmul's (S,Pk) partition is chosen from PROCESS-GLOBAL env vars
(TT_MM_NUM_SLICES / TT_MM_K_SLICES), read by the factory at program-build time. 32 threads running
shapes that need different (S,Pk) would clobber each other's env. So we PHASE the work by (S,Pk): within
a phase the env is constant, so the 32 worker threads are safe; setenv happens only single-threaded
between phases. (For a suite that's mostly (1,1) this is ~one big phase = full 32x.)

!!! UNTESTED on mesh hardware (author ran on a single 8x8 chip). The mesh/submesh API calls, the
    per-submesh device-id lookup, and the profiler-under-concurrency behaviour are marked VERIFY:
    the Galaxy instance must confirm/adjust them. A wall-clock fallback (USE_PROFILER=0) is provided.

USAGE (on the Galaxy box):
    TT_METAL_DEVICE_PROFILER=1 python block_sweep_mesh.py shapes.json out.json [num_submesh]
  where shapes.json = [[M,K,N], ...].  Falls back to the built-in 5 smoke shapes if no file given.
"""
import os, sys, math, json, time, threading, queue, torch, ttnn


# ---------------- (S,Pk) heuristic (mirror of sp_heur_final.pick_S_Pk) ----------------
def largest_divisor_leq(n, cap):
    if n == 0:
        return 1
    cap = max(1, min(cap, n))
    for d in range(cap, 1, -1):
        if n % d == 0:
            return d
    return 1


def pick_S_Pk(Mt, Nt, Kt, GY=8, GX=8):
    small, big = min(Mt, Nt), max(Mt, Nt)
    out = Mt * Nt
    cores = GX * GY
    skew = big / small if small else 1.0
    Slvl = Pklvl = 1
    if Kt <= 4:
        Slvl = 8 if (skew >= 6 and small < GY) else 1
    elif Kt >= 64:
        if out < cores // 4:
            Pklvl = 8
        elif out <= cores:
            Pklvl = 8 if skew >= 2.5 else (1 if small >= GY else 4)
        elif out < 4 * cores:
            Pklvl = 4 if skew >= 6 else 2
        else:
            Pklvl = 2 if out < 8 * cores else 1
        if skew >= 12 and small < GY:
            Slvl, Pklvl = 2, min(Pklvl, 4)
    else:
        if out >= 4 * cores:
            Slvl = 8 if (skew >= 24 and small < GY) else (2 if skew >= 2.5 else 1)
        elif out >= cores:
            Slvl, Pklvl = (8, 1) if skew >= 24 else (4, 2) if skew >= 12 else (2, 1) if skew >= 6 else (1, 1)
        elif out >= cores // 4:
            Slvl, Pklvl = (4, 2) if skew >= 12 else (2, 4) if skew >= 6 else (2, 1) if skew >= 2.5 else (1, 1)
        else:
            Slvl, Pklvl = (4, 2) if skew >= 6 else (1, 4)

    def lvl(l):
        if l >= 8:
            return GY
        if l >= 4:
            return largest_divisor_leq(GY, (GY + 1) // 2)
        if l >= 2:
            return largest_divisor_leq(GY, (GY + 3) // 4)
        return 1

    Pk = lvl(Pklvl)
    while Pk > 1 and (Kt % Pk != 0 or Kt // Pk < 2):
        Pk = largest_divisor_leq(GY, Pk - 1)
    S = largest_divisor_leq(GY // max(1, Pk), lvl(Slvl))
    return S, Pk


# ---------------- block-candidate generator (same prune rules as block_sweep.py) ----------------
M_CAND = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
N_CAND = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 48, 64, 96, 128]
K_CAND = [4, 8, 16]
IN_T = OUT_T = 2048
INTERM_T = 4096
L1_CB_BUDGET = 1310720


def footprint(mb, kb, nb):
    return mb * kb * 2 * IN_T + kb * nb * 2 * IN_T + mb * nb * 2 * OUT_T + mb * nb * INTERM_T


def cap(c, pc):
    ge = [x for x in c if x >= pc]
    hi = min(ge) if ge else max(c)
    return [x for x in c if x <= hi]


def mn_cands(cands, pc):
    cs = cap(cands, pc)
    return [1] if pc <= 1 else ([m for m in cs if m >= 2] or [1])


def k_cands(ktpb):
    cs = cap(K_CAND, ktpb)
    return [k for k in cs if ktpb % k == 0] or [min(cs)]


def adsub(mb, nb):
    def lg(v, c):
        for d in (4, 2, 1):
            if d <= c and v % d == 0:
                return d
        return 1

    return (lg(mb, 4 // lg(nb, 4)), lg(nb, 4)) if nb >= mb else (lg(mb, 4), lg(nb, 4 // lg(mb, 4)))


def gen_configs(pcM, pcN, ktpb):
    out = []
    for mb in mn_cands(M_CAND, pcM):
        for nb in mn_cands(N_CAND, pcN):
            sbh, sbw = adsub(mb, nb)
            if sbh * sbw != 4 and mb * nb >= 4:
                continue
            for kb in k_cands(ktpb):
                if footprint(mb, kb, nb) <= L1_CB_BUDGET:
                    out.append((mb, kb, nb, sbh, sbw))
    return out


# ---------------- build work list, grouped by (S,Pk) phase ----------------
def percore(M, N, S, Pk, GX=8, GY=8):
    Mt, Nt = M // 32, N // 32
    x, y = S * GX, GY // (S * Pk)
    transpose = M > N
    return (math.ceil(Mt / (x if transpose else y)), math.ceil(Nt / (y if transpose else x)))


def build_phases(shapes, GX=8, GY=8):
    phases = {}  # (S,Pk) -> list of work items
    for M, K, N in shapes:
        Mt, Nt, Kt = M // 32, N // 32, K // 32
        S, Pk = pick_S_Pk(Mt, Nt, Kt, GY, GX)
        pcM, pcN = percore(M, N, S, Pk, GX, GY)
        ktpb = Kt // max(1, Pk)
        for mb, kb, nb, sbh, sbw in gen_configs(pcM, pcN, ktpb):
            phases.setdefault((S, Pk), []).append((M, K, N, mb, kb, nb, sbh, sbw))
    return phases


# ---------------- mesh setup ----------------  (VERIFY all of this on the Galaxy)
NUM_SUBMESH = int(sys.argv[3]) if len(sys.argv) > 3 else 32
USE_PROFILER = os.environ.get("USE_PROFILER", "1") == "1"
WARMUP, REPS = 2, 4
PEAK = 2048 * 64 * 1e9

# VERIFY: open a mesh spanning the 32 chips. MeshShape may need to match the physical galaxy (e.g. 8x4).
mesh = ttnn.open_mesh_device(ttnn.MeshShape(NUM_SUBMESH, 1))  # VERIFY shape/topology
# VERIFY: carve NUM_SUBMESH independent 1x1 submeshes (single-chip each -> no fabric needed).
submeshes = mesh.create_submeshes(ttnn.MeshShape(1, 1))  # VERIFY returns NUM_SUBMESH of them
assert len(submeshes) >= 1, "no submeshes created"
NUM_SUBMESH = len(submeshes)
print(f"opened mesh, {NUM_SUBMESH} submeshes; profiler={USE_PROFILER}", flush=True)

RAW = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/profile_log_device.csv")
prof_lock = threading.Lock()  # serialize ReadDeviceProfiler + CSV parse (brief; runs stay parallel)
results_lock = threading.Lock()
results = []  # (M,K,N,mb,kb,nb,sbh,sbw,us,pcc)


def parse_device_durations(device_id, already):
    """Return per-op FW-zone durations (ns) for `device_id` beyond the `already`-th, from the CSV.
    VERIFY the CSV has a device-id column on the galaxy build (header name may differ)."""
    try:
        L = open(RAW).read().splitlines()
        h = [x.strip() for x in L[1].split(",")]
        ix = {k: i for i, k in enumerate(h)}
        di = ix.get("device id", ix.get("device_id", ix.get("PCIe slot", None)))
        zi, ti, ri, ty = ix["zone name"], ix["time[cycles since reset]"], ix["run host ID"], ix["type"]
        st, en = {}, {}
        for ln in L[2:]:
            f = ln.split(",")
            if len(f) <= zi or not f[zi].strip().endswith("-FW"):
                continue
            if di is not None and f[di].strip() != str(device_id):
                continue
            rid, t = f[ri].strip(), f[ti].strip()
            if not rid or not t:
                continue
            t = int(t)
            if f[ty].strip() == "ZONE_START":
                st[rid] = min(st.get(rid, t), t)
            elif f[ty].strip() == "ZONE_END":
                en[rid] = max(en.get(rid, t), t)
        durs = [en[r] - st[r] for r in sorted(st, key=int) if r in en]
        return durs[already:]
    except Exception:
        return []


def worker(sm):
    # VERIFY: how to get this submesh's physical device id for profiler attribution.
    try:
        dev_id = sm.get_devices()[0].id()  # VERIFY API
    except Exception:
        dev_id = None
    cc = ttnn.init_device_compute_kernel_config(
        sm.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    consumed = 0
    while True:
        try:
            item = work_q.get_nowait()
        except queue.Empty:
            break
        M, K, N, mb, kb, nb, sbh, sbw = item
        ta = torch.randn(M, K, dtype=torch.bfloat16)
        tb = torch.randn(K, N, dtype=torch.bfloat16)
        a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, device=sm, layout=ttnn.TILE_LAYOUT)
        b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, device=sm, layout=ttnn.TILE_LAYOUT)
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sbh,
            subblock_w=sbw,
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        )
        us, p = None, 0.0
        try:
            t0 = time.time()
            for _ in range(WARMUP + 1):
                out = ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg)
                out.deallocate()
            ttnn.synchronize_device(sm)
            ot = None
            for _ in range(REPS):
                out = ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg)
                ot = ttnn.to_torch(out)
                out.deallocate()
            ttnn.synchronize_device(sm)
            walltime = (time.time() - t0) / (WARMUP + 1 + REPS) * 1e6
            o = ot.flatten().float()
            r = (ta.float() @ tb.float()).flatten()
            o -= o.mean()
            r -= r.mean()
            p = float(torch.dot(o, r) / (o.norm() * r.norm() + 1e-12))
            if USE_PROFILER and dev_id is not None:
                with prof_lock:
                    ttnn.ReadDeviceProfiler(sm)  # VERIFY: per-submesh drain
                    d = parse_device_durations(dev_id, consumed)
                    consumed += len(d)
                us = (sorted(d[-REPS:])[len(d[-REPS:]) // 2] / 1000) if len(d) >= REPS else walltime
            else:
                us = walltime
        except Exception as e:
            p = -1
        with results_lock:
            results.append((M, K, N, mb, kb, nb, sbh, sbw, us, p))
        a.deallocate()
        b.deallocate()


# ---------------- run, phased by (S,Pk) ----------------
SHAPES = (
    json.load(open(sys.argv[1]))
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1])
    else [[4224, 6144, 4608], [2112, 6144, 6144], [1152, 6144, 4608], [4864, 4096, 1024], [1216, 4096, 1024]]
)
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.getcwd(), "bsweep_mesh.json")
phases = build_phases(SHAPES)
print(
    f"{len(SHAPES)} shapes -> {sum(len(v) for v in phases.values())} configs in {len(phases)} (S,Pk) phases", flush=True
)
for (S, Pk), items in phases.items():
    for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED", "TT_MM_NO_AUTO_KPAR"):
        os.environ.pop(k, None)
    os.environ["TT_MM_NUM_SLICES"] = str(S)  # set ONCE, single-threaded, before spawning
    if Pk > 1:
        os.environ["TT_MM_K_SLICES"] = str(Pk)
        os.environ["TT_MM_K_FUSED"] = "1"
    work_q = queue.Queue()
    for it in items:
        work_q.put(it)
    print(f"phase (S={S},Pk={Pk}): {len(items)} configs across {NUM_SUBMESH} submeshes", flush=True)
    ts = [threading.Thread(target=worker, args=(submeshes[i],)) for i in range(NUM_SUBMESH)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
ttnn.close_mesh_device(mesh)  # VERIFY close API


def util(M, K, N, us):
    return 100 * 2 * M * K * N / (PEAK * us * 1e-6) if us else None


json.dump(
    [
        {
            "M": M,
            "K": K,
            "N": N,
            "mb": mb,
            "kb": kb,
            "nb": nb,
            "sbh": sbh,
            "sbw": sbw,
            "us": us,
            "util": util(M, K, N, us),
            "pcc": p,
        }
        for (M, K, N, mb, kb, nb, sbh, sbw, us, p) in results
    ],
    open(OUT, "w"),
    indent=0,
)
ok = sum(1 for r in results if r[-1] > 0.99)
print(f"DONE: {len(results)} configs, {ok} PCC-pass -> {OUT}", flush=True)
