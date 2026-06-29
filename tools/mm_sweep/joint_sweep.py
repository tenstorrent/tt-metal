"""
JOINT (S, Pk, blocking) sweep for minimal_matmul, single device. Finds the true per-shape optimum by
sweeping the cross-product of grid partition (S=num_slices, Pk=num_k_slices) and pruned block sizes --
because slicing and blocking INTERACT (the block optimum depends on the per-core dims that (S,Pk) sets).

Grid-generic: queries the device's actual compute grid (so it adapts to Blackhole 11x10, harvesting, or
WH 8x8) and derives PEAK from it. (S,Pk) are enumerated from the divisors of grid.y -- so it works on
non-pow2 grids (BH grid.y=10 -> S*Pk in {1,2,5,10}). Block candidates use the validated OOM-safe pruned
generator (no per-core-divisibility constraint; max-DST subblock; L1 footprint cap; capped at the
smallest value covering per-core; K-divisors -> no K-padding).

Isolates the WH-tuned NoC/DRAM levers (TT_MM_NO_LARGE_LEVERS=1 by default) so the result measures
partition+blocking, not WH-fit heuristics. Resumable: writes one entry per shape to the output JSON and
skips shapes already present. Per-shape it opens/closes the device (the profiler CSV only flushes at
close) so partial results survive a kill.

USAGE:
    MM_CLOCK_HZ=1.35e9 TT_METAL_DEVICE_PROFILER=1 python joint_sweep.py [shapes.json] [out.json] [flux|ltx|all]
  shapes.json (optional) = [[M,K,N], ...]; otherwise the built-in FLUX/LTX lists (pick subset via arg 3).
"""
import os, sys, math, statistics, json, torch, ttnn

CLOCK = float(os.environ.get("MM_CLOCK_HZ", 1.35e9))  # BH 1.35GHz; set 1.0e9 for WH
NO_LEVERS = os.environ.get("MM_NO_LARGE_LEVERS", "1") == "1"
WARMUP, REPS = 0, 3  # 1+WARMUP discarded (compile+cache), REPS measured -> 4 calls/config, median of 3
CHUNK = 1 + WARMUP + REPS
RAW = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/profile_log_device.csv")


# ---------------- (S,Pk) heuristic (for REPORTING heuristic-vs-oracle; mirror of sp_heur_backtest) ----------------
def largest_divisor_leq(n, c):
    if n == 0:
        return 1
    c = max(1, min(c, n))
    for d in range(c, 1, -1):
        if n % d == 0:
            return d
    return 1


def pick_S_Pk(Mt, Nt, Kt, GY, GX):
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
    return largest_divisor_leq(GY // max(1, Pk), lvl(Slvl)), Pk


# ---------------- pruned block-candidate generator (validated; same rules as block_sweep.py) ----------------
M_CAND = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
N_CAND = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 48, 64, 96, 128]
K_CAND = [4, 8, 16]
IN_T = OUT_T = 2048
INTERM_T = 4096
L1_CB_BUDGET = 1310720  # VERIFY BH L1 budget (1.25MiB safe on WH)


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


def gen_blocks(pcM, pcN, ktpb):
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


def feasible_spk(Kt, GY):
    divs = [d for d in range(1, GY + 1) if GY % d == 0]
    out = set()
    for S in divs:
        for Pk in divs:
            if GY % (S * Pk) == 0 and (Pk == 1 or (Kt % Pk == 0 and Kt // Pk >= 2)):
                out.add((S, Pk))
    return sorted(out)


def percore(Mt, Nt, S, Pk, GX, GY, transpose):
    x, y = S * GX, GY // (S * Pk)
    return math.ceil(Mt / (x if transpose else y)), math.ceil(Nt / (y if transpose else x))


def durs():
    L = open(RAW).read().splitlines()
    h = [x.strip() for x in L[1].split(",")]
    ix = {k: i for i, k in enumerate(h)}
    zi, ti, ri, ty = ix["zone name"], ix["time[cycles since reset]"], ix["run host ID"], ix["type"]
    st, en = {}, {}
    for ln in L[2:]:
        f = ln.split(",")
        if len(f) <= zi or not f[zi].strip().endswith("-FW"):
            continue
        rid, t = f[ri].strip(), f[ti].strip()
        if not rid or not t:
            continue
        t = int(t)
        if f[ty].strip() == "ZONE_START":
            st[rid] = min(st.get(rid, t), t)
        elif f[ty].strip() == "ZONE_END":
            en[rid] = max(en.get(rid, t), t)
    return [en[r] - st[r] for r in sorted(st, key=int) if r in en]


def clear_spk():
    for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED"):
        os.environ.pop(k, None)


def sweep_shape(M, K, N):
    Mt, Nt, Kt = M // 32, N // 32, K // 32
    if os.path.exists(RAW):
        os.remove(RAW)
    d = ttnn.open_device(device_id=0)
    d.enable_program_cache()
    gs = d.compute_with_storage_grid_size()
    GX, GY = gs.x, gs.y  # actual device grid (BH 11x10)
    PEAK = GX * GY * 2048 * CLOCK
    cc = ttnn.init_device_compute_kernel_config(
        d.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    ta = torch.randn(M, K, dtype=torch.bfloat16)
    tb = torch.randn(K, N, dtype=torch.bfloat16)
    ref = ta.float() @ tb.float()
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, device=d, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, device=d, layout=ttnn.TILE_LAYOUT)
    if NO_LEVERS:
        os.environ["TT_MM_NO_LARGE_LEVERS"] = "1"
    man = []  # (tag, ok, pcc)

    rv = ref.flatten()
    rv = rv - rv.mean()
    rvn = rv.norm()

    def pcc(t):
        ov = t.flatten().float()[: rv.numel()]
        ov = ov - ov.mean()
        return float(torch.dot(ov, rv) / (ov.norm() * rvn + 1e-12))

    def run(cfg, tag):
        # n_exec = device dispatches that ran (each minimal_matmul call == exactly one profiled
        # program == one entry in durs()). Tracked so the duration parser can advance past configs
        # that EXECUTED but didn't qualify, keeping later configs aligned (the profiler CSV flushes
        # only at device close, so per-config reads aren't possible). Incremented AFTER each call.
        #
        # Correctness vs timing are measured on DIFFERENT runs on purpose:
        #  - fresh_pcc = PCC of the first (fresh-compile) call. This is the real correctness of the
        #    config's math. The fused split-K (Pk>1) BH program-cache bug zeros the output on CACHED
        #    replays only -- the fresh run is correct -- so fresh_pcc lets us KEEP Pk>1 configs in the
        #    ranking while still rejecting genuinely-wrong blocks (e.g. silent subblock corruption).
        #  - p (cache_pcc) = PCC of the last cached replay; <0.99 flags the cache bug (informational).
        #  - timing comes from the cached replays (production path). For Pk>1 those replays currently
        #    output zeros; the bug is data-only (same kernel/cycles), so the timing is representative.
        d.clear_program_cache()
        ok = True
        p = 0.0
        fresh = 0.0
        n_exec = 0
        try:
            ot = None
            for j in range(1 + WARMUP):
                o = (
                    ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg)
                    if cfg
                    else ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc)
                )
                n_exec += 1
                if j == 0:
                    fresh = pcc(ttnn.to_torch(o))  # correctness from the fresh-compile run
                o.deallocate()
            ttnn.synchronize_device(d)
            for _ in range(REPS):
                o = (
                    ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg)
                    if cfg
                    else ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc)
                )
                n_exec += 1
                ot = ttnn.to_torch(o)
                o.deallocate()
            ttnn.synchronize_device(d)
            ttnn.ReadDeviceProfiler(d)
            p = pcc(ot)  # cached-replay PCC (0 for the fused-K cache bug)
            ok = fresh >= 0.99  # qualify on FRESH correctness, not cached replay
        except Exception:
            ok = False
        man.append((tag, ok, p, fresh, n_exec))

    # AUTO baseline (full auto path: heuristic S/Pk + auto-block), lever-free
    clear_spk()
    run(None, "AUTO")
    # joint sweep: all feasible (S,Pk) x pruned blocks
    transpose = M > N
    for S, Pk in feasible_spk(Kt, GY):
        clear_spk()
        os.environ["TT_MM_NUM_SLICES"] = str(S)
        if Pk > 1:
            os.environ["TT_MM_K_SLICES"] = str(Pk)
            os.environ["TT_MM_K_FUSED"] = "1"
        pcM, pcN = percore(Mt, Nt, S, Pk, GX, GY, transpose)
        for mb, kb, nb, sbh, sbw in gen_blocks(pcM, pcN, Kt // Pk):
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sbh,
                subblock_w=sbw,
                compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY),
            )
            run(cfg, (S, Pk, mb, kb, nb, sbh, sbw))
    a.deallocate()
    b.deallocate()
    ttnn.close_device(d)
    # parse durations: each config emitted n_exec durations in run-order; advance the index by
    # n_exec for EVERY config (incl. PCC-failed ones that ran) so survivors stay aligned. Only
    # configs that passed (ok) and produced a full run (n_exec == CHUNK durations present) are kept.
    ds = durs()
    i = 0
    recs = []  # (tag, us, fresh_pcc, cache_pcc)
    n_pcc_fail = n_exc = n_cache_bug = 0
    util = lambda us: 100 * 2 * M * K * N / (PEAK * us * 1e-6) if us else None
    for tag, ok, p, fresh, n_exec in man:
        seg = ds[i : i + n_exec]
        i += n_exec
        if ok and n_exec == CHUNK and len(seg) == CHUNK:
            us = statistics.median(seg[-REPS:]) / 1000
            recs.append((tag, us, fresh, p))
            if p < 0.99:  # fresh-correct but cached replay wrong == the fused-K cache bug
                n_cache_bug += 1
        elif not ok:
            # distinguish fresh-correctness fail (full dispatch) from never-ran (exception)
            if n_exec == CHUNK:
                n_pcc_fail += 1
            else:
                n_exc += 1
    auto = next((r for r in recs if r[0] == "AUTO"), None)
    blocks = sorted([r for r in recs if r[0] != "AUTO"], key=lambda r: r[1])
    hS, hPk = pick_S_Pk(Mt, Nt, Kt, GY, GX)
    best = blocks[0] if blocks else None
    return {
        "shape": [M, K, N],
        "MtNtKt": [Mt, Nt, Kt],
        "grid": [GX, GY],
        "peak_tflops": PEAK / 1e12,
        "heuristic_SPk": [hS, hPk],
        "auto": {"us": auto[1], "util": util(auto[1]), "pcc": auto[2], "cache_pcc": auto[3]} if auto else None,
        "best": {
            "S": best[0][0],
            "Pk": best[0][1],
            "mb": best[0][2],
            "kb": best[0][3],
            "nb": best[0][4],
            "sbh": best[0][5],
            "sbw": best[0][6],
            "us": best[1],
            "util": util(best[1]),
            "pcc": best[2],  # fresh-compile correctness
            "cache_pcc": best[3],  # cached-replay PCC; <0.99 == fused-K cache bug (timing still valid)
        }
        if best
        else None,
        "best_vs_auto": (auto[1] / best[1]) if (auto and best) else None,
        "n_configs": len(blocks),
        "n_pcc_fail": n_pcc_fail,  # ran but FRESH PCC<0.99 (genuinely wrong math) -- excluded
        "n_exception": n_exc,  # never dispatched (validation/compile/OOM)
        "n_cache_bug": n_cache_bug,  # included configs whose cached replay is wrong (fused-K bug)
        "all": [
            {
                "S": t[0],
                "Pk": t[1],
                "mb": t[2],
                "kb": t[3],
                "nb": t[4],
                "sbh": t[5],
                "sbw": t[6],
                "us": us,
                "util": util(us),
                "pcc": fresh,
                "cache_pcc": cp,
            }
            for (t, us, fresh, cp) in blocks
        ],
    }


# ---------------- FLUX / LTX shape lists ----------------
FLUX = [
    [32, 6144, 3072],
    [32, 6144, 1536],
    [32, 6144, 6144],
    [32, 6144, 2304],
    [32, 6144, 9216],
    [32, 6144, 4608],
    [64, 6144, 9216],
    [64, 6144, 1536],
    [64, 4608, 6144],
    [64, 15360, 1536],
    [64, 6144, 4608],
    [32, 256, 6144],
    [8192, 6144, 128],
    [4096, 6144, 128],
    [128, 6144, 4608],
    [128, 2304, 6144],
    [2048, 6144, 128],
    [128, 15360, 768],
    [128, 6144, 2304],
    [1024, 6144, 128],
    [128, 6144, 768],
    [512, 6144, 128],
    [512, 6144, 1536],
    [512, 4608, 6144],
    [512, 6144, 9216],
    [512, 6144, 4608],
    [576, 6144, 9216],
    [2048, 128, 1536],
    [16512, 6144, 4608],
    [576, 6144, 6144],
    [4096, 128, 768],
    [4096, 6144, 768],
    [16384, 2304, 6144],
    [16384, 6144, 768],
    [2112, 6144, 9216],
    [16384, 6144, 4608],
    [8192, 6144, 1536],
    [1024, 6144, 768],
    [8192, 6144, 4608],
    [8256, 6144, 9216],
    [16512, 3072, 6144],
    [2048, 6144, 9216],
    [8192, 6144, 9216],
    [8192, 4608, 6144],
    [4096, 6144, 4608],
    [2048, 6144, 1536],
    [16384, 128, 768],
    [1024, 6144, 2304],
    [16384, 6144, 2304],
    [4096, 2304, 6144],
    [2048, 6144, 4608],
    [8192, 128, 1536],
    [2048, 4608, 6144],
    [4096, 6144, 2304],
    [1024, 128, 768],
    [4224, 3072, 6144],
    [8256, 6144, 6144],
    [2112, 6144, 6144],
    [1152, 3072, 6144],
    [512, 128, 1536],
    [4224, 6144, 4608],
    [1024, 6144, 4608],
    [1152, 6144, 4608],
    [1024, 2304, 6144],
    [16384, 6144, 128],
]
LTX = [
    [4864, 4096, 32],
    [32, 2048, 2048],
    [1216, 4096, 32],
    [32, 2048, 1536],
    [32, 2048, 512],
    [32, 2048, 32],
    [4864, 4096, 512],
    [256, 2048, 1024],
    [4864, 4096, 1024],
    [4864, 4096, 4096],
    [1216, 4096, 512],
    [4864, 2048, 1024],
    [4864, 4096, 3072],
    [1216, 4096, 3072],
    [1216, 4096, 1024],
    [1216, 2048, 1024],
    [1216, 4096, 4096],
]

if __name__ == "__main__":
    sel = sys.argv[3] if len(sys.argv) > 3 else "all"
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        SHAPES = json.load(open(sys.argv[1]))
    else:
        SHAPES = (FLUX if sel in ("flux", "all") else []) + (LTX if sel in ("ltx", "all") else [])
    OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.getcwd(), "bh_joint_sweep.json")
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(
        f"clock={CLOCK/1e9:.2f}GHz  no_levers={NO_LEVERS}  {len(SHAPES)} shapes, {len(done)} already done -> {OUT}",
        flush=True,
    )
    for M, K, N in SHAPES:
        key = f"{M}x{K}x{N}"
        if key in done:
            continue
        try:
            r = sweep_shape(M, K, N)
            done[key] = r
            json.dump(done, open(OUT, "w"), indent=0)
            b = r["best"]
            a = r["auto"]
            auto_s = (f"AUTO {a['util']:.1f}%" + ("*" if a["cache_pcc"] < 0.99 else "")) if a else "AUTO FAIL"
            best_s = (
                f"BEST S{b['S']}Pk{b['Pk']} {b['mb']}/{b['kb']}/{b['nb']} {b['util']:.1f}%"
                + ("*" if b["cache_pcc"] < 0.99 else "")
                + (f" ({r['best_vs_auto']:.2f}x)" if r["best_vs_auto"] else "")
                if b
                else "BEST none"
            )
            print(
                f"{key:<18} grid{r['grid']} heurSPk{r['heuristic_SPk']} | {auto_s} | {best_s} | "
                f"{r['n_configs']} ok ({r['n_cache_bug']} cachebug*), {r['n_pcc_fail']} wrong, {r['n_exception']} exc",
                flush=True,
            )
        except Exception as e:
            print(f"{key}: FAILED {str(e)[:100]}", flush=True)
    print("DONE", flush=True)
