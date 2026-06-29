"""
INCREMENTAL max-parallelism sweep. After the rows_per_group==1 off-by-one fix, the partitions with
S*Pk == grid.y ((10,1),(1,10),(5,2),(2,5)) are newly runnable. This script:
  1. merges the headroom full-resweep (bh_headroom.json) into bh_joint.json (those 14 are already redone),
  2. for every OTHER shape already in bh_joint.json, sweeps ONLY the new S*Pk==grid.y (S,Pk) x blocks
     (skipping any (S,Pk) already present in that shape's 'all'), merges them in, and recomputes best.
No re-sweep of already-swept (S,Pk). Resumable: a shape whose new (S,Pk) are all present is skipped.
   python tools/mm_sweep/incr_maxpar_sweep.py [bh_joint.json] [bh_headroom.json]
"""
import os, sys, statistics, json, torch, ttnn, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("js", os.path.join(HERE, "joint_sweep.py"))
js = importlib.util.module_from_spec(spec)
spec.loader.exec_module(js)

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "bh_joint.json")
HEADROOM_JSON = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "bh_headroom.json")
WARMUP, REPS, CHUNK, CLOCK, RAW = js.WARMUP, js.REPS, js.CHUNK, js.CLOCK, js.RAW
FIELDS = ("S", "Pk", "mb", "kb", "nb", "sbh", "sbw", "us", "util", "pcc", "cache_pcc")


def measure_new(M, K, N, existing_spk):
    Mt, Nt, Kt = M // 32, N // 32, K // 32
    if os.path.exists(RAW):
        os.remove(RAW)
    d = ttnn.open_device(device_id=0)
    d.enable_program_cache()
    gs = d.compute_with_storage_grid_size()
    GX, GY = gs.x, gs.y
    PEAK = GX * GY * 2048 * CLOCK
    targets = [(S, Pk) for (S, Pk) in js.feasible_spk(Kt, GY) if S * Pk == GY and (S, Pk) not in existing_spk]
    if not targets:
        ttnn.close_device(d)
        return [], GX, GY
    cc = ttnn.init_device_compute_kernel_config(
        d.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    ta = torch.randn(M, K, dtype=torch.bfloat16)
    tb = torch.randn(K, N, dtype=torch.bfloat16)
    ref = (ta.float() @ tb.float()).flatten()
    rv = ref - ref.mean()
    rvn = rv.norm()
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, device=d, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, device=d, layout=ttnn.TILE_LAYOUT)
    os.environ["TT_MM_NO_LARGE_LEVERS"] = "1"

    def pcc(t):
        ov = t.flatten().float()[: rv.numel()]
        ov = ov - ov.mean()
        return float(torch.dot(ov, rv) / (ov.norm() * rvn + 1e-12))

    man = []
    transpose = M > N
    for S, Pk in targets:
        js.clear_spk()
        os.environ["TT_MM_NUM_SLICES"] = str(S)
        if Pk > 1:
            os.environ["TT_MM_K_SLICES"] = str(Pk)
            os.environ["TT_MM_K_FUSED"] = "1"
        pcM, pcN = js.percore(Mt, Nt, S, Pk, GX, GY, transpose)
        for mb, kb, nb, sbh, sbw in js.gen_blocks(pcM, pcN, Kt // Pk):
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sbh,
                subblock_w=sbw,
                compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY),
            )
            d.clear_program_cache()
            ok = True
            p = 0.0
            fresh = 0.0
            n_exec = 0
            try:
                ot = None
                for j in range(1 + WARMUP):
                    o = ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg)
                    n_exec += 1
                    if j == 0:
                        fresh = pcc(ttnn.to_torch(o))
                    o.deallocate()
                ttnn.synchronize_device(d)
                for _ in range(REPS):
                    o = ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg)
                    n_exec += 1
                    ot = ttnn.to_torch(o)
                    o.deallocate()
                ttnn.synchronize_device(d)
                ttnn.ReadDeviceProfiler(d)
                p = pcc(ot)
                ok = fresh >= 0.99
            except Exception:
                ok = False
            man.append(((S, Pk, mb, kb, nb, sbh, sbw), ok, p, fresh, n_exec))
    a.deallocate()
    b.deallocate()
    ttnn.close_device(d)
    ds = js.durs()
    i = 0
    recs = []
    util = lambda us: 100 * 2 * M * K * N / (PEAK * us * 1e-6) if us else None
    for tag, ok, p, fresh, n_exec in man:
        seg = ds[i : i + n_exec]
        i += n_exec
        if ok and n_exec == CHUNK and len(seg) == CHUNK:
            us = statistics.median(seg[-REPS:]) / 1000
            recs.append(dict(zip(FIELDS, (*tag, us, util(us), fresh, p))))
    return recs, GX, GY


def main():
    data = json.load(open(OUT))
    # 1) merge headroom full-resweep
    if os.path.exists(HEADROOM_JSON):
        hd = json.load(open(HEADROOM_JSON))
        for k, v in hd.items():
            data[k] = v
        json.dump(data, open(OUT, "w"), indent=0)
        print(f"merged {len(hd)} headroom resweeps into {OUT}", flush=True)
    HEADROOM = set(json.load(open(HEADROOM_JSON)).keys()) if os.path.exists(HEADROOM_JSON) else set()

    # 2) incremental max-par for the other shapes
    for key in list(data.keys()):
        if key in HEADROOM:
            continue
        entry = data[key]
        if not entry.get("all") or not entry.get("best"):
            continue
        M, K, N = entry["shape"]
        GY = entry["grid"][1]
        existing = set((c["S"], c["Pk"]) for c in entry["all"])
        new_targets = [(S, Pk) for (S, Pk) in js.feasible_spk(K // 32, GY) if S * Pk == GY and (S, Pk) not in existing]
        if not new_targets:
            print(f"{key:<18} no new maxpar (S,Pk) -> skip", flush=True)
            continue
        recs, GX, GY = measure_new(M, K, N, existing)
        if not recs:
            print(f"{key:<18} tried {new_targets}, 0 valid (PCC/exc) -> skip", flush=True)
            continue
        old_best = entry["best"]
        entry["all"].extend(recs)
        best = min(entry["all"], key=lambda c: c["us"])
        entry["best"] = {k: best[k] for k in FIELDS}
        auto = entry.get("auto")
        entry["best_vs_auto"] = (auto["us"] / best["us"]) if (auto and best["us"]) else None
        entry["n_configs"] = len(entry["all"])
        json.dump(data, open(OUT, "w"), indent=0)
        maxpar_best = best["S"] * best["Pk"] == GY
        print(
            f"{key:<18} +{len(recs)} maxpar cfgs {new_targets} | "
            f"best {old_best['util']:.1f}% -> {best['util']:.1f}% (S{best['S']}Pk{best['Pk']})"
            f"{'  <-- MAXPAR WINS' if maxpar_best else ''}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
