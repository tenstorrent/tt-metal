"""
Hardened SINGLE-CARD Blackhole two-phase sweep driver (resumable, hang-safe, correct timing).

Profiler fact (verified): ttnn.ReadDeviceProfiler does NOT flush the CSV mid-run; it is only written at
close_device. So timing MUST be parsed after a close. Design:
  * ONE device session per GROUP = (phase, S, Pk, M, K, N). Each invocation runs the next group's
    not-yet-attempted configs, closes the device, then parse_durations -> assign us in run order
    (advance by per-config n_exec so PCC/err configs don't misalign). Exit 10 = progress (more groups),
    0 = all groups done, 2 = hang.
  * Per-config ATTEMPTED marking (flushed before running) -> a hang skips only that config on resume;
    the resumed session re-runs the group's remaining configs and times them correctly.
  * TT_METAL_OPERATION_TIMEOUT_SECONDS (set by the wrapper) makes a hung op RAISE instead of wedging ->
    on a device-affecting error we flush + os._exit(2); the wrapper tt-smi -r's and resumes. We never
    kill -9 mid-op.

Phase 1 (baseline=="optimized main"): levers OFF, S1Pk1, full block sweep. Phase 2 (branch): joint
(S,Pk,blocking), levers ON. Speedup = baseline-best / branch-best per shape.
"""
import os, sys, json, statistics, torch, ttnn
from block_sweep_mp import gen_configs, spk_combos_for, percore, parse_durations

CLOCK = float(os.environ.get("MM_CLOCK_HZ", 1.35e9))
RAW = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/profile_log_device.csv")
WARMUP, REPS = 1, 4
CHUNK = 1 + WARMUP + REPS
CKPT = sys.argv[1] if len(sys.argv) > 1 else "tools/mm_sweep/bh_sweep_ckpt.json"
EXIT_DONE, EXIT_PROGRESS, EXIT_HANG = 0, 10, 2

SHAPES = [
    tuple(int(x) for x in s.split("x"))
    for s in os.environ.get("BH_SHAPES", "32x6144x4608,512x6144x9216,2048x6144x4608").split(",")
]
HANG_MARKERS = (
    "timeout",
    "timed out",
    "hang",
    "no such device",
    "0xffffffff",
    "watcher",
    "fatal",
    "ttdevice",
    "device error",
)


def keyof(it):
    return "|".join(str(x) for x in it)


def gkey(w):  # group key: phase,S,Pk,M,K,N
    return tuple(w[0:6])


def set_baseline_env():
    for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED", "TT_MM_MCAST_PREFETCH", "TT_MM_MCAST_BROADCAST"):
        os.environ.pop(k, None)
    os.environ["TT_MM_NO_LARGE_LEVERS"] = "1"
    os.environ["TT_MM_NO_AUTO_KPAR"] = "1"
    os.environ["TT_MM_NO_AUTO_PREFETCH"] = "1"


def set_branch_spk_env(S, Pk):
    for k in (
        "TT_MM_NUM_SLICES",
        "TT_MM_K_SLICES",
        "TT_MM_K_FUSED",
        "TT_MM_NO_AUTO_KPAR",
        "TT_MM_NO_LARGE_LEVERS",
        "TT_MM_NO_AUTO_PREFETCH",
    ):
        os.environ.pop(k, None)
    os.environ["TT_MM_NUM_SLICES"] = str(S)
    if Pk > 1:
        os.environ["TT_MM_K_SLICES"] = str(Pk)
        os.environ["TT_MM_K_FUSED"] = "1"


def build_work(GX, GY):
    work = []
    for M, K, N in SHAPES:
        Kt = K // 32
        pcM, pcN = percore(M, N, 1, 1, GX, GY)
        for mb, kb, nb, sbh, sbw in gen_configs(pcM, pcN, Kt):
            work.append(["base", 1, 1, M, K, N, mb, kb, nb, sbh, sbw])
        for S, Pk in spk_combos_for(Kt, GY, GX):
            pcM, pcN = percore(M, N, S, Pk, GX, GY)
            for mb, kb, nb, sbh, sbw in gen_configs(pcM, pcN, Kt // max(1, Pk)):
                work.append(["branch", S, Pk, M, K, N, mb, kb, nb, sbh, sbw])
    work.sort(key=lambda w: (w[0], w[1], w[2], w[3], w[4], w[5]))
    return work


def load_ckpt():
    return json.load(open(CKPT)) if os.path.exists(CKPT) else {"results": [], "attempted": []}


def save_ckpt(ck):
    json.dump(ck, open(CKPT + ".tmp", "w"))
    os.replace(CKPT + ".tmp", CKPT)


def report(ck, GX, GY):
    PEAK = GX * GY * 2048 * CLOCK
    recs = ck["results"]
    print("\n==== BH single-card two-phase sweep ====")
    print(
        f"{'shape':>18} {'base us':>9} {'base%':>6} {'branch us':>10} {'br%':>6} " f"{'best(S,Pk)':>11} {'speedup':>8}"
    )
    for M, K, N in SHAPES:
        rs = [r for r in recs if (r["M"], r["K"], r["N"]) == (M, K, N) and r["us"]]
        base = [r for r in rs if r["phase"] == "base"]
        br = [r for r in rs if r["phase"] == "branch"]
        if not base or not br:
            print(f"{M}x{K}x{N:>6}  (base={len(base)} branch={len(br)} timed)")
            continue
        bb = min(base, key=lambda r: r["us"])
        bbr = min(br, key=lambda r: r["us"])
        print(
            f"{M}x{K}x{N:>6} {bb['us']:>9.1f} {bb['util']:>6.1f} {bbr['us']:>10.1f} {bbr['util']:>6.1f} "
            f"{'S%dPk%d' % (bbr['S'], bbr['Pk']):>11} {bb['us'] / bbr['us']:>7.2f}x"
        )
    print(f"# {len(recs)} timed, {sum(1 for r in recs if r.get('hang'))} hangs")


def main():
    ck = load_ckpt()
    attempted = set(ck["attempted"])

    d = ttnn.open_device(device_id=0)
    d.enable_program_cache()
    gs = d.compute_with_storage_grid_size()
    GX, GY = gs.x, gs.y
    PEAK = GX * GY * 2048 * CLOCK
    cc = ttnn.init_device_compute_kernel_config(
        d.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    work = build_work(GX, GY)
    # next group (in work order) that has >=1 unattempted config
    groups = []
    seen = set()
    for w in work:
        g = gkey(w)
        if g not in seen:
            seen.add(g)
            groups.append(g)
    target = None
    for g in groups:
        if any(keyof(w) not in attempted for w in work if gkey(w) == g):
            target = g
            break
    if target is None:
        print(f"# all {len(groups)} groups done", flush=True)
        report(ck, GX, GY)
        ttnn.close_device(d)
        return EXIT_DONE

    phase, S, Pk, M, K, N = target
    todo = [w for w in work if gkey(w) == target and keyof(w) not in attempted]
    ndone = len(groups) - sum(1 for g in groups if any(keyof(w) not in attempted for w in work if gkey(w) == g))
    print(
        f"# grid {GX}x{GY}; group {phase} S{S}Pk{Pk} {M}x{K}x{N}: {len(todo)} configs "
        f"({ndone}/{len(groups)} groups done before this)",
        flush=True,
    )

    set_baseline_env() if phase == "base" else set_branch_spk_env(S, Pk)
    ta = torch.randn(M, K, dtype=torch.bfloat16)
    tb = torch.randn(K, N, dtype=torch.bfloat16)
    ref = (ta.float() @ tb.float()).flatten()
    ref = ref - ref.mean()
    refn = ref.norm()
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, device=d, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, device=d, layout=ttnn.TILE_LAYOUT)

    if os.path.exists(RAW):
        os.remove(RAW)
    session = []  # (rec, n_exec) in run order
    hung = False
    for w in todo:
        _, _, _, _, _, _, mb, kb, nb, sbh, sbw = w
        attempted.add(keyof(w))
        ck["attempted"].append(keyof(w))
        save_ckpt(ck)  # mark attempted BEFORE running -> hang skips this config on resume
        rec = dict(
            phase=phase,
            S=S,
            Pk=Pk,
            M=M,
            K=K,
            N=N,
            mb=mb,
            kb=kb,
            nb=nb,
            sbh=sbh,
            sbw=sbw,
            us=None,
            util=None,
            pcc=0.0,
            hang=False,
            err="",
        )
        n_exec = 0
        built = False
        try:
            d.clear_program_cache()
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sbh,
                subblock_w=sbw,
                compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY),
            )
            out = ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg)
            n_exec += 1
            built = True
            o = ttnn.to_torch(out).flatten().float()[: ref.numel()]
            o = o - o.mean()
            rec["pcc"] = round(float((o @ ref) / (o.norm() * refn + 1e-12)), 4)
            out.deallocate()
            for _ in range(WARMUP):
                ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg).deallocate()
                n_exec += 1
            ttnn.synchronize_device(d)
            for _ in range(REPS):
                ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg).deallocate()
                n_exec += 1
            ttnn.synchronize_device(d)
            session.append((rec, n_exec))
        except Exception as e:
            msg = str(e)[:200]
            is_hang = built or any(m in msg.lower() for m in HANG_MARKERS)
            rec["err"], rec["hang"] = msg, is_hang
            session.append((rec, n_exec))
            if is_hang:
                # persist what ran, then hard-exit (op already returned via timeout -> not mid-op) so the
                # wrapper resets. Timing for this session's earlier configs is lost (no close), but they are
                # marked attempted; the resumed session re-runs+times the group's remaining configs.
                print(f"HANG/device-error on {keyof(w)}: {msg}", flush=True)
                ck["results"].extend(r for r, _ in session)
                save_ckpt(ck)
                os._exit(EXIT_HANG)
            print(f"  build-skip {keyof(w)}: {msg}", flush=True)  # benign, device clean -> continue group

    # group finished cleanly: flush profiler via close, parse, assign us in run order
    ttnn.ReadDeviceProfiler(d)
    for t in (a, b):
        try:
            t.deallocate()
        except Exception:
            pass
    ttnn.close_device(d)
    ds = parse_durations(RAW)
    i = 0
    for rec, n_exec in session:
        seg = ds[i : i + n_exec]
        if rec["pcc"] >= 0.99 and n_exec == CHUNK and len(seg) == CHUNK:
            us = statistics.median(sorted(seg)[-REPS:]) / CLOCK * 1e6
            rec["us"] = round(us, 2)
            rec["util"] = round(100 * 2 * M * K * N / (PEAK * us * 1e-6), 2)
        i += n_exec
        ck["results"].append(rec)
    save_ckpt(ck)
    timed = sum(1 for r, _ in session if r["us"])
    print(f"# group done: {timed}/{len(session)} timed", flush=True)
    return EXIT_PROGRESS


if __name__ == "__main__":
    sys.exit(main())
