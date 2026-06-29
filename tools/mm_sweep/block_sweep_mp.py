"""
MULTIPROCESS parallel block sweep across N chips (e.g. 32 on a Wormhole Galaxy).

WHY MULTIPROCESS (not threads): the threaded sibling (block_sweep_mesh.py) carves 32 submeshes in one
process and runs 32 worker threads, but the per-config HOST work (from_torch tilize, to_torch untilize,
the fp32 PCC reference matmul) holds the Python GIL, so the chips are ~fully serialized -- 369 configs
took ~330s whether on 1 chip or 32. Here each chip gets its OWN PROCESS (no shared GIL), pinned via
TT_METAL_VISIBLE_DEVICES, with its OWN profiler CSV via TT_METAL_PROFILER_DIR. Each worker is the proven
single-chip flow (ttnn.open_device(0) + program cache + ReadDeviceProfiler per config + parse-after-close,
mirroring block_sweep.py). The orchestrator shards configs across chips, launches the workers, merges.

USAGE (on the Galaxy box, inside the dev container with env + python_env sourced):
    TT_METAL_DEVICE_PROFILER=1 python tools/mm_sweep/block_sweep_mp.py shapes.json out.json [num_chips]
  shapes.json = [[M,K,N], ...]; falls back to the built-in 5 smoke shapes if the file is missing.
  USE_PROFILER=0 -> wall-clock only (skip the profiler path).
"""
import os, sys, math, json, time, subprocess, statistics, threading


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


def percore(M, N, S, Pk, GX=8, GY=8):
    Mt, Nt = M // 32, N // 32
    x, y = S * GX, GY // (S * Pk)
    transpose = M > N
    return (math.ceil(Mt / (x if transpose else y)), math.ceil(Nt / (y if transpose else x)))


def spk_combos_for(Kt, GY=8, GX=8):
    """(S,Pk) combos to sweep jointly: S,Pk in {1,2,4,8}, S*Pk<=GY (grid rows), Pk divides Kt with >=2
    tiles/band. Includes 8 so the heuristic's own pick (which can be S=8 or Pk=8) is always covered."""
    out = []
    for S in (1, 2, 4, 8):
        for Pk in (1, 2, 4, 8):
            if S * Pk <= GY and Kt % Pk == 0 and (Pk == 1 or Kt // Pk >= 2):
                out.append((S, Pk))
    return out


JOINT_SPK = os.environ.get("BSWEEP_JOINT_SPK") == "1"


BASELINE = os.environ.get("BSWEEP_BASELINE") == "1"  # main-equivalent: no slicing/K-par, levers off
AUTO = (
    os.environ.get("BSWEEP_AUTO") == "1"
)  # full auto: factory picks (S,Pk) AND blocks (no pins) -> validate heuristic
CONFIGS_FILE = os.environ.get("BSWEEP_CONFIGS")  # path to a JSON list of exact [S,Pk,M,K,N,mb,kb,nb,sbh,sbw]
# items: run EXACTLY those (1 config/shape, no block sweep) -- e.g. an mcast on/off A/B at the optimal config.


def build_work(shapes, GX=8, GY=8):
    if CONFIGS_FILE:
        return [list(c) for c in json.load(open(CONFIGS_FILE))]
    """Flat list of work items [S, Pk, M, K, N, mb, kb, nb, sbh, sbw] across all shapes.
    Default: blocks at the heuristic (S,Pk). BSWEEP_JOINT_SPK=1: blocks at EVERY reasonable (S,Pk) combo.
    BSWEEP_BASELINE=1: force S=1,Pk=1 (no slicing/K-par) -> the 'main' block sweep (levers also disabled
    in the worker), for an optimized-baseline comparison."""
    if AUTO:  # one item/shape, S=Pk=0 sentinel -> worker leaves (S,Pk) AND blocks to the factory auto path
        return [[0, 0, M, K, N, 0, 0, 0, 0, 0] for M, K, N in shapes]
    work = []
    for M, K, N in shapes:
        Mt, Nt, Kt = M // 32, N // 32, K // 32
        spks = [(1, 1)] if BASELINE else (spk_combos_for(Kt, GY, GX) if JOINT_SPK else [pick_S_Pk(Mt, Nt, Kt, GY, GX)])
        for S, Pk in spks:
            pcM, pcN = percore(M, N, S, Pk, GX, GY)
            ktpb = Kt // max(1, Pk)
            for mb, kb, nb, sbh, sbw in gen_configs(pcM, pcN, ktpb):
                work.append([S, Pk, M, K, N, mb, kb, nb, sbh, sbw])
    return work


WARMUP, REPS = 2, 4
CHUNK = WARMUP + 1 + REPS
PEAK = 2048 * 64 * 1e9
USE_PROFILER = os.environ.get("USE_PROFILER", "1") == "1"


def util(M, K, N, us):
    return 100 * 2 * M * K * N / (PEAK * us * 1e-6) if us else None


# ============================== WORKER (one chip per process) ==============================
def worker_main():
    import torch, ttnn

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    shard = json.load(open(os.environ["BSWEEP_SHARD"]))
    out_path = os.environ["BSWEEP_OUT"]
    prof_dir = os.environ.get("TT_METAL_PROFILER_DIR")
    RAW = os.path.join(prof_dir, ".logs", "profile_log_device.csv") if prof_dir else None
    if RAW and os.path.exists(RAW):
        os.remove(RAW)

    d = ttnn.open_device(device_id=0)  # the single chip exposed by TT_METAL_VISIBLE_DEVICES
    try:
        d.enable_program_cache()
    except Exception:
        pass
    cc = ttnn.init_device_compute_kernel_config(
        d.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    open(out_path + ".started", "w").close()  # marker: device opened OK (orchestrator distinguishes
    # a stuck-at-open worker from one that's slow but working, so it only kills the genuinely stuck ones)

    if BASELINE:  # 'main' dataflow: disable the always-on large-N DRAM levers + any auto prefetch/K-par.
        for k in (
            "TT_MM_NUM_SLICES",
            "TT_MM_K_SLICES",
            "TT_MM_K_FUSED",
            "TT_MM_MCAST_PREFETCH",
            "TT_MM_MCAST_BROADCAST",
            "TT_MM_NO_AUTO_PREFETCH",
        ):
            os.environ.pop(k, None)
        os.environ["TT_MM_NO_LARGE_LEVERS"] = "1"
        os.environ["TT_MM_NO_AUTO_KPAR"] = "1"
        if os.environ.get("BSWEEP_FORCE_PREFETCH") == "1":
            # A/B the lever: force mcast+prefetch ON (TT_MM_MCAST_PREFETCH also enables mcast_broadcast)
            os.environ["TT_MM_MCAST_PREFETCH"] = "1"
        else:
            os.environ["TT_MM_NO_AUTO_PREFETCH"] = "1"  # forced OFF (== the main baseline)

    # Non-baseline (e.g. pinned exact configs at optimal S/Pk): honor the mcast+prefetch A/B flag too.
    # With K-par (Pk>1 -> K_FUSED) the factory gates mcast off anyway, so this only engages for Pk==1.
    if not BASELINE and os.environ.get("BSWEEP_FORCE_PREFETCH") == "1":
        os.environ["TT_MM_MCAST_PREFETCH"] = "1"

    # Large-N DRAM lever ablation: force the always-on output-contention levers OFF (N>=WIDE_OUTPUT_DIM=4096)
    # so a wide-shape joint sweep can be A/B'd against the levers-ON run to isolate their contribution.
    if not BASELINE and os.environ.get("BSWEEP_NO_LARGE_LEVERS") == "1":
        os.environ["TT_MM_NO_LARGE_LEVERS"] = "1"

    # Order by (S,Pk) then shape so we set the slicing env once per group and reuse inputs/ref per shape.
    shard.sort(key=lambda w: (w[0], w[1], w[2], w[3], w[4]))
    man = []  # [S,Pk,M,K,N,mb,kb,nb,sbh,sbw, ok, pcc]
    cur_spk = None
    cur_shape = None
    a = b = ref = None

    def free_inputs():
        nonlocal a, b
        for t in (a, b):
            try:
                if t is not None:
                    t.deallocate()
            except Exception:
                pass
        a = b = None

    for w in shard:
        S, Pk, M, K, N, mb, kb, nb, sbh, sbw = w
        if (S, Pk) != cur_spk:
            if AUTO:  # leave (S,Pk) to the factory auto heuristic: ensure nothing is pinned
                for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED", "TT_MM_NO_AUTO_KPAR"):
                    os.environ.pop(k, None)
            elif not BASELINE:  # baseline keeps slicing env unset (true 'main' non-sliced path)
                for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED", "TT_MM_NO_AUTO_KPAR"):
                    os.environ.pop(k, None)
                os.environ["TT_MM_NUM_SLICES"] = str(S)
                if Pk > 1:
                    os.environ["TT_MM_K_SLICES"] = str(Pk)
                    os.environ["TT_MM_K_FUSED"] = "1"
            cur_spk = (S, Pk)
            cur_shape = None  # env change -> rebuild inputs/programs
        if (M, K, N) != cur_shape:
            free_inputs()
            ta = torch.randn(M, K, dtype=torch.bfloat16)
            tb = torch.randn(K, N, dtype=torch.bfloat16)
            ref = (ta.float() @ tb.float()).flatten()
            ref = ref - ref.mean()
            ref_norm = ref.norm()
            a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, device=d, layout=ttnn.TILE_LAYOUT)
            b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, device=d, layout=ttnn.TILE_LAYOUT)
            cur_shape = (M, K, N)
        ok, p = True, 0.0
        try:
            d.clear_program_cache()  # bound cache; each config is a distinct program
            cfg = (
                None  # AUTO: no pinned config -> factory's default block sizer (+ auto S,Pk) decides everything
                if AUTO
                else ttnn.MinimalMatmulConfig(
                    M_block_size=mb,
                    K_block_size=kb,
                    N_block_size=nb,
                    subblock_h=sbh,
                    subblock_w=sbw,
                    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                )
            )
            # PCC on the FIRST invocation only: fused K-par corrupts the output on repeated invocations of
            # the same cached program (call 0 ok, calls 1+ -> garbage), but the device WORK (and timing) is
            # identical, so we validate the first call and time the rest. (Mirrors test_zz_jointsweep.py.)
            out = ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg)
            ot = ttnn.to_torch(out)
            out.deallocate()
            o = ot.flatten().float()
            o = o - o.mean()
            p = float((o @ ref) / (o.norm() * ref_norm + 1e-12))
            if p < 0.99:
                ok = False
            # timing: WARMUP + REPS more invocations (1 + WARMUP + REPS == CHUNK op-groups); outputs discarded
            for _ in range(WARMUP):
                ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg).deallocate()
            ttnn.synchronize_device(d)
            for _ in range(REPS):
                ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg).deallocate()
            ttnn.synchronize_device(d)
            if USE_PROFILER:
                ttnn.ReadDeviceProfiler(d)
        except Exception as e:
            ok, p = False, -1.0
            print(
                f"[worker dev={os.environ.get('TT_VISIBLE_DEVICES')}] FAIL {M}x{K}x{N} "
                f"mb{mb} kb{kb} nb{nb}: {str(e).splitlines()[0][:120]}",
                flush=True,
            )
            if "TIMEOUT" in str(e) or "hang" in str(e) or "system_memory_manager" in str(e):
                # device wedged: close_device would hang and the CSV won't flush, so this shard's results
                # are unrecoverable here -> force-exit now; the orchestrator retries the shard on another chip.
                print(
                    f"[worker dev={os.environ.get('TT_VISIBLE_DEVICES')}] device hang -> force exit "
                    f"(shard retried)",
                    flush=True,
                )
                sys.stdout.flush()
                os._exit(1)
        man.append([S, Pk, M, K, N, mb, kb, nb, sbh, sbw, ok, p])

    free_inputs()
    if USE_PROFILER:
        # Guarded close: a device that hung mid-shard makes close_device hang forever (outstanding reads),
        # which would block the orchestrator's whole round. Force-exit instead -> no output file -> the
        # orchestrator retries this shard on a healthy chip. (close also flushes this process's CSV.)
        ct = threading.Thread(target=lambda: ttnn.close_device(d), daemon=True)
        ct.start()
        ct.join(timeout=45)
        if ct.is_alive():
            print(
                f"[worker dev={os.environ.get('TT_VISIBLE_DEVICES')}] close_device hung -> force exit "
                f"(shard retried)",
                flush=True,
            )
            sys.stdout.flush()
            os._exit(1)
        ds = parse_durations(RAW)
        i = 0
        recs = []
        for it in man:
            S, Pk, M, K, N, mb, kb, nb, sbh, sbw, ok, p = it
            us = None
            if ok and len(ds[i : i + CHUNK]) == CHUNK:
                us = statistics.median(sorted(ds[i : i + CHUNK])[-REPS:]) / 1000.0
                i += CHUNK
            recs.append(
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
                    "S": S,
                    "Pk": Pk,
                }
            )
        json.dump(recs, open(out_path, "w"))
    else:
        try:
            ttnn.close_device(d)
        except Exception:
            pass
        recs = [
            {
                "M": M,
                "K": K,
                "N": N,
                "mb": mb,
                "kb": kb,
                "nb": nb,
                "sbh": sbh,
                "sbw": sbw,
                "us": None,
                "util": None,
                "pcc": p,
                "S": S,
                "Pk": Pk,
            }
            for (S, Pk, M, K, N, mb, kb, nb, sbh, sbw, ok, p) in man
        ]
        json.dump(recs, open(out_path, "w"))
    print(f"[worker dev={os.environ.get('TT_VISIBLE_DEVICES')}] done {len(man)} configs -> {out_path}", flush=True)
    os._exit(0)


def parse_durations(RAW):
    """Single-chip CSV -> [per-op FW-zone durations (ns) in run order]."""
    try:
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
    except Exception as e:
        print("parse_durations failed:", e, flush=True)
        return []


# ============================== ORCHESTRATOR ==============================
TIMEOUT = int(os.environ.get("BSWEEP_WORKER_TIMEOUT", "3600"))  # generous overall cap per round
STAGGER = float(os.environ.get("BSWEEP_LAUNCH_STAGGER", "0.3"))  # delay between launches (avoid TLB races)
STARTUP_GRACE = int(os.environ.get("BSWEEP_STARTUP_GRACE", "180"))  # kill a worker with no .started (stuck at open)


def run_round(round_work, chip_ids, workdir, tag):
    """Shard round_work across chip_ids, launch one staggered process per chip (TT_VISIBLE_DEVICES=chip),
    grace-kill stragglers. Returns (merged_results, set(chips_that_produced_output), {chip: leftover_shard})."""
    self_path = os.path.abspath(__file__)
    # LPT balancing: assign heaviest configs (cost ~ M*K*N) first to the least-loaded chip, so wall-times
    # are even (round-robin by count is bad when shape costs vary wildly, as in FLUX).
    shards = {c: [] for c in chip_ids}
    load = {c: 0 for c in chip_ids}
    for w in sorted(round_work, key=lambda w: -(w[2] * w[3] * w[4])):
        c = min(chip_ids, key=lambda c: load[c])
        shards[c].append(w)
        load[c] += w[2] * w[3] * w[4]
    procs = []
    for chip in chip_ids:
        if not shards[chip]:
            continue
        shard_f = os.path.join(workdir, f"shard_{tag}_{chip}.json")
        out_f = os.path.join(workdir, f"out_{tag}_{chip}.json")
        log_f = os.path.join(workdir, f"log_{tag}_{chip}.txt")
        json.dump(shards[chip], open(shard_f, "w"))
        for stale in (out_f, out_f + ".started"):
            if os.path.exists(stale):
                os.remove(stale)
        env = dict(os.environ)
        env["BSWEEP_WORKER"] = "1"
        env["BSWEEP_SHARD"] = shard_f
        env["BSWEEP_OUT"] = out_f
        # TT_VISIBLE_DEVICES (UMD-level) restricts the cluster open to just this chip -> tiny TLB footprint
        # so many processes coexist. (TT_METAL_VISIBLE_DEVICES only gates ttnn, UMD still opens all 32.)
        env["TT_VISIBLE_DEVICES"] = str(chip)
        env.pop("TT_METAL_VISIBLE_DEVICES", None)
        env["TT_METAL_PROFILER_DIR"] = os.path.join(workdir, f"prof_{tag}_{chip}")
        for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED", "TT_MM_NO_AUTO_KPAR"):
            env.pop(k, None)  # worker sets these itself per (S,Pk) group
        lf = open(log_f, "w")
        p = subprocess.Popen([sys.executable, self_path], env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((chip, p, out_f, lf))
        time.sleep(STAGGER)  # stagger so 32 opens don't race for TLB windows / ETH heartbeat
    print(f"[{tag}] launched {len(procs)} workers on chips {[c for c,_,_,_ in procs]}", flush=True)

    # Wait: kill ONLY workers genuinely stuck at device open (no .started marker within STARTUP_GRACE) or
    # the rare worker exceeding the generous hard TIMEOUT. A worker that opened (.started) is left to finish
    # its shard however long it legitimately takes -- we do NOT kill it because other chips finished first.
    t0 = time.time()
    while True:
        alive = [(chip, p, out_f) for chip, p, out_f, lf in procs if p.poll() is None]
        if not alive:
            break
        now = time.time()
        for chip, p, out_f in alive:
            started = os.path.exists(out_f + ".started")
            if not started and now - t0 > STARTUP_GRACE:
                print(f"[{tag}] chip {chip}: killing (stuck at open, no .started in {STARTUP_GRACE}s)", flush=True)
                p.kill()
            elif now - t0 > TIMEOUT:
                print(f"[{tag}] chip {chip}: killing (hard timeout {TIMEOUT}s)", flush=True)
                p.kill()
        time.sleep(2)
    for chip, p, out_f, lf in procs:
        try:
            p.wait(timeout=10)
        except Exception:
            pass
        lf.close()

    results, good, leftover = [], set(), {}
    for chip, p, out_f, lf in procs:
        if os.path.exists(out_f):
            try:
                results.extend(json.load(open(out_f)))
                good.add(chip)
                continue
            except Exception:
                pass
        leftover[chip] = shards[chip]  # produced nothing usable -> its work needs a retry elsewhere
    return results, good, leftover


def orchestrator_main():
    SHAPES = (
        json.load(open(sys.argv[1]))
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1])
        else [[4224, 6144, 4608], [2112, 6144, 6144], [1152, 6144, 4608], [4864, 4096, 1024], [1216, 4096, 1024]]
    )
    OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.getcwd(), "bsweep_mp.json")
    NUM_CHIPS = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    WORKDIR = os.environ.get("BSWEEP_WORKDIR", os.path.join(os.path.dirname(os.path.abspath(OUT)), "mp_work"))
    os.makedirs(WORKDIR, exist_ok=True)

    work = build_work(SHAPES)
    work.sort(key=lambda w: (w[0], w[1], w[2], w[3], w[4]))  # same-shape adjacent -> input reuse per worker
    print(
        f"{len(SHAPES)} shapes -> {len(work)} configs across {NUM_CHIPS} chips "
        f"(~{len(work)//NUM_CHIPS}/chip); profiler={USE_PROFILER}",
        flush=True,
    )

    t_start = time.time()
    merged, good, leftover = run_round(work, list(range(NUM_CHIPS)), WORKDIR, "r1")
    # Retry dropped chips' work on the chips that succeeded (known-healthy; TLB pressure now gone).
    rnd = 2
    while leftover and good and rnd <= 4:
        retry_work = [w for sh in leftover.values() for w in sh]
        print(
            f"retry round {rnd}: {len(retry_work)} configs from {len(leftover)} dropped chips "
            f"-> {len(good)} healthy chips",
            flush=True,
        )
        r, g, leftover = run_round(retry_work, sorted(good), WORKDIR, f"r{rnd}")
        merged.extend(r)
        rnd += 1

    json.dump(merged, open(OUT, "w"), indent=0)
    ok = sum(1 for r in merged if r["pcc"] > 0.99)
    timed = sum(1 for r in merged if r.get("us"))
    dropped = sum(len(sh) for sh in leftover.values())
    print(
        f"DONE: {len(merged)} configs, {ok} PCC-pass, {timed} timed, {dropped} configs unrecovered "
        f"in {time.time()-t_start:.0f}s -> {OUT}",
        flush=True,
    )


if __name__ == "__main__":
    if os.environ.get("BSWEEP_WORKER"):
        worker_main()
    else:
        orchestrator_main()
