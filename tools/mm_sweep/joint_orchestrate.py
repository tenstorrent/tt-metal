#!/usr/bin/env python3
# Resumable, SHARDED joint (S,Pk,blocking) sweep over the full WH grid. Runs test_zz_jointsweep in
# config-budgeted batches with the on-device profiler (NO tracy), computes per-config device kernel
# duration from the raw profiler CSV, appends to a per-shard JSONL.
#   python tools/mm_sweep/joint_orchestrate.py [cfg_budget] [shard_id] [num_shards]
# Shard k processes grid shapes where index % num_shards == k; writes results_shard{k}.jsonl +
# done_shard{k}.json. Excludes done_global.json. Merge offline: results_main.jsonl + results_shard*.jsonl.
#
# ROBUSTNESS (see minimal-matmul-kpar-deadlock-was-tracy + the N-slicing profiler race):
#  - No tracy -> no capture-release host daemon (which wedged on large traces and orphaned, holding the
#    device). We read the raw device CSV directly.
#  - A latent timing race in minimal_matmul's N-slicing dataflow can HANG the device on some high-slice
#    large-N configs when the profiler perturbs timing. A device hang can't be caught in-process (it
#    wedges pytest), so we: (a) keep batches small (cfg budget) so a hang costs little; (b) SALVAGE the
#    completed-prefix shapes from a timed-out batch (manifest+CSV written before the hang are valid);
#    (c) retry incomplete shapes a few times (the race is intermittent -> usually clears); (d) after
#    MAX_ATTEMPTS, record whatever partial data exists for the stuck shape and LOG it.
import json, math, os, statistics, subprocess, sys, time
from collections import deque

REPO = "/localdev/cglagovich/tt-metal"
OUT = "/localdev/cglagovich/mm_jointsweep"  # PERSISTENT (not /tmp)
CFG_BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 80  # configs per batch (small -> cheap hang)
SHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NSHARD = int(sys.argv[3]) if len(sys.argv) > 3 else 1
RESULTS = os.path.join(OUT, f"results_shard{SHARD}.jsonl")
DONE = os.path.join(OUT, f"done_shard{SHARD}.json")
STUCK_LOG = os.path.join(OUT, f"stuck_shard{SHARD}.json")  # shapes that kept hanging (partial data)
GLOBAL_DONE = os.path.join(OUT, "done_global.json")
T = "tests/ttnn/nightly/unit_tests/operations/experimental/test_zz_jointsweep.py::test_jointsweep"
REPS, WARMUP = 5, 2
CHUNK = 1 + WARMUP + REPS  # minimal-matmul ops per ok config (to_torch adds none)
# Primary hang detection is the DISPATCH layer: TT_METAL_OPERATION_TIMEOUT_SECONDS makes a hung device
# op raise (caught by the test, which then aborts the process) in ~OP_TIMEOUT s instead of the host
# spinning forever. Our matmuls are us-ms, so 15s has a ~100x margin (the check is on no-dispatch-
# progress, not wall time, so slow-but-progressing ops don't false-trip). WALL_TIMEOUT is just a
# backstop for hang modes the dispatch timeout misses (e.g. a host-side compile/link hang).
OP_TIMEOUT = "15"
WALL_TIMEOUT = 240  # backstop per-batch wall limit
MAX_ATTEMPTS = 2  # per-shape hang retries before declaring it stuck (race is intermittent)
RAW_CSV = os.path.join(REPO, "generated/profiler/.logs/profile_log_device.csv")
GX = GY = 8


def device_durations_ns(csv_path):
    """Ordered list of per-op device kernel durations (ns) from the raw profiler CSV.
    WH runs at 1GHz so 1 cycle = 1ns. Per op (grouped by run-host-id, monotonic with dispatch order):
    duration = max(FW ZONE_END) - min(FW ZONE_START) across all cores/RISCs."""
    lines = open(csv_path).read().splitlines()
    hdr = [h.strip() for h in lines[1].split(",")]
    ix = {h: i for i, h in enumerate(hdr)}
    zi, ti, ri, ty = ix["zone name"], ix["time[cycles since reset]"], ix["run host ID"], ix["type"]
    starts, ends = {}, {}
    for ln in lines[2:]:
        f = ln.split(",")
        if len(f) <= zi or not f[zi].strip().endswith("-FW"):
            continue
        rid, t = f[ri].strip(), f[ti].strip()
        if not rid or not t:
            continue
        t = int(t)
        if f[ty].strip() == "ZONE_START":
            starts[rid] = min(starts.get(rid, t), t)
        elif f[ty].strip() == "ZONE_END":
            ends[rid] = max(ends.get(rid, t), t)
    return [ends[r] - starts[r] for r in sorted(starts, key=int) if r in ends]


# --- mirror the test's config enumeration so we can budget batches by config count ---
def _valid_SPk(Kt):
    return [(S, Pk) for S in (1, 2, 4, 8) for Pk in (1, 2, 4, 8) if S * Pk <= GY and Kt % Pk == 0]


def _percore(Mt, Nt, S, Pk):
    transpose = Mt > Nt
    y = GY // (S * Pk)
    x = S * GX
    in0 = x if transpose else y
    in1 = y if transpose else x
    return math.ceil(Mt / in0), math.ceil(Nt / in1)


def _l1_fits(mb, nb, kb):
    est = 2 * 2048 * (mb * kb + kb * nb + mb * nb) + 4096 * mb * nb + 2048 * mb * nb
    return est <= int(0.92 * 1499136)


def _nblockings(Mpc, Npc, Ktb):
    nbs = sorted({Npc, max(1, (Npc + 1) // 2), 1})
    kbs = sorted({k for k in (4, 8, 16) if k <= Ktb}) or [Ktb]
    return sum(1 for nb in nbs for kb in kbs if _l1_fits(Mpc, nb, kb))


def est_configs(M, K, N):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    n = 1  # the auto/heuristic config
    for S, Pk in _valid_SPk(Kt):
        Mpc, Npc = _percore(Mt, Nt, S, Pk)
        n += 1 + _nblockings(Mpc, Npc, Kt // Pk)
    return n


M_TILES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
K_TILES = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]
N_TILES = M_TILES
ALL = [(mt * 32, kt * 32, nt * 32) for mt in M_TILES for kt in K_TILES for nt in N_TILES]

os.makedirs(OUT, exist_ok=True)
exclude = set(json.load(open(GLOBAL_DONE))) if os.path.exists(GLOBAL_DONE) else set()
exclude |= set(json.load(open(DONE))) if os.path.exists(DONE) else set()
done = set(json.load(open(DONE))) if os.path.exists(DONE) else set()
stuck = json.load(open(STUCK_LOG)) if os.path.exists(STUCK_LOG) else {}
todo = [s for i, s in enumerate(ALL) if i % NSHARD == SHARD and f"{s[0]}x{s[1]}x{s[2]}" not in exclude]
# work queue of (shape, attempts); each batch greedily packs shapes up to CFG_BUDGET configs
queue = deque((s, 0) for s in todo)
print(
    f"shard {SHARD}/{NSHARD} total={len(ALL)} excluded={len(exclude)} todo={len(todo)} "
    f"cfg_budget={CFG_BUDGET} op_timeout={OP_TIMEOUT}s wall={WALL_TIMEOUT}s",
    flush=True,
)


def reset_device():
    subprocess.run(["tt-smi", "-r"], capture_output=True)


t0 = time.time()
ncfg = 0
batch_no = 0
while queue:
    # pack a batch up to the config budget (always take >=1 shape even if it alone exceeds budget)
    batch, est = [], 0
    while queue and (not batch or est + est_configs(*queue[0][0]) <= CFG_BUDGET):
        s, att = queue.popleft()
        batch.append((s, att))
        est += est_configs(*s)
    batch_no += 1
    shapes = [s for s, _ in batch]
    tag = f"/tmp/joint_s{SHARD}_b{batch_no}"
    manf, shp = tag + "_man.json", tag + "_shapes.json"
    json.dump([list(s) for s in shapes], open(shp, "w"))
    for p in (manf, RAW_CSV):
        if os.path.exists(p):
            os.remove(p)
    env = dict(
        os.environ,
        FC_SHAPELIST=shp,
        FC_MANIFEST=manf,
        FC_REPS=str(REPS),
        TT_METAL_DEVICE_PROFILER="1",
        TT_METAL_OPERATION_TIMEOUT_SECONDS=OP_TIMEOUT,
    )
    clean = False
    try:
        r = subprocess.run(
            ["python", "-m", "pytest", "-q", T.split("::")[0], "-k", "jointsweep"],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=WALL_TIMEOUT,
            start_new_session=True,
        )
        clean = r.returncode == 0 and "BENCH_DONE" in (r.stdout or "")
    except subprocess.TimeoutExpired:
        pass  # backstop wall timeout; reset handled below
    # Any non-clean finish means a hang/crash occurred -> the device is likely unrecoverable; reset
    # before the next batch so it starts clean. (A dispatch timeout explicitly marks it unrecoverable.)
    if not clean:
        reset_device()

    man = json.load(open(manf)) if os.path.exists(manf) else []
    durs = device_durations_ns(RAW_CSV) if os.path.exists(RAW_CSV) else []
    # shapes present in the manifest, in execution order
    seen_order, present = [], set()
    for m in man:
        key = (m["M"], m["K"], m["N"])
        if key not in present:
            present.add(key)
            seen_order.append(key)
    # Clean finish (rc==0 + BENCH_DONE) -> ALL present shapes complete. Otherwise (hang/crash) the LAST
    # present shape is partial -> drop it (retry); earlier present shapes are fully complete.
    complete = set(seen_order) if clean else set(seen_order[:-1])

    # attach timings (chunk the ordered durations across ok configs)
    okrecs = [m for m in man if m["ok"]]
    i = 0
    for m in okrecs:
        ch = durs[i : i + CHUNK]
        i += CHUNK
        if len(ch) == CHUNK:
            m["us"] = round(statistics.median(ch[-REPS:]) / 1000, 3)

    # decide per shape: done (complete), stuck (hit MAX_ATTEMPTS while still failing), or requeue
    recorded_keys, newly_done, requeued, stuck_now = set(), [], 0, []
    for s, att in batch:
        key = (s[0], s[1], s[2])
        name = f"{s[0]}x{s[1]}x{s[2]}"
        if key in complete:
            recorded_keys.add(key)
            newly_done.append(name)
        elif att + 1 >= MAX_ATTEMPTS:
            # stuck: salvage whatever configs DID complete for this shape, mark done, log it
            recorded_keys.add(key)
            newly_done.append(name)
            stuck[name] = stuck.get(name, 0) + 1
            stuck_now.append(name)
        else:
            queue.append((s, att + 1))
            requeued += 1

    # append recorded configs (only for shapes we're marking done) to the results JSONL
    if recorded_keys:
        with open(RESULTS, "a") as f:
            for m in man:
                if (m["M"], m["K"], m["N"]) in recorded_keys:
                    f.write(json.dumps(m) + "\n")
                    ncfg += 1
        for name in newly_done:
            done.add(name)
        json.dump(sorted(done), open(DONE, "w"))
    if stuck_now:
        json.dump(stuck, open(STUCK_LOG, "w"))

    el = (time.time() - t0) / 60
    print(
        f"batch {batch_no}: {len(shapes)} shapes (~{est} cfgs){'' if clean else ' INCOMPLETE'} | "
        f"done+={len(newly_done)} requeue={requeued} stuck={stuck_now} | "
        f"total done={len(done)} cfgs={ncfg} queue={len(queue)} elapsed={el:.1f}min",
        flush=True,
    )

print(
    f"ORCH_DONE done={len(done)}/{len(ALL)} cfgs_this_run={ncfg} stuck={len(stuck)} "
    f"elapsed={(time.time()-t0)/60:.1f}min",
    flush=True,
)
