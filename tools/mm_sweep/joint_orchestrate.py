#!/usr/bin/env python3
# Resumable joint (S,Pk,blocking) sweep over the full WH grid. Runs test_zz_jointsweep in batches under
# tracy, joins per-config device kernel duration, and appends rows to a persistent JSONL. Re-running
# skips shapes already in done.json (survives /tmp wipes, reboots, device hangs).
#   python tools/mm_sweep/joint_orchestrate.py [batch_size] [max_shapes]
import csv, json, math, os, subprocess, sys, time

REPO = "/localdev/cglagovich/tt-metal"
OUT = "/localdev/cglagovich/mm_jointsweep"  # PERSISTENT (not /tmp)
RESULTS = os.path.join(OUT, "results.jsonl")
DONE = os.path.join(OUT, "done.json")
T = "tests/ttnn/nightly/unit_tests/operations/experimental/test_zz_jointsweep.py::test_jointsweep"
REPS, WARMUP = 8, 2
CHUNK = 1 + WARMUP + REPS  # minimal-matmul rows per ok config (to_torch adds none)
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 25
MAXSHAPES = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9

M_TILES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
K_TILES = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]
N_TILES = M_TILES
ALL = [(mt * 32, kt * 32, nt * 32) for mt in M_TILES for kt in K_TILES for nt in N_TILES]

os.makedirs(OUT, exist_ok=True)
done = set(json.load(open(DONE))) if os.path.exists(DONE) else set()
todo = [s for s in ALL if f"{s[0]}x{s[1]}x{s[2]}" not in done][:MAXSHAPES]
print(f"total={len(ALL)} done={len(done)} todo={len(todo)} batch={BATCH}", flush=True)

t0 = time.time()
ncfg = 0
for bi in range(0, len(todo), BATCH):
    batch = todo[bi : bi + BATCH]
    rundir = f"/tmp/joint_b{bi}"
    manf = f"/tmp/joint_b{bi}_man.json"
    shp = f"/tmp/joint_b{bi}_shapes.json"
    subprocess.run(["rm", "-rf", rundir])
    json.dump([list(s) for s in batch], open(shp, "w"))
    env = dict(os.environ, FC_SHAPELIST=shp, FC_MANIFEST=manf, FC_REPS=str(REPS))
    if os.path.exists(manf):
        os.remove(manf)
    csvp = os.path.join(rundir, ".logs/cpp_device_perf_report.csv")
    try:
        subprocess.run(
            ["python", "-m", "tracy", "-p", "-r", "-o", rundir, "-m", f"pytest -q {T}"],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print(f"batch {bi}: TIMEOUT/HANG -> tt-smi -r, skip (retried on a later run)", flush=True)
        subprocess.run(["tt-smi", "-r"], capture_output=True)
        continue
    if not os.path.exists(manf) or not os.path.exists(csvp):
        # crash / dirty device: reset and SKIP (shapes stay undone -> retried next orchestrator run)
        print(f"batch {bi}: FAILED (no manifest/csv) -> tt-smi -r, skip", flush=True)
        subprocess.run(["tt-smi", "-r"], capture_output=True)
        continue
    man = json.load(open(manf))
    rows = [r for r in csv.DictReader(open(csvp)) if r.get("DEVICE KERNEL DURATION [ns]", "").strip()]
    rows.sort(key=lambda r: int(r["GLOBAL CALL COUNT"]))
    okrecs = [m for m in man if m["ok"]]
    import statistics

    i = 0
    for m in okrecs:
        ch = rows[i : i + CHUNK]
        i += CHUNK
        if len(ch) == CHUNK:
            m["us"] = round(statistics.median(float(r["DEVICE KERNEL DURATION [ns]"]) for r in ch[-REPS:]) / 1000, 3)
    with open(RESULTS, "a") as f:
        for m in man:
            f.write(json.dumps(m) + "\n")
    ncfg += len(man)
    for s in batch:
        done.add(f"{s[0]}x{s[1]}x{s[2]}")
    json.dump(sorted(done), open(DONE, "w"))
    el = time.time() - t0
    print(
        f"batch {bi//BATCH+1}: shapes[{bi}:{bi+len(batch)}] {len(man)} cfgs "
        f"({sum(1 for m in man if m['ok'])} ok) | total cfgs={ncfg} done={len(done)} "
        f"elapsed={el/60:.1f}min rate={ncfg/el:.1f}cfg/s",
        flush=True,
    )

print(f"ORCH_DONE done={len(done)}/{len(ALL)} cfgs_this_run={ncfg} elapsed={(time.time()-t0)/60:.1f}min", flush=True)
