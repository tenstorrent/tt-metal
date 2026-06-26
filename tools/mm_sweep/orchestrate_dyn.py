#!/usr/bin/env python3
"""
DYNAMIC multi-galaxy dispatcher for the minimal_matmul sweep. Runs on the controller (A1).

- Work unit = ONE shape per chunk (small: a galaxy failing loses <=1 shape).
- Dynamic assignment: each free galaxy gets the next pending chunk (auto load-balances).
- Semi-robust: if a chunk job dies / yields no output / the node goes down, the chunk is REQUEUED onto
  another galaxy and the dead node is dropped from the pool.
- Live monitoring: per-galaxy status (chunk, elapsed, chips-done/32, alive?) + pending/done/dead counts
  written to mm_sweep_out/dyn_status.txt (and stdout) every poll, so progress is visible.

Each chunk is an `sbatch --nodelist=<node>` job that runs block_sweep_mp.py on that node's 32 chips
(tt-metal pre-staged by rsync). Controller pulls per-chunk JSON and merges at the end.

USAGE: python tools/mm_sweep/orchestrate_dyn.py shapes.json out.json [num_nodes]
  env: BSWEEP_JOINT_SPK=1 (joint S,Pk), PART=wh_cluster, MM_NODES="a b c", STALL_SECS=600
"""
import os, sys, json, subprocess, time

HOME = "/home/cglagovich"
TTM = f"{HOME}/tt-metal"
OUTD = f"{HOME}/mm_sweep_out"
SBATCH = f"{TTM}/tools/mm_sweep/run_node.sbatch"
WORK = f"{OUTD}/dyn"
STATUS = f"{OUTD}/dyn_status.txt"
EXC = [
    "--exclude=.git",
    "--exclude=generated/profiler",
    "--exclude=generated/watcher",
    "--exclude=generated/inspector",
    "--exclude=mp_work",
]
PART = os.environ.get("PART", "wh_cluster")
JOINT = os.environ.get("BSWEEP_JOINT_SPK", "1")
STALL = int(os.environ.get("STALL_SECS", "900"))  # job alive but no progress this long -> kill+requeue
POLL = 15
RSH = "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"


def sh(c, t=60):
    try:
        return subprocess.run(c, shell=True, text=True, capture_output=True, timeout=t)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(c, 124, "", "timeout")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def idle_nodes():
    return [n for n in sh(f"sinfo -h -p {PART} -t idle -o '%n'").stdout.split() if n]


def node_down(n):
    st = sh(f"sinfo -h -n {n} -o '%t'", t=20).stdout.split()
    return any(any(k in s for k in ("down", "drain", "fail", "inval")) for s in st) if st else False


def chips_done(n, cid):  # best-effort progress: # of per-chip result files for the running chunk
    r = sh(f"{RSH} {n} 'ls {WORK}/mp_work/out_*_*.json 2>/dev/null | wc -l'", t=20)
    try:
        return int(r.stdout.strip())
    except Exception:
        return -1


def main():
    shapes = json.load(open(sys.argv[1]))
    out_file = sys.argv[2] if len(sys.argv) > 2 else f"{OUTD}/all_joint_out.json"
    nnodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    nodes = (os.environ["MM_NODES"].split() if os.environ.get("MM_NODES") else idle_nodes())[:nnodes]
    os.makedirs(WORK, exist_ok=True)
    chunks = []
    for i, s in enumerate(shapes):
        cf = f"{WORK}/chunk_{i}.json"
        json.dump([s], open(cf, "w"))
        chunks.append({"id": i, "shape": tuple(s), "cf": cf, "out": f"{WORK}/out_{i}.json", "tries": 0})
    pending = list(range(len(chunks)))
    log(f"controller={os.uname().nodename} nodes({len(nodes)})={nodes} chunks={len(chunks)} joint={JOINT}")

    log("staging tt-metal to all nodes (parallel rsync; ~7GB first time, incremental after)...")
    ps = [
        subprocess.Popen(
            f"rsync -a -e \"{RSH}\" {' '.join(EXC)} {TTM}/ {n}:{TTM}/ && " f"{RSH} {n} 'mkdir -p {WORK}'", shell=True
        )
        for n in nodes
    ]
    for n, p in zip(nodes, ps):
        if p.wait() != 0:
            log(f"  WARN staging failed: {n}")

    running = {}  # node -> {cid,jid,start,prog,last_prog_t}
    results = []
    done = set()
    dead = set()
    t0 = time.time()
    while pending or running:
        # ---- assign pending chunks to free, healthy nodes ----
        for n in [x for x in nodes if x not in running and x not in dead]:
            if not pending:
                break
            if node_down(n):
                dead.add(n)
                log(f"node {n} down -> dropped")
                continue
            cid = pending.pop(0)
            ch = chunks[cid]
            ch["tries"] += 1
            sh(f"rsync -a -e \"{RSH}\" {ch['cf']} {n}:{ch['cf']}", t=60)
            env = (
                (f"BSWEEP_JOINT_SPK={JOINT} " if JOINT else "")
                + (f"BSWEEP_BASELINE={os.environ['BSWEEP_BASELINE']} " if os.environ.get("BSWEEP_BASELINE") else "")
                + (
                    f"BSWEEP_FORCE_PREFETCH={os.environ['BSWEEP_FORCE_PREFETCH']} "
                    if os.environ.get("BSWEEP_FORCE_PREFETCH")
                    else ""
                )
                + (
                    f"BSWEEP_NO_LARGE_LEVERS={os.environ['BSWEEP_NO_LARGE_LEVERS']} "
                    if os.environ.get("BSWEEP_NO_LARGE_LEVERS")
                    else ""
                )
            )
            r = sh(f"{env}sbatch --partition={PART} --nodelist={n} --parsable {SBATCH} {ch['cf']} {ch['out']} 32")
            jid = r.stdout.strip().split(";")[0]
            if not jid.isdigit():
                pending.append(cid)
                log(f"  sbatch failed on {n} ({r.stdout}{r.stderr}); will retry")
                continue
            running[n] = {"cid": cid, "jid": jid, "start": time.time(), "prog": 0, "lpt": time.time()}
            log(f"assign chunk {cid} {ch['shape']} -> {n} job {jid} (try {ch['tries']})")

        # ---- poll running ----
        live = {ln.split()[0]: ln.split()[1] for ln in sh("squeue -h -o '%i %t'").stdout.splitlines() if ln.split()}
        for n in list(running):
            R = running[n]
            cid = R["cid"]
            if R["jid"] not in live:  # job ended
                col = f"{WORK}/collected_{cid}.json"
                ok = sh(
                    f"rsync -a -e \"{RSH}\" {n}:{chunks[cid]['out']} {col}", t=60
                ).returncode == 0 and os.path.exists(col)
                rows = []
                if ok:
                    try:
                        rows = json.load(open(col))
                    except Exception:
                        rows = []
                if rows:
                    results.extend(rows)
                    done.add(cid)
                    log(f"chunk {cid} DONE on {n}: {len(rows)} cfgs in {time.time()-R['start']:.0f}s")
                else:
                    log(f"chunk {cid} FAILED on {n} (no/empty output) -> requeue")
                    pending.append(cid)
                    if chunks[cid]["tries"] >= 3:
                        log(f"  chunk {cid} failed 3x; will keep retrying on other nodes")
                del running[n]
            else:
                p = chips_done(n, cid)
                if p > R["prog"]:
                    R["prog"] = p
                    R["lpt"] = time.time()
                # node died mid-job, or stalled with no progress -> kill + requeue elsewhere
                if node_down(n):
                    log(f"node {n} went DOWN mid-chunk {cid} -> scancel + requeue + drop node")
                    sh(f"scancel {R['jid']}")
                    pending.append(cid)
                    dead.add(n)
                    del running[n]
                elif time.time() - R["lpt"] > STALL:
                    log(f"chunk {cid} on {n} STALLED ({STALL}s no progress) -> scancel + requeue")
                    sh(f"scancel {R['jid']}")
                    pending.append(cid)
                    del running[n]

        # ---- live status ----
        lines = [
            f"=== dyn sweep  t+{time.time()-t0:.0f}s  done={len(done)}/{len(chunks)} "
            f"pending={len(pending)} running={len(running)} dead={sorted(dead)} ==="
        ]
        for n in nodes:
            if n in running:
                R = running[n]
                lines.append(
                    f"  {n}: chunk {R['cid']} {chunks[R['cid']]['shape']} "
                    f"chips~{R['prog']}/32 {time.time()-R['start']:.0f}s job {R['jid']}"
                )
            elif n in dead:
                lines.append(f"  {n}: DEAD")
            else:
                lines.append(f"  {n}: idle")
        open(STATUS, "w").write("\n".join(lines) + "\n")
        if pending or running:
            time.sleep(POLL)

    json.dump(results, open(out_file, "w"), indent=0)
    ok = sum(1 for r in results if r["pcc"] > 0.99)
    shp = len({(r["M"], r["K"], r["N"]) for r in results})
    log(
        f"ALL DONE: {len(results)} cfgs, {ok} PCC-pass, {shp}/{len(chunks)} shapes, "
        f"{len(dead)} dead nodes, {time.time()-t0:.0f}s -> {out_file}"
    )


if __name__ == "__main__":
    main()
