#!/usr/bin/env python3
"""Persistent batch measurement worker for the generator-driven campaign.

Removes the ~9 s/candidate one-process-per-config overhead: Torch/TTNN are imported ONCE for the whole
run, and the device is opened once per MINI-BATCH of configs (amortising device-open across the batch).
Per-config kernel wall is recovered from the device-profiler CSV by RUN-HOST-ID demux (the CSV is only
flushed on device close, so each mini-batch is one open/run/close/parse cycle). One JSON record is
atomically appended per configuration immediately after its mini-batch closes; a restart skips configs
already present in the output JSONL. PCC is verified against an fp32 reference computed once per shape.

Timing methodology is identical work to the isolated worker (resident inputs, 1 warmup + N timed iters,
device-profiler -KERNEL wall = max over cores); only the process/device lifecycle differs. Verified to
agree with the isolated method within ~1% (see verify_batch_timing.py).

Job file (JSON): {"M","K","N","iters","do_pcc","minibatch","out_jsonl",
                  "configs":[{"cfg":[Ns,Pk,Sm,kb,nsb], ...generator metadata to echo into results...}]}
The worker echoes each config's generator metadata (geometry/model_costs/reasons) into its result record.
"""
import csv, json, os, statistics, sys
import torch
import ttnn

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9  # BH


def clear_csv():
    try:
        os.remove(CSV)
    except OSError:
        pass


def demux_walls_us(n_ran):
    """Parse the mini-batch CSV. Group -KERNEL zones by run-host-id, order groups chronologically, and
    chunk into `n_ran` consecutive per-config groups of (iters+1) invocations. Returns list (len n_ran)
    of per-config sample lists (us, warmup dropped), or None entries if a group is malformed."""
    if not os.path.exists(CSV):
        return [None] * n_ran
    per = {}  # runid -> {(core,zone): [(type,cyc)]}, and min start
    for row in csv.reader(open(CSV)):
        if len(row) < 12:
            continue
        z = row[10].strip()
        if not z.endswith("-KERNEL"):
            continue
        runid = row[7].strip()
        per.setdefault(runid, {}).setdefault((row[1], row[2], row[3]), []).append((row[11].strip(), int(row[5])))
    # per-runid wall (max over cores of END-START) + min start cycle for chronological ordering.
    wall, start = {}, {}
    for runid, cores in per.items():
        mx, mn = 0, None
        for lst in cores.values():
            st = None
            for t, c in lst:
                if t == "ZONE_START":
                    st = c
                    mn = c if mn is None else min(mn, c)
                elif t == "ZONE_END" and st is not None:
                    mx = max(mx, c - st)
                    st = None
        wall[runid] = mx
        start[runid] = mn if mn is not None else 0
    order = sorted(wall, key=lambda r: start[r])
    if n_ran == 0:
        return []
    if len(order) % n_ran != 0:
        return None  # misalignment -> caller marks the whole batch for redo
    per_cfg = len(order) // n_ran
    out = []
    for i in range(n_ran):
        grp = order[i * per_cfg:(i + 1) * per_cfg]
        samples = [wall[r] / FREQ * 1e6 for r in grp[1:]]  # drop warmup (first invocation)
        out.append(samples if samples else None)
    return out


def pcc(a, b):
    a = a.flatten().to(torch.float32); b = b.flatten().to(torch.float32)
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 1.0 if a.norm().item() == 0 and b.norm().item() == 0 else 0.0
    return torch.dot(a, b).item() / denom


def classify_err(msg):
    val = any(s in msg for s in ("L1 over budget", "cores", "ownership", "width-shard", "must be <=",
                                 "planner rejected"))
    return "validation" if val else "runtime"


def run_minibatch(M, K, N, t0, t1, ref, batch, iters, do_pcc):
    """Open device, upload resident inputs, run each config's (iters+1) invocations, close, demux.
    Returns list of result dicts aligned with `batch`."""
    results = [None] * len(batch)
    ran_idx = []           # indices of configs that produced profiler invocations, in issue order
    live = {}              # idx -> {'pcc':..} captured while device open
    clear_csv()
    dev = ttnn.open_device(device_id=0)
    try:
        a = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, dev)
        b = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wcfg)
        for i, item in enumerate(batch):
            Ns, Pk, Sm, kb, nsb = item["cfg"]
            try:
                cfg = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm,
                                               k_block_tiles=kb, n_subblock_tiles=nsb)
                out = None
                for _ in range(iters + 1):  # 1 warmup + iters timed
                    out = ttnn.experimental.regime_a_matmul(a, b, config=cfg)
                ttnn.synchronize_device(dev)
                host = ttnn.to_torch(ttnn.from_device(out))
                p = pcc(ref, host) if do_pcc else None
                live[i] = {"pcc": (round(p, 6) if p is not None else None)}
                ran_idx.append(i)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                results[i] = {"outcome": classify_err(msg), "wall_us": None, "samples": None,
                              "pcc": None, "err": msg[:300]}
    finally:
        ttnn.close_device(dev)  # flush CSV
    # demux timing for the configs that ran
    walls = demux_walls_us(len(ran_idx))
    if walls is None:  # misaligned -> mark ran configs for redo (leave results[i] None => not checkpointed)
        return results, False
    for pos, i in enumerate(ran_idx):
        samples = walls[pos]
        p = live[i]["pcc"]
        if samples is None:
            results[i] = {"outcome": "runtime", "wall_us": None, "samples": None, "pcc": p,
                          "err": "no profiler walls"}
        else:
            med = round(statistics.median(samples), 3)
            outcome = "ok"
            err = ""
            if do_pcc and p is not None and p < 0.99:
                outcome, err = "pcc", f"pcc={p:.5f}"
            results[i] = {"outcome": outcome, "wall_us": med,
                          "samples": [round(s, 3) for s in samples], "pcc": p, "err": err}
    return results, True


def main():
    job = json.load(open(sys.argv[1]))
    M, K, N = job["M"], job["K"], job["N"]
    iters, do_pcc, mb = job["iters"], job["do_pcc"], job["minibatch"]
    out_path = job["out_jsonl"]
    configs = job["configs"]

    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            line = line.strip()
            if line:
                try:
                    done.add(tuple(json.loads(line)["cfg"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [c for c in configs if tuple(c["cfg"]) not in done]
    if not todo:
        print(json.dumps({"status": "already-complete", "shape": f"{M}x{K}x{N}"}), flush=True)
        return

    t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    ref = (t0.to(torch.float32) @ t1.to(torch.float32)) if do_pcc else None

    n_ok = 0
    for s in range(0, len(todo), mb):
        batch = todo[s:s + mb]
        results, aligned = run_minibatch(M, K, N, t0, t1, ref, batch, iters, do_pcc)
        with open(out_path, "a") as f:
            for item, res in zip(batch, results):
                if res is None:
                    continue  # misaligned/never-ran -> not checkpointed, retried on resume
                # Echo the generator's full provenance (geometry fields, model costs, reasons, factor
                # class) beside the measurement. The generator flattens Geometry fields at the row top
                # level, so store everything except cfg under "gen".
                rec = {"cfg": item["cfg"], "gen": {k: v for k, v in item.items() if k != "cfg"}, **res}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
                if res["outcome"] == "ok":
                    n_ok += 1
        print(json.dumps({"status": "progress", "shape": f"{M}x{K}x{N}",
                          "batch_end": s + len(batch), "total": len(todo), "aligned": aligned}), flush=True)
    print(json.dumps({"status": "done", "shape": f"{M}x{K}x{N}", "ok": n_ok}), flush=True)


if __name__ == "__main__":
    main()
