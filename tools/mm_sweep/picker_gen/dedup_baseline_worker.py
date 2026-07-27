#!/usr/bin/env python3
"""One fresh-process/device relaunch for the cross-Ns in0-dedup baseline.

Measures ONE fixed config on the CURRENT PRODUCTION path (explicit RegimeAMatmulConfig, UNFUSED output),
with resident BF16 inputs, 1 warmup + 8 timed iterations. Emits one JSON line with every per-iteration
kernel wall sample, per-RISC spans (BRISC / NCRISC / TRISC = data-movement/data-movement/compute), per-core
spread, PCC vs a CPU FP32 golden, and a cached-program-replay PCC.

Config tuple order is (Ns,Pk,Sm,kb,nsb) per the task. argv: M K N Ns Pk Sm kb nsb iters
"""
import sys, os, csv, json, statistics
import torch
import ttnn

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9  # BH

M, K, N, Ns, Pk, Sm, kb, nsb, iters = (int(x) for x in sys.argv[1:10])


def parse_zones():
    """Return per-(core,risc_proc) ordered duration lists (cycles), one per invocation.
    core = (x,y) from cols 1,2; risc_proc = col 3 (BRISC/NCRISC/TRISC_0/1/2)."""
    if not os.path.exists(CSV):
        return None
    ev = {}
    for row in csv.reader(open(CSV)):
        if len(row) < 12:
            continue
        z = row[10].strip()
        if not z.endswith("-KERNEL"):
            continue
        key = ((row[1].strip(), row[2].strip()), row[3].strip())
        ev.setdefault(key, []).append((row[11].strip(), int(row[5])))
    dur = {}
    for key, lst in ev.items():
        ds, st = [], None
        for t, c in lst:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                ds.append(c - st)
                st = None
        dur[key] = ds
    return dur


def summarize(dur, n_timed):
    """Return the EXACT n_timed timed-iteration samples (invocation 0 = warmup dropped; any trailing
    cached-replay invocation excluded), per-RISC median spans (us), and per-core spread (median + max, %)."""
    n = min((len(v) for v in dur.values()), default=0)
    if n < 2:
        return None
    idxs = list(range(1, 1 + min(n_timed, n - 1)))  # drop warmup (0) and the trailing replay call
    cores = sorted({k[0] for k in dur})
    procs = {k[1] for k in dur}

    def dur_of(key, i):
        return dur[key][i] if i < len(dur[key]) else 0

    walls = [max(dur_of(k, i) for k in dur) / FREQ * 1e6 for i in idxs]  # whole-kernel per iter

    def risc_group(prefix):
        keys = [k for k in dur if k[1].startswith(prefix)]
        if not keys:
            return None
        per_iter = [max(dur_of(k, i) for k in keys) / FREQ * 1e6 for i in idxs]
        return round(statistics.median(per_iter), 3)

    risc = {"BRISC": risc_group("BRISC"), "NCRISC": risc_group("NCRISC"), "TRISC": risc_group("TRISC")}

    # per-core wall (max over that core's procs) -> spread across cores, per iter.
    spreads = []
    for i in idxs:
        per_core = []
        for c in cores:
            cd = [dur_of(k, i) for k in dur if k[0] == c]
            if cd:
                per_core.append(max(cd))
        if len(per_core) >= 2 and min(per_core) > 0:
            spreads.append((max(per_core) - min(per_core)) / max(per_core) * 100)
    spread = {"median_pct": round(statistics.median(spreads), 2) if spreads else None,
              "max_pct": round(max(spreads), 2) if spreads else None,
              "n_cores": len(cores)}
    return walls, risc, spread


def pcc(a, b):
    a = a.flatten().to(torch.float32); b = b.flatten().to(torch.float32)
    a = a - a.mean(); b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    if d == 0:
        return 1.0 if a.norm().item() == 0 and b.norm().item() == 0 else 0.0
    return torch.dot(a, b).item() / d


def main():
    try:
        os.remove(CSV)
    except OSError:
        pass
    res = {"outcome": "runtime", "err": ""}
    ran = False
    host = host2 = None
    dev = ttnn.open_device(device_id=0)
    try:
        t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        a = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, dev)
        b = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wcfg)
        cfg = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
        try:
            out = None
            for _ in range(iters + 1):  # 1 warmup + iters timed, UNFUSED, production path
                out = ttnn.experimental.regime_a_matmul(a, b, config=cfg)
            ttnn.synchronize_device(dev)
            host = ttnn.to_torch(ttnn.from_device(out))
            # cached-program replay: a fresh call after sync hits the program cache; must stay correct.
            out2 = ttnn.experimental.regime_a_matmul(a, b, config=cfg)
            ttnn.synchronize_device(dev)
            host2 = ttnn.to_torch(ttnn.from_device(out2))
            ran = True
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            res["outcome"] = "validation" if ("planner rejected" in msg or "L1 over budget" in msg
                                              or "must be <=" in msg or "cores" in msg) else "runtime"
            res["err"] = msg[:400]
    finally:
        ttnn.close_device(dev)

    if not ran:
        print(json.dumps(res), flush=True)
        return
    dur = parse_zones()
    s = summarize(dur, iters) if dur else None
    ref = t0.to(torch.float32) @ t1.to(torch.float32)
    res["pcc"] = round(pcc(ref, host), 6)
    res["pcc_cached_replay"] = round(pcc(ref, host2), 6)
    res["replay_matches"] = bool(torch.equal(host, host2))
    if s is None:
        res["outcome"] = "runtime"; res["err"] = "no profiler walls"
        print(json.dumps(res), flush=True); return
    walls, risc, spread = s
    res["outcome"] = "ok" if res["pcc"] >= 0.999 else "pcc"
    res["samples_us"] = [round(w, 3) for w in walls]
    res["median_us"] = round(statistics.median(walls), 3)
    res["risc_spans_us"] = risc
    res["core_spread"] = spread
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
