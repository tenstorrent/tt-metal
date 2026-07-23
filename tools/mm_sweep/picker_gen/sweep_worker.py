#!/usr/bin/env python3
"""Single (shape, config) measurement worker for the picker-generalization campaign.

One measurement per process (fresh device + fresh profiler CSV) as mandated by the campaign's
"one process/relaunch per candidate" discipline. Measures the PUBLIC op
ttnn.experimental.regime_a_matmul with an EXPLICIT RegimeAMatmulConfig (or config=None for the
production-picker baseline), using resident inputs, 1 warmup iteration, and >=8 timed iterations.

Prints exactly one JSON line to stdout:
  {"outcome": "...", "wall_us": <median|null>, "samples": [...], "pcc": <float|null>, "err": "..."}
outcome in {ok, validation, runtime, pcc}. (hang is detected by the orchestrator via timeout.)

argv: M K N Pk Ns Sm kb nsb iters do_pcc
  - config=None baseline: pass Pk=0 (sentinel) -> op called with no config.
  - do_pcc: "1" to compute PCC vs a torch fp32 reference, else "0".
"""
import sys, os, csv, json, statistics
import torch
import ttnn

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9  # BH

M, K, N, Pk, Ns, Sm, kb, nsb, iters, do_pcc = (int(x) for x in sys.argv[1:11])


def parse_walls_us():
    """Per-run wall = max over all (core, -KERNEL zone) durations; drop warmup run 0. Returns list us."""
    if not os.path.exists(CSV):
        return None
    rows = list(csv.reader(open(CSV)))
    ev = {}
    for row in rows[2:]:
        if len(row) < 12:
            continue
        z = row[10].strip()
        if not z.endswith("-KERNEL"):
            continue
        ev.setdefault(((row[1], row[2], row[3]), z), []).append((row[11].strip(), int(row[5])))
    dur = {}
    for k, lst in ev.items():
        ds, st = [], None
        for t, c in lst:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                ds.append(c - st)
                st = None
        dur[k] = ds
    nruns = min((len(v) for v in dur.values()), default=0)
    if nruns < 2:
        return None
    return [max(v[i] for v in dur.values()) / FREQ * 1e6 for i in range(1, nruns)]


def pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 1.0 if a.norm().item() == 0 and b.norm().item() == 0 else 0.0
    return torch.dot(a, b).item() / denom


def main():
    try:
        os.remove(CSV)
    except OSError:
        pass
    res = {"outcome": "runtime", "wall_us": None, "samples": None, "pcc": None, "err": ""}
    host = None
    t0 = t1 = None
    ran = False
    dev = ttnn.open_device(device_id=0)
    try:
        t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        a = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, dev)
        b = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wcfg)
        cfg = None
        if Pk != 0:
            cfg = ttnn.RegimeAMatmulConfig(
                k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb
            )
        try:
            out = None
            for _ in range(iters + 1):  # 1 warmup (dropped) + iters timed
                out = (
                    ttnn.experimental.regime_a_matmul(a, b, config=cfg)
                    if cfg is not None
                    else ttnn.experimental.regime_a_matmul(a, b)
                )
            ttnn.synchronize_device(dev)
            host = ttnn.to_torch(ttnn.from_device(out))
            ran = True
        except Exception as e:  # noqa: BLE001 — a build_plan TT_FATAL surfaces as a RuntimeError here
            msg = str(e)
            # Distinguish validation-time rejects (feasibility) from mid-run failures.
            res["outcome"] = (
                "validation"
                if (
                    "L1 over budget" in msg
                    or "cores" in msg
                    or "ownership" in msg
                    or "width-shard" in msg
                    or "must be <=" in msg
                )
                else "runtime"
            )
            res["err"] = msg[:400]
    finally:
        ttnn.close_device(dev)  # flushes the profiler CSV (walls must be parsed AFTER this)

    if not ran:
        print(json.dumps(res), flush=True)
        return
    walls = parse_walls_us()  # after close_device: CSV is now flushed
    res["samples"] = [round(w, 3) for w in walls] if walls else None
    res["wall_us"] = round(statistics.median(walls), 3) if walls else None
    if do_pcc and host is not None:
        ref = t0.to(torch.float32) @ t1.to(torch.float32)
        p = pcc(ref, host)
        res["pcc"] = round(p, 6)
        if p < 0.99:
            res["outcome"] = "pcc"
            res["err"] = f"pcc={p:.5f}"
            print(json.dumps(res), flush=True)
            return
    if res["wall_us"] is not None:
        res["outcome"] = "ok"
    else:
        res["outcome"] = "runtime"
        res["err"] = "no profiler walls parsed"
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
