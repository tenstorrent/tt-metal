#!/usr/bin/env python3
# Steady-state performance characterization of ttnn.experimental.regime_a_matmul on Blackhole
# (the INDEPENDENT TTNN op is the system under test — NOT the C++ prototype).
#
# Design:
#   - Resident device inputs: in0 (DRAM interleaved) + in1 (pre-sharded resident via the op's
#     create_regime_a_weight_memory_config). Conversion/allocation is done ONCE, outside the timed loop.
#   - Per measurement: verify PCC >= 0.999 (vs torch) BEFORE timing; 1 compile/warmup iter; then N timed
#     iters -> min / median / spread (max-min) of device-profiler kernel time.
#   - Hang-safe + resumable: one config per SUBPROCESS (a hang kills only that worker, tt-smi -r resets);
#     results cached in regime_a_bench.json, skipped on resume.
#   - Two modes per shape: PRODUCT (config=None) and BEST-MANUAL (seed from fluxltx_regimeA_sweep.json /
#     the picker, then a focused local search over (Pk,Ns,Sm,kb,nsb)).
#
# No machine-specific paths beyond the repo root (derived from this file's location).
import csv, json, os, subprocess, sys, statistics
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIN_CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
SWEEP = f"{os.path.dirname(__file__)}/fluxltx_regimeA_sweep.json"
OUT = f"{os.path.dirname(__file__)}/regime_a_bench.json"
FREQ = 1.35e9
PEAK512 = 512e9
TB = 2048
ITERS = 8  # timed steady-state iters after 1 warmup


def cdiv(a, b):
    return (a + b - 1) // b


def rup(x, y):
    return cdiv(x, y) * y


def logical_bytes(M, K, N):
    Mt, Kt, Nt = cdiv(M, 32), cdiv(K, 32), cdiv(N, 32)
    return (Mt * Kt + Kt * Nt + Mt * Nt) * TB


# ---------------------------------------------------------------- worker (one config, one process)
def parse_runs():
    """Return (per-run total kernel cycles, distinct core count). Total = max across all (core,RISC)
    KERNEL zones per run. Run 0 (warmup/compile) is dropped by the caller."""
    try:
        rows = list(csv.reader(open(BIN_CSV)))
    except Exception:
        return [], 0
    ev = defaultdict(list)
    cores = set()
    for row in rows[2:]:
        if len(row) < 12 or not row[10].strip().endswith("-KERNEL"):
            continue
        ev[(row[1], row[2], row[3], row[10].strip())].append((row[11].strip(), int(row[5])))
        cores.add((row[1], row[2], row[3]))
    dur = {}
    for k, l in ev.items():
        ds, st = [], None
        for t, c in l:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                ds.append(c - st)
                st = None
        dur[k] = ds
    if not dur:
        return [], 0
    nruns = min(len(v) for v in dur.values())
    return [max(v[i] for v in dur.values()) for i in range(nruns)], len(cores)


def worker(M, K, N, Ns, Pk, Sm, kb, nsb):
    import torch
    import ttnn
    from models.common.utility_functions import comp_pcc

    try:
        os.remove(BIN_CSV)
    except OSError:
        pass
    pcc, ok, finite = 0.0, False, False
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        ref = (t0.float() @ t1.float())[0, 0]
        in0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, dev)
        in1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wcfg)
        cfg = None
        if Pk > 0:
            cfg = ttnn.RegimeAMatmulConfig(
                k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb
            )
        # PCC check (before timing). This first op call is also the compile/warmup (run 0, dropped later).
        out = ttnn.experimental.regime_a_matmul(in0, in1, config=cfg)
        got = ttnn.to_torch(ttnn.from_device(out))[0, 0].float()
        ok, pcc = comp_pcc(ref, got, 0.999)
        finite = bool(torch.isfinite(got).all())
        for _ in range(ITERS):
            o = ttnn.experimental.regime_a_matmul(in0, in1, config=cfg)
            ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
    finally:
        ttnn.close_device(dev)  # flushes the device-profiler CSV
    runs, cores = parse_runs()  # AFTER close: CSV is fully written
    runs = runs[1:] if len(runs) > 1 else runs  # drop warmup/compile (run 0)
    print("RESULT " + json.dumps({"runs": runs, "pcc": float(pcc), "ok": bool(ok), "finite": finite, "cores": cores}))


# ---------------------------------------------------------------- driver
def run_cfg(M, K, N, cfg, cache):
    """cfg = (Ns,Pk,Sm,kb,nsb) or None (product). Returns dict or None on fail/hang."""
    key = f"{M}x{K}x{N}:" + ("auto" if cfg is None else ",".join(map(str, cfg)))
    if key in cache:
        return cache[key]
    if cfg is None:
        args = [M, K, N, 1, 0, 1, 1, 1]  # Pk=0 => config=None
    else:
        Ns, Pk, Sm, kb, nsb = cfg
        args = [M, K, N, Ns, Pk, Sm, kb, nsb]
    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TT_METAL_HOME"] = ROOT
    env["ARCH_NAME"] = "blackhole"
    env["PYTHONPATH"] = ROOT
    cmd = ["timeout", "-s", "TERM", "150", sys.executable, __file__, "--run"] + [str(a) for a in args]
    rec = None
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180, cwd=ROOT)
        for line in r.stdout.splitlines():
            if line.startswith("RESULT "):
                d = json.loads(line[7:])
                runs = d["runs"]
                if runs and d["ok"] and d["finite"]:
                    cmin = min(runs)
                    rec = {
                        "key": key,
                        "M": M,
                        "K": K,
                        "N": N,
                        "cfg": cfg,
                        "cores": d["cores"],
                        "us_min": cmin / FREQ * 1e6,
                        "us_med": statistics.median(runs) / FREQ * 1e6,
                        "us_spread_pct": (max(runs) - cmin) / cmin * 100 if cmin else 0,
                        "eff_gbps": logical_bytes(M, K, N) / (cmin / FREQ) / 1e9,
                        "pct512": logical_bytes(M, K, N) / (cmin / FREQ) / PEAK512 * 100,
                        "pcc": d["pcc"],
                        "niters": len(runs),
                    }
                elif not (d["ok"] and d["finite"]):
                    rec = {"key": key, "fail": "pcc/finite", "pcc": d.get("pcc")}
        if rec is None:
            rec = {"key": key, "fail": "no-result", "stderr": r.stderr[-200:]}
    except subprocess.TimeoutExpired:
        rec = {"key": key, "fail": "hang"}
        subprocess.run(["tt-smi", "-r"], capture_output=True)
    cache[key] = rec
    json.dump(cache, open(OUT, "w"))
    return rec


def sweep_seeds(M, K, N, sweep, topk=3):
    rows = [r for r in sweep if (r["M"], r["K"], r["N"]) == (M, K, N) and r["Sm"] == 1]
    rows.sort(key=lambda r: -r["bwp"])
    seen, out = set(), []
    for r in rows:
        c = (r["Ns"], r["Pk"], r["Sm"], r["kb"], r["nsb"])
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= topk:
            break
    return out


def neighbors(cfg):
    cfg = tuple(cfg)  # cache round-trips cfg through JSON as a list; normalize to a hashable tuple
    Ns, Pk, Sm, kb, nsb = cfg
    out = set()
    for dPk in (-2, -1, 1, 2):
        if 1 <= Pk + dPk <= 12:
            out.add((Ns, Pk + dPk, Sm, kb, nsb))
    for v in (1, 2, 3, 4, 6):
        out.add((v, Pk, Sm, kb, nsb))
    for v in (1, 2, 4, 8):
        out.add((Ns, Pk, Sm, v, nsb))
    for dn in (-1, 1, 2):
        if nsb + dn >= 1:
            out.add((Ns, Pk, Sm, kb, nsb + dn))
    out.discard(cfg)
    return list(out)


def best_manual(M, K, N, sweep, cache, picker=None):
    seeds = sweep_seeds(M, K, N, sweep)
    if not seeds and picker is not None:
        seeds = [picker(M, K, N)]
    if not seeds:
        seeds = [(1, 6, 1, 2, 1)]
    results = []
    for c in seeds:
        r = run_cfg(M, K, N, c, cache)
        if r and "us_min" in r:
            results.append(r)
    if not results:
        return None
    best = min(results, key=lambda r: r["us_min"])
    # one focused local-search round around the best seed
    for c in neighbors(best["cfg"]):
        r = run_cfg(M, K, N, c, cache)
        if r and "us_min" in r:
            results.append(r)
    return min(results, key=lambda r: r["us_min"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        worker(*[int(x) for x in sys.argv[2:10]])
        sys.exit(0)
    # driver
    sweep = json.load(open(SWEEP))
    sweep_shapes = sorted(set((r["M"], r["K"], r["N"]) for r in sweep))
    added = [
        (32, 6144, 4608),
        (256, 2048, 1024),
        (512, 6144, 768),
        (512, 15360, 768),
        (512, 6144, 2304),
        (512, 2304, 6144),
        (512, 3072, 6144),
        (512, 6144, 4608),
    ]
    balanced_tail = [
        (32, 6080, 4640),
        (64, 6080, 4640),
        (128, 6080, 4640),
        (256, 6080, 4640),  # vary M, non-div K,N
        (32, 6144, 4600),
        (32, 6100, 4608),
        (48, 6144, 4608),  # independent N / K / M sub-tile
    ]
    corpus = (
        [("sweep", s) for s in sweep_shapes]
        + [("added", s) for s in added]
        + [("balanced_tail", s) for s in balanced_tail]
    )
    cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
    picker = None
    try:
        import importlib.util, io, contextlib

        spec = importlib.util.spec_from_file_location("pv", f"{os.path.dirname(__file__)}/picker_v2.py")
        pv = importlib.util.module_from_spec(spec)
        cwd = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(pv)
        os.chdir(cwd)
        picker = lambda M, K, N: pv.pickfree(M, K, N, pv.bestP)
    except Exception as e:
        print("picker unavailable:", e)

    out = []
    for cat, (M, K, N) in corpus:
        Mt = cdiv(M, 32)
        prod = run_cfg(M, K, N, None, cache)
        # Mt>=16 is DIAGNOSTIC-ONLY (out of the Mt<=8 acceptance scope). Skip the manual local search
        # for it: several of these deep-K/large-Mt shapes have no feasible config and would burn the
        # whole budget on 180s worker timeouts. Report product only.
        man = None if Mt >= 16 else best_manual(M, K, N, sweep, cache, picker)
        rec = {"cat": cat, "M": M, "K": K, "N": N, "Mt": Mt, "diagnostic": Mt >= 16, "product": prod, "manual": man}
        out.append(rec)
        pu = prod.get("us_min") if prod else None
        mu = man.get("us_min") if man else None
        print(
            f"[{cat:13}] {M}x{K}x{N} Mt{Mt}"
            + (f"  product {pu:.1f}us {prod.get('pct512',0):.0f}%" if pu else f"  product {prod.get('fail')}")
            + (f"  manual {mu:.1f}us {man.get('pct512',0):.0f}% cfg={man.get('cfg')}" if mu else "  manual -"),
            flush=True,
        )
    json.dump({"corpus": out, "cache": cache}, open(OUT, "w"), indent=2)
    print(f"\nWROTE {OUT}")
