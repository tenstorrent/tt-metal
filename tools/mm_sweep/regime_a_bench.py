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
OUT = f"{os.path.dirname(__file__)}/regime_a_bench.json"  # finalized {corpus, cache} (driver writes)
CACHE = f"{os.path.dirname(__file__)}/regime_a_bench_cache.json"  # flat resumable cache (run_cfg writes)
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


L1_BUDGET = 1440 * 1024


def planner_feasible(M, K, N, cfg):
    """Python mirror of the C++ host planner build_plan() rejects. Returns (ok, reason). Lets the driver
    reject invalid configs WITHOUT launching a device subprocess (avoids 150s timeouts on infeasible
    configs). Kept in lock-step with regime_a_matmul_plan.hpp."""
    Ns, Pk, Sm, kb, nsb = cfg
    Mt, Kt, Nt = cdiv(M, 32), cdiv(K, 32), cdiv(N, 32)
    if Pk < 1 or Ns < 1 or Sm < 1 or kb < 1 or nsb < 1:
        return False, "range"
    if Sm > Mt:
        return False, "Sm>Mt"
    if Pk > Kt:
        return False, "Pk>Kt"
    N_band = cdiv(Nt, 8)
    if 7 * N_band >= Nt:
        return False, "empty-bank"  # width shard leaves some banks wholly-pad
    if (Nt - 7 * N_band) < Ns:
        return False, "empty-n-slice"  # smallest bank (7) can't feed Ns owners
    cores = 8 * Pk * Ns * Sm
    if cores > 104:
        return False, "cores>104"
    Ktl = rup(cdiv(Kt, Pk), kb * 8)
    Mblk = cdiv(Mt, Sm)
    N_own = cdiv(N_band, Ns)
    if nsb > N_own:
        return False, "nsb>N_own"
    N_bpc = cdiv(N_own, nsb)
    N_slice = N_bpc * nsb
    cb0, cb1, cb2, cb3 = Ktl * Mblk, 4 * kb * nsb, 2 * Mblk * nsb, Mblk * nsb
    cb7 = 2 * Mblk * nsb if Pk > 1 else 0
    l1 = (cb0 + cb1 + cb2 + cb7) * TB + cb3 * 4096
    if l1 > L1_BUDGET:
        return False, "L1"
    return True, "ok"


_PICKERS = {}


def _pickers():
    if not _PICKERS:
        import importlib.util, io, contextlib

        for name in ("picker_table", "picker_v2"):
            spec = importlib.util.spec_from_file_location(name, f"{os.path.dirname(__file__)}/{name}.py")
            m = importlib.util.module_from_spec(spec)
            cwd = os.getcwd()
            os.chdir(os.path.dirname(__file__))
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    spec.loader.exec_module(m)
            finally:
                os.chdir(cwd)
            _PICKERS[name] = m
    return _PICKERS


def auto_config(M, K, N):
    """Mirror of C++ auto_select_config: element-keyed lookup table else v2 cost-model fallback."""
    p = _pickers()
    tbl = p["picker_table"].PICKER_TABLE
    if (M, K, N) in tbl:
        return tuple(tbl[(M, K, N)])
    pv = p["picker_v2"]
    return tuple(pv.pickfree(M, K, N, pv.bestP))


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
def _metrics(M, K, N, cfg, d):
    """Build a success record. Rank by MEDIAN; keep min/spread. cores = 8*Pk*Ns*Sm (config-derived);
    for product (cfg resolved by driver) the resolved cfg is passed in."""
    runs = d["runs"]
    cmed = statistics.median(runs)
    cmin = min(runs)
    Ns, Pk, Sm, kb, nsb = cfg
    return {
        "M": M,
        "K": K,
        "N": N,
        "cfg": list(cfg),
        "cores": 8 * Pk * Ns * Sm,
        "cls": "ok",
        "us_med": cmed / FREQ * 1e6,
        "us_min": cmin / FREQ * 1e6,
        "us_spread_pct": (max(runs) - cmin) / cmin * 100 if cmin else 0,
        "eff_gbps": logical_bytes(M, K, N) / (cmed / FREQ) / 1e9,
        "pct512": logical_bytes(M, K, N) / (cmed / FREQ) / PEAK512 * 100,
        "pct512_min": logical_bytes(M, K, N) / (cmin / FREQ) / PEAK512 * 100,
        "pcc": d["pcc"],
        "niters": len(runs),
    }


def run_cfg(M, K, N, cfg, cache):
    """cfg = (Ns,Pk,Sm,kb,nsb) or None (product). Returns a record with a 'cls' outcome class:
    validation | runtime | hang | pcc | ok. Feasibility is checked HERE (host planner mirror) so
    invalid configs never launch a device subprocess."""
    key = f"{M}x{K}x{N}:" + ("auto" if cfg is None else ",".join(map(str, cfg)))
    if key in cache:
        return cache[key]
    # Manual configs: reject invalid ones before launching (product resolves in C++, always feasible or
    # it FATALs -> classified 'runtime').
    if cfg is not None:
        ok, why = planner_feasible(M, K, N, cfg)
        if not ok:
            rec = {"key": key, "cls": "validation", "reason": why, "cfg": list(cfg)}
            cache[key] = rec
            json.dump(cache, open(CACHE, "w"))
            return rec
    args = [M, K, N, 1, 0, 1, 1, 1] if cfg is None else [M, K, N, *cfg[:1], cfg[1], cfg[2], cfg[3], cfg[4]]
    if cfg is not None:
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
                rcfg = cfg if cfg is not None else auto_config(M, K, N)  # product: resolved config
                if d["runs"] and d["ok"] and d["finite"]:
                    rec = _metrics(M, K, N, rcfg, d)
                    rec["key"] = key
                else:
                    rec = {"key": key, "cls": "pcc", "pcc": d.get("pcc"), "finite": d.get("finite"), "cfg": list(rcfg)}
        if rec is None:
            rec = {"key": key, "cls": "runtime", "stderr": r.stderr[-300:]}
    except subprocess.TimeoutExpired:
        rec = {"key": key, "cls": "hang"}
        subprocess.run(["tt-smi", "-r"], capture_output=True)
    cache[key] = rec
    json.dump(cache, open(CACHE, "w"))
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


def _ok(r):
    return bool(r) and r.get("cls") == "ok"


def best_manual(M, K, N, sweep, cache, picker=None):
    """Seed from the prototype sweep (else picker) + one focused local-search round. Ranks by MEDIAN."""
    seeds = sweep_seeds(M, K, N, sweep)
    if picker is not None:
        seeds = seeds + [picker(M, K, N)]
    if not seeds:
        seeds = [(1, 6, 1, 2, 1)]
    results = [r for c in seeds if _ok(r := run_cfg(M, K, N, c, cache))]
    if not results:
        return None
    best = min(results, key=lambda r: r["us_med"])
    for c in neighbors(best["cfg"]):
        r = run_cfg(M, K, N, c, cache)
        if _ok(r):
            results.append(r)
    return min(results, key=lambda r: r["us_med"])


def enumerate_feasible(M, K, N):
    """Every planner-valid config across ALL 5 levers within the 104-worker + L1 limits (NOT Sm=1 only)."""
    Mt, Kt, Nt = cdiv(M, 32), cdiv(K, 32), cdiv(N, 32)
    N_band = cdiv(Nt, 8)
    cfgs = []
    for Pk in range(1, min(Kt, 13) + 1):
        for Ns in range(1, 7):
            N_own = cdiv(N_band, Ns)
            for Sm in range(1, Mt + 1):
                if 8 * Pk * Ns * Sm > 104:
                    continue
                for kb in (1, 2, 4, 8, 16, 32):
                    for nsb in range(1, N_own + 1):
                        c = (Ns, Pk, Sm, kb, nsb)
                        if planner_feasible(M, K, N, c)[0]:
                            cfgs.append(c)
    return cfgs


def load_cache():
    """Genuinely resumable: the flat CACHE file (written per config by run_cfg). Falls back to a
    finalized {corpus, cache} OUT only if it carries new-format ('cls') records."""
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    if os.path.exists(OUT):
        d = json.load(open(OUT))
        c = d.get("cache") if isinstance(d, dict) else None
        if isinstance(c, dict) and any(isinstance(v, dict) and "cls" in v for v in c.values()):
            return c
    return {}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        worker(*[int(x) for x in sys.argv[2:10]])
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        M, K, N = (int(x) for x in sys.argv[2:5])
        cache = load_cache()
        cfgs = enumerate_feasible(M, K, N)
        print(f"exhaustive sweep {M}x{K}x{N}: {len(cfgs)} planner-valid configs", flush=True)
        res = []
        for i, c in enumerate(cfgs):
            r = run_cfg(M, K, N, c, cache)
            if _ok(r):
                res.append(r)
                if r["us_med"] <= min(x["us_med"] for x in res):
                    print(
                        f"  [{i+1}/{len(cfgs)}] NEW BEST cfg={c} med={r['us_med']:.1f}us "
                        f"{r['pct512']:.1f}% cores={r['cores']}",
                        flush=True,
                    )
        res.sort(key=lambda r: r["us_med"])
        swout = f"{os.path.dirname(__file__)}/regime_a_sweep_{M}x{K}x{N}.json"
        json.dump({"M": M, "K": K, "N": N, "results": res, "n_valid": len(cfgs)}, open(swout, "w"), indent=2)
        print(f"\nTOP 8 (by median us):")
        for r in res[:8]:
            print(
                f"  cfg={r['cfg']} med={r['us_med']:.1f}us min={r['us_min']:.1f} "
                f"spread={r['us_spread_pct']:.1f}% {r['pct512']:.1f}% cores={r['cores']} pcc={r['pcc']:.5f}"
            )
        print(f"WROTE {swout}")
        sys.exit(0)

    # driver (full corpus)
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
        (256, 6080, 4640),
        (32, 6144, 4600),
        (32, 6100, 4608),
        (48, 6144, 4608),
    ]
    corpus = (
        [("sweep", s) for s in sweep_shapes]
        + [("added", s) for s in added]
        + [("balanced_tail", s) for s in balanced_tail]
    )
    cache = load_cache()
    picker = auto_config

    out = []
    for cat, (M, K, N) in corpus:
        Mt = cdiv(M, 32)
        prod = run_cfg(M, K, N, None, cache)
        man = None if Mt >= 16 else best_manual(M, K, N, sweep, cache, picker)
        rec = {"cat": cat, "M": M, "K": K, "N": N, "Mt": Mt, "diagnostic": Mt >= 16, "product": prod, "manual": man}
        out.append(rec)
        pcls, pu = (prod or {}).get("cls"), (prod or {}).get("us_med")
        mu = man.get("us_med") if _ok(man) else None
        print(
            f"[{cat:13}] {M}x{K}x{N} Mt{Mt}"
            + (
                f"  product {pu:.1f}us {prod.get('pct512',0):.0f}% cfg={prod.get('cfg')}"
                if _ok(prod)
                else f"  product [{pcls}]"
            )
            + (f"  manual {mu:.1f}us {man.get('pct512',0):.0f}% cfg={man.get('cfg')}" if mu else "  manual -"),
            flush=True,
        )
    json.dump({"corpus": out, "cache": cache}, open(OUT, "w"), indent=2)
    print(f"\nWROTE {OUT}")
