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


def parse_runs_detail():
    """Like parse_runs but ALSO per-RISC critical span + per-core span spread (warmup run 0 dropped here
    when >1 run, so callers must NOT slice again). RISC = col3 (TRISC_0/1/2 collapsed to TRISC by max).
    Returns (runs, ncores, per_risc_us, core_spread_pct):
      runs           = per-run total kernel cycles (max over all (core,RISC)), warmup dropped
      ncores         = distinct (core_x,core_y) with a KERNEL zone
      per_risc_us    = {risc: median-over-runs of (max-over-cores span) in us} -- which RISC is critical
      core_spread_pct= median-over-runs of (max_core-min_core)/min_core*100 (core = max over its RISCs)"""
    try:
        rows = list(csv.reader(open(BIN_CSV)))
    except Exception:
        return [], 0, {}, None
    ev = defaultdict(list)  # (x,y,risc) -> [(START/END, cycle)]
    for row in rows[2:]:
        if len(row) < 12 or not row[10].strip().endswith("-KERNEL"):
            continue
        ev[(row[1], row[2], row[3])].append((row[11].strip(), int(row[5])))
    span = {}  # (x,y,risc) -> [span per run]
    for k, l in ev.items():
        ds, st = [], None
        for t, c in l:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                ds.append(c - st)
                st = None
        span[k] = ds
    if not span:
        return [], 0, {}, None
    nruns = min(len(v) for v in span.values())
    drop = 1 if nruns > 1 else 0  # warmup/compile run 0
    idx = range(drop, nruns)
    runs = [max(v[i] for v in span.values()) for i in idx]
    ncores = len({(x, y) for (x, y, _r) in span})

    def fam(r):
        return "TRISC" if r.startswith("TRISC") else r

    risc_run = defaultdict(lambda: {i: 0 for i in idx})
    core_run = defaultdict(lambda: {i: 0 for i in idx})
    for (x, y, r), ds in span.items():
        f = fam(r)
        for i in idx:
            risc_run[f][i] = max(risc_run[f][i], ds[i])
            core_run[(x, y)][i] = max(core_run[(x, y)][i], ds[i])
    per_risc_us = {f: statistics.median(list(v.values())) / FREQ * 1e6 for f, v in risc_run.items()}
    spreads = []
    for i in idx:
        vals = [core_run[c][i] for c in core_run]
        mn, mx = min(vals), max(vals)
        if mn > 0:
            spreads.append((mx - mn) / mn * 100)
    core_spread_pct = statistics.median(spreads) if spreads else None
    return runs, ncores, per_risc_us, core_spread_pct


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
    runs, cores, per_risc_us, core_spread_pct = parse_runs_detail()  # AFTER close; warmup already dropped
    print(
        "RESULT "
        + json.dumps(
            {
                "runs": runs,
                "pcc": float(pcc),
                "ok": bool(ok),
                "finite": finite,
                "cores": cores,
                "per_risc_us": per_risc_us,
                "core_spread_pct": core_spread_pct,
            }
        )
    )


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
        "per_risc_us": d.get("per_risc_us"),
        "core_spread_pct": d.get("core_spread_pct"),
    }


INNER_TIMEOUT = 150  # GNU `timeout` on the worker subprocess
OUTER_TIMEOUT = 180  # subprocess.run watchdog (fires only if the inner timeout itself wedges)


def classify_timeout(returncode):
    """True if the worker process ended by our timeout/kill (a HANG), not a normal crash. GNU `timeout`
    returns 124 when it times out the command; 137 = SIGKILL (128+9, e.g. escalation / OOM-kill); a
    negative code means Python's immediate child was signal-terminated. SIGABRT/SIGSEGV (134/139) are
    genuine crashes -> NOT a timeout, so they stay 'runtime'."""
    return returncode == 124 or returncode == 137 or returncode < 0


class CacheStore:
    """cfg-key -> record map. Persistent ONLY when bound to a path; a path-less store (the default for
    interactive callers) never touches disk, so an ad-hoc run cannot clobber a sweep's cache. On persist
    it MERGES with the current on-disk cache (so a concurrent/earlier writer's records survive) and writes
    ATOMICALLY (temp file + os.replace). A plain dict is also accepted by run_cfg and is treated as a
    path-less (non-persistent) store."""

    def __init__(self, path=None):
        self.path = path
        self.data = {}
        if path and os.path.exists(path):
            try:
                self.data = json.load(open(path))
            except Exception:
                self.data = {}

    def __contains__(self, k):
        return k in self.data

    def __getitem__(self, k):
        return self.data[k]

    def get(self, k, default=None):
        return self.data.get(k, default)

    def values(self):
        return self.data.values()

    def put(self, key, rec):
        self.data[key] = rec
        if not self.path:
            return  # non-persistent
        merged = {}
        if os.path.exists(self.path):
            try:
                merged = json.load(open(self.path))
            except Exception:
                merged = {}
        merged.update(self.data)  # our records win over any stale on-disk copy of the same keys
        tmp = f"{self.path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(merged, f)
        os.replace(tmp, self.path)  # atomic on POSIX
        self.data = merged


def _cache_put(cache, key, rec):
    """Persist through a CacheStore; a plain dict stays in-memory only (never writes CACHE)."""
    if hasattr(cache, "put"):
        cache.put(key, rec)
    else:
        cache[key] = rec
    return rec


def _reset_device():
    subprocess.run(["tt-smi", "-r"], capture_output=True)


def run_cfg(M, K, N, cfg, cache):
    """cfg = (Ns,Pk,Sm,kb,nsb) or None (product). Returns a record with a 'cls' outcome class:
    validation | runtime | hang | pcc | ok. Feasibility is checked HERE (host planner mirror) so
    invalid configs never launch a device subprocess. `cache` may be a CacheStore (persists to its bound
    path, merge+atomic) or a plain dict (in-memory only -> an interactive run_cfg(...,{}) cannot clobber
    the sweep cache). Inner GNU-timeout (124/kill) and the outer watchdog both -> 'hang' + device reset."""
    key = f"{M}x{K}x{N}:" + ("auto" if cfg is None else ",".join(map(str, cfg)))
    if key in cache:
        return cache[key]
    # Manual configs: reject invalid ones before launching (product resolves in C++, always feasible or
    # it FATALs -> classified 'runtime').
    if cfg is not None:
        ok, why = planner_feasible(M, K, N, cfg)
        if not ok:
            return _cache_put(cache, key, {"key": key, "cls": "validation", "reason": why, "cfg": list(cfg)})
    if cfg is None:
        args = [M, K, N, 1, 0, 1, 1, 1]
    else:
        Ns, Pk, Sm, kb, nsb = cfg
        args = [M, K, N, Ns, Pk, Sm, kb, nsb]
    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TT_METAL_HOME"] = ROOT
    env["ARCH_NAME"] = "blackhole"
    env["PYTHONPATH"] = ROOT
    cmd = ["timeout", "-s", "TERM", str(INNER_TIMEOUT), sys.executable, __file__, "--run"] + [str(a) for a in args]
    rec = None
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=OUTER_TIMEOUT, cwd=ROOT)
        if classify_timeout(r.returncode):  # inner GNU-timeout / kill BEFORE parsing stdout
            _reset_device()
            return _cache_put(cache, key, {"key": key, "cls": "hang", "returncode": r.returncode})
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
            rec = {"key": key, "cls": "runtime", "returncode": r.returncode, "stderr": r.stderr[-300:]}
    except subprocess.TimeoutExpired:  # outer watchdog: the inner timeout itself wedged
        _reset_device()
        rec = {"key": key, "cls": "hang", "returncode": "outer-timeout"}
    return _cache_put(cache, key, rec)


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


def enumerate_feasible(M, K, N, kb_set=(1, 2, 4, 8, 16, 32), nsb_max=None):
    """Planner-valid configs within the 104-worker + L1 limits (NOT Sm=1 only). Default = ALL 5 levers
    (kb up to 32, every nsb). Pass kb_set / nsb_max to restrict to a documented practical sub-domain."""
    Mt, Kt, Nt = cdiv(M, 32), cdiv(K, 32), cdiv(N, 32)
    N_band = cdiv(Nt, 8)
    cfgs = []
    for Pk in range(1, min(Kt, 13) + 1):
        for Ns in range(1, 7):
            N_own = cdiv(N_band, Ns)
            nmax = N_own if nsb_max is None else min(N_own, nsb_max)
            for Sm in range(1, Mt + 1):
                if 8 * Pk * Ns * Sm > 104:
                    continue
                for kb in kb_set:
                    for nsb in range(1, nmax + 1):
                        c = (Ns, Pk, Sm, kb, nsb)
                        if planner_feasible(M, K, N, c)[0]:
                            cfgs.append(c)
    return cfgs


def plan_metrics(M, K, N, cfg):
    """Host-planner geometry for a config: L1 bytes, per-CB tile counts, schedule capacities/padding, and
    the DRAM-BW reference points (logical bytes, ideal time @512 GB/s). Mirrors build_plan()/planner_feasible."""
    Ns, Pk, Sm, kb, nsb = cfg
    Mt, Kt, Nt = cdiv(M, 32), cdiv(K, 32), cdiv(N, 32)
    N_band = cdiv(Nt, 8)
    Ktl = rup(cdiv(Kt, Pk), kb * 8)  # K-slice capacity (padded up to kb*8)
    Mblk = cdiv(Mt, Sm)
    N_own = cdiv(N_band, Ns)
    N_bpc = cdiv(N_own, nsb)
    N_slice = N_bpc * nsb
    cb0, cb1, cb2, cb3 = Ktl * Mblk, 4 * kb * nsb, 2 * Mblk * nsb, Mblk * nsb
    cb7 = 2 * Mblk * nsb if Pk > 1 else 0
    l1 = (cb0 + cb1 + cb2 + cb7) * TB + cb3 * 4096
    lb = logical_bytes(M, K, N)
    return {
        "cores": 8 * Pk * Ns * Sm,
        "Ktl": Ktl,
        "Mblk": Mblk,
        "N_own": N_own,
        "N_bpc": N_bpc,
        "N_slice": N_slice,
        "k_pad_tiles": Ktl * Pk - Kt,  # split-K padding of the K dim (schedule capacity - real)
        "cb0_in1": cb0,
        "cb1_in0": cb1,
        "cb2_interm": cb2,
        "cb3_out": cb3,
        "cb7_reduce": cb7,
        "l1_bytes": l1,
        "l1_pct": l1 / L1_BUDGET * 100,
        "logical_bytes": lb,
        "ideal_us_512": lb / PEAK512 * 1e6,
    }


def load_cache(path=CACHE, persist=True):
    """Return a resumable CacheStore bound to `path` (the flat CACHE, written per config by run_cfg).
    Seeds from a finalized {corpus, cache} OUT only when no CACHE exists yet AND OUT carries new-format
    ('cls') records. persist=False gives a path-less (non-persistent) store for interactive use."""
    store = CacheStore(path if persist else None)
    if persist and not os.path.exists(path) and os.path.exists(OUT):
        d = json.load(open(OUT))
        c = d.get("cache") if isinstance(d, dict) else None
        if isinstance(c, dict) and any(isinstance(v, dict) and "cls" in v for v in c.values()):
            store.data = dict(c)  # seed in-memory; first put() will persist to CACHE (never to OUT)
    return store


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
    # OUT is the finalized report; run_cfg only ever writes CACHE, so OUT is written here exactly once.
    json.dump({"corpus": out, "cache": cache.data}, open(OUT, "w"), indent=2)
    print(f"\nWROTE {OUT}")
