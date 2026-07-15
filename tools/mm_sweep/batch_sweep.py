#!/usr/bin/env python3
# Chunked-batch sweep worker: opens the device ONCE per chunk and loops many configs, instead of one
# subprocess per config. in0/in1 are allocated once per shape (they depend only on M,K,N; the config only
# changes the program), and the device profiler is drained per config. Records are identical to
# regime_a_bench.run_cfg (us_med/us_min/spread/pct512/eff_gbps/pcc/cores/cfg/cls) and written to the SAME
# CacheStore, so this is a drop-in accelerator for the per-subprocess path and shares its cache.
#
# Hang-safety is preserved at the CHUNK level: the parent runs each chunk under a wall-clock timeout; on a
# death it resets the device and resumes (the worker skips already-cached configs). A config that kills its
# chunk before reporting is marked 'hang' so the sweep still makes forward progress.
#
# Worker:  python3 batch_sweep.py --worker M K N <cfglist.json> <cachepath>
# Parent:  python3 batch_sweep.py fill        (fills the mt8 characterization cache, chunked + resumable)
import json, os, subprocess, sys, statistics

import regime_a_bench as b

HERE = os.path.dirname(__file__)
CHUNK = 100
CACHE = f"{HERE}/mt8_characterize_cache.json"


# ---------------------------------------------------------------- worker (process alive; device reopened
# per config to flush the profiler CSV — mid-session ReadDeviceProfiler does NOT write it). The expensive
# `import ttnn` (~4.7s) is paid ONCE per process; per-config cost is open+alloc+close (~2s). IDENTICAL
# CSV kernel-zone methodology to regime_a_bench.worker -> fully comparable to the per-subprocess cache.
def worker(M, K, N, cfg_list, cache_path):
    import torch
    import ttnn
    from models.common.utility_functions import comp_pcc

    store = b.CacheStore(cache_path)
    torch.manual_seed(0)
    t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    ref = (t0.float() @ t1.float())[0, 0]
    for cfg in cfg_list:
        cfg = None if cfg is None else tuple(cfg)
        key = f"{M}x{K}x{N}:" + ("auto" if cfg is None else ",".join(map(str, cfg)))
        if key in store:
            continue
        if cfg is not None and not b.planner_feasible(M, K, N, cfg)[0]:
            store.put(
                key, {"key": key, "cls": "validation", "reason": b.planner_feasible(M, K, N, cfg)[1], "cfg": list(cfg)}
            )
            continue
        rcfg = cfg if cfg is not None else b.auto_config(M, K, N)
        ccfg = None
        if cfg is not None:
            Ns, Pk, Sm, kb, nsb = cfg
            ccfg = ttnn.RegimeAMatmulConfig(
                k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb
            )
        try:
            os.remove(b.BIN_CSV)
        except OSError:
            pass
        dev = ttnn.open_device(device_id=0)
        pcc, ok, finite = 0.0, False, False
        try:
            in0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, dev)
            in1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wcfg)
            out = ttnn.experimental.regime_a_matmul(in0, in1, config=ccfg)  # compile/warmup (run 0)
            got = ttnn.to_torch(ttnn.from_device(out))[0, 0].float()
            ok, pcc = comp_pcc(ref, got, 0.999)
            finite = bool(torch.isfinite(got).all())
            for _ in range(b.ITERS):
                o = ttnn.experimental.regime_a_matmul(in0, in1, config=ccfg)
                ttnn.synchronize_device(dev)
            ttnn.ReadDeviceProfiler(dev)
        finally:
            ttnn.close_device(dev)  # flushes profile_log_device.csv
        runs, _ = b.parse_runs()  # AFTER close: CSV is written
        runs = runs[1:] if len(runs) > 1 else runs  # drop warmup
        if runs and ok and finite:
            rec = b._metrics(M, K, N, rcfg, {"runs": runs, "pcc": float(pcc)})
            rec["key"] = key
        else:
            rec = {"key": key, "cls": "pcc", "pcc": float(pcc), "finite": finite, "cfg": list(rcfg)}
        store.put(key, rec)
        print(f"BATCHDONE {key} {rec['cls']} {rec.get('us_med', 0):.1f}", flush=True)


# ---------------------------------------------------------------- parent (chunk + hang-safe)
def _env():
    e = dict(os.environ)
    e.update(TT_METAL_DEVICE_PROFILER="1", TT_METAL_HOME=b.ROOT, ARCH_NAME="blackhole", PYTHONPATH=b.ROOT)
    return e


def run_chunks(M, K, N, cfgs, cache_path):
    """cfgs: list of configs (None allowed). Runs uncached ones in device-open-once chunks; resumable."""
    store = b.CacheStore(cache_path)

    def keyof(c):
        return f"{M}x{K}x{N}:" + ("auto" if c is None else ",".join(map(str, c)))

    pending = [c for c in cfgs if keyof(c) not in store]
    while pending:
        chunk = pending[:CHUNK]
        tf = f"{HERE}/.batch_chunk_{os.getpid()}.json"
        json.dump(chunk, open(tf, "w"))
        cmd = [
            "timeout",
            "-s",
            "TERM",
            str(90 + 4 * len(chunk)),
            sys.executable,
            __file__,
            "--worker",
            str(M),
            str(K),
            str(N),
            tf,
            cache_path,
        ]
        r = subprocess.run(cmd, env=_env(), cwd=b.ROOT, capture_output=True, text=True)
        store = b.CacheStore(cache_path)  # reload what the worker persisted
        still = [c for c in pending if keyof(c) not in store]
        advanced = len(pending) - len(still)
        if b.classify_timeout(r.returncode) or r.returncode != 0:
            b._reset_device()
            if advanced == 0 and still:  # the head config wedged the chunk before any result -> mark hang
                hk = keyof(still[0])
                store.put(hk, {"key": hk, "cls": "hang", "cfg": (None if still[0] is None else list(still[0]))})
                store = b.CacheStore(cache_path)
                still = [c for c in pending if keyof(c) not in store]
            print(
                f"  [chunk died rc={r.returncode}] advanced {advanced}; {len(still)} left " f"{r.stderr[-160:]}",
                flush=True,
            )
        else:
            print(f"  [chunk ok] advanced {advanced}; {len(still)} left", flush=True)
        pending = still
        try:
            os.remove(tf)
        except OSError:
            pass


def fill():
    prim = [(256, 2048, 1024), (256, 6144, 768), (256, 6144, 2304), (256, 6144, 4608)]
    mscale = [
        (32, 2048, 1024),
        (64, 2048, 1024),
        (128, 2048, 1024),
        (32, 6144, 4608),
        (64, 6144, 4608),
        (128, 6144, 4608),
    ]
    for M, K, N in prim:
        cfgs = [None] + [list(c) for c in b.enumerate_feasible(M, K, N, kb_set=(1, 2, 4, 8), nsb_max=8)]
        print(f"[fill primary] {M}x{K}x{N}: {len(cfgs)} configs", flush=True)
        run_chunks(M, K, N, cfgs, CACHE)
    for M, K, N in mscale:
        # M-scaling only needs config=None + a focused best-manual neighborhood (NOT a full sweep): the
        # auto pick + its lever neighbors. characterize_mt8's best_manual refines from cache (few misses).
        auto = list(b.auto_config(M, K, N))
        cand = {tuple(auto)} | {tuple(c) for c in b.neighbors(auto) if b.planner_feasible(M, K, N, c)[0]}
        cfgs = [None] + [list(c) for c in cand]
        print(f"[fill mscale] {M}x{K}x{N}: {len(cfgs)} configs (auto+neighbors)", flush=True)
        run_chunks(M, K, N, cfgs, CACHE)
    print("FILL DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        M, K, N = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        cfg_list = json.load(open(sys.argv[5]))
        worker(M, K, N, cfg_list, sys.argv[6])
    elif sys.argv[1] == "fill":
        fill()
    elif sys.argv[1] == "validate":
        # re-run a few ALREADY-cached configs into a throwaway cache; compare medians.
        M, K, N = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        cfgs = [json.loads(sys.argv[i]) for i in range(5, len(sys.argv))]
        tmp = f"{HERE}/.batch_validate.json"
        if os.path.exists(tmp):
            os.remove(tmp)
        tf = f"{HERE}/.batch_vcfgs.json"
        json.dump(cfgs, open(tf, "w"))
        subprocess.run(
            ["timeout", "300", sys.executable, __file__, "--worker", str(M), str(K), str(N), tf, tmp],
            env=_env(),
            cwd=b.ROOT,
        )
        new = b.CacheStore(tmp)
        old = b.CacheStore(CACHE)
        for c in cfgs:
            k = f"{M}x{K}x{N}:" + ("auto" if c is None else ",".join(map(str, c)))
            no = new.get(k, {}).get("us_med")
            oo = old.get(k, {}).get("us_med")
            d = (no / oo - 1) * 100 if (no and oo) else None
            print(f"{k}: batch={no} per-subproc={oo} delta={d if d is None else round(d,1)}%")
