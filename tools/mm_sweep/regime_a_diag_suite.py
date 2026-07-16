#!/usr/bin/env python3
# Driver for the regime_a_matmul test-only diagnostic ablations. Launches the ttnn gtest
# `unit_tests_ttnn --gtest_filter=RegimeADiagFixture.Run` once per (shape, config, mask) with the params in
# env vars; the test drives the INTERNAL ttnn::prim::regime_a_matmul_diag entry (mask in the hashed op
# param, never Python/nanobind). One config per process => the device-profiler CSV flushes on TearDown and
# is parsed here (per-RISC + wall). Ablations are non-additive critical-path counterfactuals.
#
# Modes:
#   smoke                 -> run every mask once on the 256x2048x1024 winner (no-deadlock check)
#   matrix                -> 4 shapes x {winner, best pure-K, best KxM, best KxN} x 8 ablations, 3x relaunch
#   mscale                -> full + key ablations over M={32,64,128,256} x {(2048,1024),(6144,4608)}
import json, os, statistics, subprocess, sys

sys.path.insert(0, os.path.dirname(__file__))
import oracle_ablate as oa  # reuse parse_csv (per-RISC + wall from the profiler CSV)
import regime_a_bench as rb  # reuse classify_timeout

HERE = os.path.dirname(__file__)
ROOT = oa.ROOT
BIN = f"{ROOT}/build_Release/test/ttnn/unit_tests_ttnn"
FREQ = 1.35e9
PEAK512 = 512e9

# (name, mask). LOCAL_FEED (16) is intentionally not implemented yet (see MT8_FINDINGS).
ABLATIONS = [
    ("full", 0),
    ("skipin1", 1),
    ("skipin0", 2),
    ("skipfwd", 4),
    ("noreduce", 8),
    ("skipin0+in1", 3),
    ("skipin0+in1+noreduce", 11),
]
PRIMARY = [(256, 2048, 1024), (256, 6144, 768), (256, 6144, 2304), (256, 6144, 4608)]


def cdiv(a, b):
    return (a + b - 1) // b


def ideal_us(M, K, N):
    Mt, Kt, Nt = cdiv(M, 32), cdiv(K, 32), cdiv(N, 32)
    return (Mt * Kt + Kt * Nt + Mt * Nt) * 2048 / PEAK512 * 1e6


def _reset():
    subprocess.run(["tt-smi", "-r"], capture_output=True)


def run_one(M, K, N, cfg, mask, iters=8, timeout=150):
    """Launch the gtest for one (shape,cfg,mask); parse per-RISC + wall from the profiler CSV. Hang-safe."""
    Ns, Pk, Sm, kb, nsb = cfg
    try:
        os.remove(oa.BIN_CSV)
    except OSError:
        pass
    env = dict(os.environ)
    env.update(
        TT_METAL_DEVICE_PROFILER="1",
        TT_METAL_HOME=ROOT,
        ARCH_NAME="blackhole",
        RA_M=str(M),
        RA_K=str(K),
        RA_N=str(N),
        RA_NS=str(Ns),
        RA_PK=str(Pk),
        RA_SM=str(Sm),
        RA_KB=str(kb),
        RA_NSB=str(nsb),
        RA_MASK=str(mask),
        RA_ITERS=str(iters),
        TT_MM_RINGCOST="1",  # emit the factory's RINGCOST route-cost line (ring-order diag); harmless otherwise
    )
    cmd = ["timeout", "-s", "TERM", str(timeout), BIN, "--gtest_filter=RegimeADiagFixture.Run"]
    try:
        r = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-x", "unit_tests_ttnn"], capture_output=True)
        _reset()
        return {"cfg": list(cfg), "mask": mask, "ok": False, "cls": "hang", "wall_us": None}
    done = "DIAGDONE" in r.stdout
    passed = "[  PASSED  ]" in r.stdout
    if rb.classify_timeout(r.returncode) or not done:
        _reset()
        return {
            "cfg": list(cfg),
            "mask": mask,
            "ok": False,
            "cls": "hang" if rb.classify_timeout(r.returncode) else "fail",
            "rc": r.returncode,
            "wall_us": None,
            "stderr": r.stderr[-300:],
        }
    by, wall, _ = oa.parse_csv()
    wall_us = statistics.median(wall) / FREQ * 1e6 if wall else None
    risc = {
        t: {
            "n": len(c),
            "max_us": max(c) / FREQ * 1e6,
            "med_us": statistics.median(c) / FREQ * 1e6,
            "min_us": min(c) / FREQ * 1e6,
        }
        for t, c in sorted(by.items())
    }
    maxrel = None
    ringcost = []  # RINGCOST lines (factory, ring-order diag): per-group bank/greedy/opt max+total edge cost
    for line in r.stdout.splitlines():
        if "DIAGPCC" in line:
            maxrel = float(line.split("max_rel_err=")[1])
        elif "RINGCOST" in line:
            g = {}
            for tok in line.split():
                if tok.startswith(("group=", "wnoc=", "sel=", "Sm=", "sel_perring=")):
                    k, v = tok.split("=", 1)
                    g[k] = v
                for od in ("bank", "greedy", "mm0", "agg"):
                    # token form: <od>[perm]aggmax=<M>aggtot=<T>
                    if tok.startswith(od + "[") and "aggmax=" in tok and "aggtot=" in tok:
                        g[od + "_aggmax"] = int(tok.split("aggmax=")[1].split("aggtot=")[0])
                        g[od + "_aggtot"] = int(tok.split("aggtot=")[1])
            if "group" in g:
                ringcost.append(g)
    # masks 0 (public path) and the correct in0-delivery variants (32=scatter, 64=repl2, 128=repl4) are
    # correctness-checked by the gtest -> require the PASS; the pure ablations produce garbage, not checked.
    checked = mask in (0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
    return {
        "cfg": list(cfg),
        "mask": mask,
        "ok": bool(wall_us) and (not checked or passed),
        "cls": "ok",
        "wall_us": wall_us,
        "risc": risc,
        "max_rel_err": maxrel,
        "ringcost": ringcost,
    }


def _load_sweep(M, K, N):
    """Load the practical sweep results, transparently handling the committed .json.gz (a fresh checkout
    only has the gzipped artifact; the raw .json is gitignored)."""
    base = f"{HERE}/regime_a_sweep_{M}x{K}x{N}.json"
    if os.path.exists(base):
        return json.load(open(base))
    if os.path.exists(base + ".gz"):
        import gzip

        with gzip.open(base + ".gz", "rt") as f:
            return json.load(f)
    return None


def pick_configs(M, K, N):
    """winner + best pure-K / KxM / KxN from the practical sweep JSON (falls back to auto if absent)."""
    d = _load_sweep(M, K, N)
    picks = {}
    if d is not None:
        res = [r for r in d["results"] if r.get("cls") == "ok"]
        res.sort(key=lambda r: r["us_med"])

        def best(pred):
            for r in res:
                if pred(tuple(r["cfg"])):
                    return tuple(r["cfg"])
            return None

        picks["winner"] = tuple(res[0]["cfg"]) if res else None
        picks["pureK"] = best(lambda c: c[0] == 1 and c[2] == 1)
        picks["KxM"] = best(lambda c: c[2] > 1 and c[0] == 1)
        picks["KxN"] = best(lambda c: c[0] > 1 and c[2] == 1)
    # de-dup while keeping labels
    out = {}
    for label, c in picks.items():
        if c is not None:
            out.setdefault(c, []).append(label)
    return out  # {cfg: [labels]}


def smoke():
    M, K, N, cfg = 256, 2048, 1024, (1, 4, 2, 2, 2)
    for name, mask in ABLATIONS:
        r = run_one(M, K, N, cfg, mask)
        w = r.get("wall_us")
        print(
            f"[smoke] {name:22} mask={mask:2} cls={r['cls']} wall={w if w is None else round(w,1)}us "
            f"pcc_err={r.get('max_rel_err')}",
            flush=True,
        )


def matrix():
    out = []
    for M, K, N in PRIMARY:
        cfgs = pick_configs(M, K, N)
        print(
            f"\n=== {M}x{K}x{N} ideal={ideal_us(M,K,N):.1f}us configs={ {','.join(v):list(k) for k,v in cfgs.items()} }",
            flush=True,
        )
        for cfg, labels in cfgs.items():
            lab = "/".join(labels)
            for name, mask in ABLATIONS:
                runs = [run_one(M, K, N, cfg, mask) for _ in range(3)]
                # Aggregate on the real `ok` field (mask 0 requires PCC pass), NOT cls=="ok" (which is set
                # whenever the profiler parsed, regardless of a mask-0 correctness failure).
                oks = [x for x in runs if x.get("ok") and x["wall_us"]]
                walls = [x["wall_us"] for x in oks]
                wall = min(walls) if walls else None
                med = statistics.median(walls) if walls else None
                rec = {
                    "M": M,
                    "K": K,
                    "N": N,
                    "cfg": list(cfg),
                    "labels": labels,
                    "ablation": name,
                    "mask": mask,
                    "wall_us_all": walls,  # every relaunch measurement (audit stability independently)
                    "wall_us_min": wall,
                    "wall_us_med": med,
                    "n_ok": len(oks),
                    "n_runs": len(runs),
                    "max_rel_err": [x.get("max_rel_err") for x in runs],
                    "ideal_us": ideal_us(M, K, N),
                    "risc": (oks[0]["risc"] if oks else None),
                }
                out.append(rec)
                print(
                    f"  [{lab:14}] {name:22} wall_med={med if med is None else round(med,1)}us "
                    f"all={[round(w,1) for w in walls]} ({len(oks)}/3 ok)",
                    flush=True,
                )
                json.dump(out, open(f"{HERE}/regime_a_diag_matrix.json", "w"), indent=2)
    print("MATRIX DONE", flush=True)


def mscale():
    # full + key ablations across M, at each shape's auto (picker) config, to see which component's excess
    # grows with M. Two series: small-N (2048,1024) and wide-N (6144,4608).
    key = [
        ("full", 0),
        ("skipin1", 1),
        ("skipfwd", 4),
        ("noreduce", 8),
        ("skipin0+in1", 3),
        ("skipin0+in1+noreduce", 11),
    ]
    series = {
        "smallN": [(M, 2048, 1024) for M in (32, 64, 128, 256)],
        "wideN": [(M, 6144, 4608) for M in (32, 64, 128, 256)],
    }
    out = []
    for sname, shapes in series.items():
        for M, K, N in shapes:
            cfg = tuple(rb.auto_config(M, K, N))
            print(f"\n=== {sname} {M}x{K}x{N} cfg={cfg} ideal={ideal_us(M,K,N):.1f}us", flush=True)
            for name, mask in key:
                runs = [run_one(M, K, N, cfg, mask) for _ in range(3)]
                oks = [x for x in runs if x.get("ok") and x["wall_us"]]
                walls = [x["wall_us"] for x in oks]
                med = statistics.median(walls) if walls else None
                exc = (med - ideal_us(M, K, N)) if med else None
                out.append(
                    {
                        "series": sname,
                        "M": M,
                        "K": K,
                        "N": N,
                        "cfg": list(cfg),
                        "ablation": name,
                        "mask": mask,
                        "wall_us_all": walls,
                        "wall_us_med": med,
                        "excess_us": exc,
                        "ideal_us": ideal_us(M, K, N),
                        "n_ok": len(oks),
                    }
                )
                print(
                    f"  {name:22} wall_med={med if med is None else round(med,1)}us "
                    f"excess={exc if exc is None else round(exc,1)}us ({len(oks)}/3)",
                    flush=True,
                )
                json.dump(out, open(f"{HERE}/regime_a_diag_mscale.json", "w"), indent=2)
    print("MSCALE DONE", flush=True)


def scatter():
    # Benchmark the DIAG_IN0_SCATTER variant (mask 32) vs ring (mask 0) at each shape's winner config.
    # Targets = narrow-N (must show >=8-10% stable end-to-end to justify porting); controls = wide-N (must
    # NOT materially regress). Both masks are correctness-checked (PCC) by the gtest.
    groups = {"target": [(256, 2048, 1024), (256, 6144, 768)], "control": [(256, 6144, 2304), (256, 6144, 4608)]}
    out = []
    for grp, shapes in groups.items():
        for M, K, N in shapes:
            cfgs = pick_configs(M, K, N)
            winner = next((c for c, labels in cfgs.items() if "winner" in labels), None)
            if winner is None:
                print(f"[scatter] {M}x{K}x{N}: no winner cfg (sweep missing); skipping", flush=True)
                continue
            res = {}
            for name, mask in (("ring", 0), ("scatter", 32)):
                runs = [run_one(M, K, N, winner, mask) for _ in range(3)]
                oks = [x for x in runs if x.get("ok") and x["wall_us"]]
                walls = [x["wall_us"] for x in oks]
                res[name] = {
                    "walls": walls,
                    "med": statistics.median(walls) if walls else None,
                    "min": min(walls) if walls else None,
                    "n_ok": len(oks),
                    "pcc": [x.get("max_rel_err") for x in runs],
                }
            rm, sm = res["ring"]["med"], res["scatter"]["med"]
            delta = (sm / rm - 1) * 100 if (rm and sm) else None  # negative = scatter faster
            rec = {
                "group": grp,
                "M": M,
                "K": K,
                "N": N,
                "cfg": list(winner),
                "ideal_us": ideal_us(M, K, N),
                "ring": res["ring"],
                "scatter": res["scatter"],
                "scatter_vs_ring_pct": delta,
            }
            out.append(rec)
            print(
                f"[scatter/{grp}] {M}x{K}x{N} cfg={winner} ring={rm if rm is None else round(rm,1)}us "
                f"scatter={sm if sm is None else round(sm,1)}us "
                f"delta={delta if delta is None else round(delta,1)}% "
                f"ring_all={[round(w,1) for w in res['ring']['walls']]} "
                f"scat_all={[round(w,1) for w in res['scatter']['walls']]}",
                flush=True,
            )
            json.dump(out, open(f"{HERE}/regime_a_scatter_bench.json", "w"), indent=2)
    print("SCATTER DONE", flush=True)


def _bytes(M, K, N):
    Mt, Kt, Nt = cdiv(M, 32), cdiv(K, 32), cdiv(N, 32)
    return {"in0": Mt * Kt * 2048, "in1": Kt * Nt * 2048, "out": Mt * Nt * 2048}


# in0-delivery variants: (name, mask, in0 DRAM replication factor R). Replicated rings read in0 R times.
VARIANTS = [("ring", 0, 1), ("repl2", 64, 2), ("xchg", 256, 1), ("xchgrr", 512, 1)]


def variants():
    # Compare all correct in0-delivery variants vs the ring at each shape's winner config, per the protocol:
    # retain every relaunch, per-RISC spans, PCC; report logical eff-BW (logical/time) AND delivered eff-BW
    # (actual DRAM bytes incl. R x in0 replication / time). Gate: target >=8% faster, control regress <=3%.
    groups = {"target": [(256, 2048, 1024), (256, 6144, 768)], "control": [(256, 6144, 2304), (256, 6144, 4608)]}
    out = []
    for grp, shapes in groups.items():
        for M, K, N in shapes:
            cfgs = pick_configs(M, K, N)
            winner = next((c for c, labels in cfgs.items() if "winner" in labels), None)
            if winner is None:
                print(f"[variants] {M}x{K}x{N}: no winner cfg (sweep missing); skipping", flush=True)
                continue
            b = _bytes(M, K, N)
            logical = b["in0"] + b["in1"] + b["out"]
            per = {}
            for name, mask, R in VARIANTS:
                runs = [run_one(M, K, N, winner, mask) for _ in range(3)]
                oks = [x for x in runs if x.get("ok") and x["wall_us"]]
                walls = sorted(x["wall_us"] for x in oks)
                med = statistics.median(walls) if walls else None
                delivered = R * b["in0"] + b["in1"] + b["out"]
                per[name] = {
                    "mask": mask,
                    "repl": R,
                    "walls": walls,
                    "med_us": med,
                    "min_us": (walls[0] if walls else None),
                    "n_ok": len(oks),
                    "pcc": [x.get("max_rel_err") for x in runs],
                    "risc": (oks[0]["risc"] if oks else None),
                    "logical_gbps": (logical / (med / 1e6) / 1e9 if med else None),
                    "delivered_gbps": (delivered / (med / 1e6) / 1e9 if med else None),
                    "delivered_bytes": delivered,
                }
            rm = per["ring"]["med_us"]
            for name, _, _ in VARIANTS:
                m = per[name]["med_us"]
                per[name]["vs_ring_pct"] = ((m / rm - 1) * 100) if (rm and m) else None
            out.append(
                {
                    "group": grp,
                    "M": M,
                    "K": K,
                    "N": N,
                    "cfg": list(winner),
                    "ideal_us": ideal_us(M, K, N),
                    "logical_bytes": logical,
                    "variants": per,
                }
            )
            summ = " ".join(
                f"{n}={per[n]['med_us'] if per[n]['med_us'] is None else round(per[n]['med_us'],1)}"
                f"({'' if per[n]['vs_ring_pct'] is None else ('%+d' % round(per[n]['vs_ring_pct']))}%)"
                for n, _, _ in VARIANTS
            )
            print(f"[variants/{grp}] {M}x{K}x{N} cfg={winner} ideal={ideal_us(M,K,N):.1f}  {summ}", flush=True)
            json.dump(out, open(f"{HERE}/regime_a_variants_bench.json", "w"), indent=2)
    print("VARIANTS DONE", flush=True)


def _cfg_for(M, K, N, explicit):
    """Explicit winning cfg if given; else the sweep winner; else the picker auto config."""
    if explicit is not None:
        return tuple(explicit)
    cfgs = pick_configs(M, K, N)
    winner = next((c for c, labels in cfgs.items() if "winner" in labels), None)
    return winner if winner is not None else tuple(rb.auto_config(M, K, N))


# (group, label, M, K, N, explicit_cfg or None). Primary targets pin the spec's winning configs; controls
# use the sweep winner / picker auto config. cfg tuple order = (Ns, Pk, Sm, kb, nsb).
PROG_SHAPES = [
    ("target", "256x2048x1024", 256, 2048, 1024, (1, 4, 2, 2, 2)),
    ("target", "256x6144x768", 256, 6144, 768, (1, 12, 1, 2, 1)),
    ("control", "256x6144x2304", 256, 6144, 2304, None),
    ("control", "256x6144x4608", 256, 6144, 4608, None),
    ("control", "mt1_32x6144x4608", 32, 6144, 4608, None),
    ("control", "mt2_64x6144x4608", 64, 6144, 4608, None),
    ("control", "mt4_128x6144x4608", 128, 6144, 4608, None),
]


def progressive():
    # A/B: progressive cumulative in0 waits (default, mask 0) vs the OLD full-slice startup barrier
    # (DIAG_FULL_IN0_WAIT, mask 1024) in the SAME binary at the SAME config. Both are correctness-checked
    # (constant-input max_rel_err) by the gtest; the real PCC + bit-identical A/B is the ProgressiveVsFullWait
    # gtest. Report: full/prog median kernel us, %change, every relaunch, logical %512, per-RISC spans, cfg.
    out = []
    for grp, label, M, K, N, explicit in PROG_SHAPES:
        cfg = _cfg_for(M, K, N, explicit)
        ideal = ideal_us(M, K, N)
        res = {}
        for name, mask in (("full", 1024), ("prog", 0)):  # freeze full-wait baseline first
            runs = [run_one(M, K, N, cfg, mask) for _ in range(3)]
            oks = [x for x in runs if x.get("ok") and x["wall_us"]]
            walls = sorted(x["wall_us"] for x in oks)
            med = statistics.median(walls) if walls else None
            res[name] = {
                "walls": walls,
                "med_us": med,
                "min_us": (walls[0] if walls else None),
                "n_ok": len(oks),
                "util512_pct": (ideal / med * 100 if med else None),
                "max_rel_err": [x.get("max_rel_err") for x in runs],
                "risc": (oks[0]["risc"] if oks else None),
            }
        fm, pm = res["full"]["med_us"], res["prog"]["med_us"]
        delta = (pm / fm - 1) * 100 if (fm and pm) else None  # negative = progressive faster
        rec = {
            "group": grp,
            "label": label,
            "M": M,
            "K": K,
            "N": N,
            "cfg": list(cfg),
            "ideal_us": ideal,
            "full_wait": res["full"],
            "progressive": res["prog"],
            "prog_vs_full_pct": delta,
        }
        out.append(rec)
        print(
            f"[prog/{grp}] {label:18} cfg={cfg} ideal={ideal:.1f}us  "
            f"full={fm if fm is None else round(fm,1)}us{res['full']['walls']!s:>0} "
            f"prog={pm if pm is None else round(pm,1)}us "
            f"delta={delta if delta is None else round(delta,1)}%  "
            f"full_all={[round(w,1) for w in res['full']['walls']]} "
            f"prog_all={[round(w,1) for w in res['prog']['walls']]} "
            f"util512 full={res['full']['util512_pct'] and round(res['full']['util512_pct'],1)}%"
            f"->prog={res['prog']['util512_pct'] and round(res['prog']['util512_pct'],1)}%",
            flush=True,
        )
        json.dump(out, open(f"{HERE}/regime_a_progressive_bench.json", "w"), indent=2)
    print("PROGRESSIVE DONE", flush=True)


def _rup(x, y):
    return cdiv(x, y) * y


def _pd_geom(M, K, N, cfg):
    """K_num_blocks, N_bpc, out-block tiles (M_block_cap * N_sub) — mirrors build_plan."""
    Ns, Pk, Sm, kb, nsb = cfg
    Mt, Kt, Nt = cdiv(M, 32), cdiv(K, 32), cdiv(N, 32)
    Knb = _rup(cdiv(Kt, Pk), kb * 8) // kb
    N_own = cdiv(Nt, 8 * Ns)
    N_sub = nsb if nsb else N_own
    N_bpc = cdiv(N_own, N_sub)
    M_block_cap = cdiv(Mt, Sm)
    return Knb, N_bpc, M_block_cap * N_sub


# (group, label, M, K, N, explicit cfg or None). control_pk1 isolates output draining (no reduction).
PD_SHAPES = [
    ("target", "256x2048x1024", 256, 2048, 1024, None),
    ("target", "256x6144x768", 256, 6144, 768, None),
    ("control", "256x6144x2304", 256, 6144, 2304, None),
    ("control", "256x6144x4608", 256, 6144, 4608, None),
    ("control", "mt1_32x6144x4608", 32, 6144, 4608, None),
    ("control", "mt2_64x6144x4608", 64, 6144, 4608, None),
    ("control", "mt4_128x6144x4608", 128, 6144, 4608, None),
    ("control_pk1", "pk1_32x6144x3072", 32, 6144, 3072, (1, 1, 1, 4, 6)),
]


def pipelined(relaunches=3):
    # A/B: barrier baseline (mask 0) vs pipelined phase-2 drain (DIAG_PIPELINED_DRAIN, mask 2048) at identical
    # configs, INTERLEAVED relaunches. Reports median, %change vs baseline, util%512, per-RISC, PCC, Pk, N_bpc,
    # out-block tiles. Raw: regime_a_pipelined_bench.json.
    VARIANTS = [("pipelined", 0), ("barrier", 2048)]  # mask 0 = pipelined (default); 2048 = DIAG_BARRIER_DRAIN
    out = []
    for grp, label, M, K, N, explicit in PD_SHAPES:
        cfg = _cfg_for(M, K, N, explicit)
        Pk = cfg[1]
        Knb, N_bpc, out_blk_tiles = _pd_geom(M, K, N, cfg)
        ideal = ideal_us(M, K, N)
        runs = {v: [] for v, _ in VARIANTS}
        for _r in range(relaunches):  # interleaved
            for v, mask in VARIANTS:
                runs[v].append(run_one(M, K, N, cfg, mask))
        per = {}
        for v, mask in VARIANTS:
            oks = [x for x in runs[v] if x.get("ok") and x["wall_us"]]
            walls = sorted(x["wall_us"] for x in oks)
            med = statistics.median(walls) if walls else None
            per[v] = {
                "mask": mask,
                "walls": walls,
                "med_us": med,
                "min_us": (walls[0] if walls else None),
                "n_ok": len(oks),
                "util512_pct": (ideal / med * 100 if med else None),
                "max_rel_err": [x.get("max_rel_err") for x in runs[v]],
                "risc": (oks[0]["risc"] if oks else None),
            }
        bm = per["barrier"]["med_us"]
        pm = per["pipelined"]["med_us"]
        delta = (pm / bm - 1) * 100 if (bm and pm) else None  # negative = pipelined faster
        out.append(
            {
                "group": grp,
                "label": label,
                "M": M,
                "K": K,
                "N": N,
                "cfg": list(cfg),
                "Pk": Pk,
                "N_bpc": N_bpc,
                "out_blk_tiles": out_blk_tiles,
                "ideal_us": ideal,
                "barrier": per["barrier"],
                "pipelined": per["pipelined"],
                "pipelined_vs_barrier_pct": delta,
            }
        )
        print(
            f"[pd/{grp}] {label:18} cfg={cfg} Pk={Pk} N_bpc={N_bpc} oblk={out_blk_tiles} ideal={ideal:.1f}  "
            f"barrier={bm if bm is None else round(bm,1)} pipe={pm if pm is None else round(pm,1)} "
            f"delta={delta if delta is None else round(delta,1)}%  "
            f"bar_all={[round(w,1) for w in per['barrier']['walls']]} "
            f"pipe_all={[round(w,1) for w in per['pipelined']['walls']]} "
            f"util {per['barrier']['util512_pct'] and round(per['barrier']['util512_pct'],1)}"
            f"->{per['pipelined']['util512_pct'] and round(per['pipelined']['util512_pct'],1)}%",
            flush=True,
        )
        json.dump(out, open(f"{HERE}/regime_a_pipelined_bench.json", "w"), indent=2)
    print("PIPELINED DONE", flush=True)


# Sm>1 shapes (explicit cfg) exercise the aggregate-vs-mm0 objective difference; Sm=1 controls confirm parity.
RING_SHAPES = [
    ("target", "256x2048x1024_sm2", 256, 2048, 1024, None),  # auto -> (1,4,2,2,2), Sm=2
    ("sm", "128x6144x4608_sm2", 128, 6144, 4608, (1, 6, 2, 2, 1)),  # Sm=2 wide
    ("sm", "256x2048x1024_sm4", 256, 2048, 1024, (1, 1, 4, 2, 2)),  # Sm=4 (4 mm-rings)
    ("control", "256x6144x768", 256, 6144, 768, None),  # Sm=1 primary
    ("control", "256x6144x4608", 256, 6144, 4608, None),  # Sm=1 wide-N
    ("control", "256x6144x2304", 256, 6144, 2304, None),  # Sm=1 wide-N
]


def _ring_agg(ringcost):
    # op-level route cost per order, aggregated ACROSS the (kk,nn) ring groups: worst group-aggregate max-edge
    # (max over groups of the group's worst-over-mm-rings edge) + sum of group-aggregate total hops.
    agg = {}
    for od in ("bank", "greedy", "mm0", "agg"):
        mx = [g[od + "_aggmax"] for g in ringcost if (od + "_aggmax") in g]
        tt = [g[od + "_aggtot"] for g in ringcost if (od + "_aggtot") in g]
        agg[od] = {"max_edge": (max(mx) if mx else None), "total_hops": (sum(tt) if tt else None)}
    return agg


def ringorder(relaunches=3):
    # A/B: bank (1<<12) vs mm0-opt (1<<14) vs agg-opt (default, mask 0) in0 ring ordering, INTERLEAVED
    # relaunches. Route cost (per-order group-aggregate max-edge/total-hops, aggregated across ring groups,
    # from the factory RINGCOST) + wall/%change/util/per-RISC/PCC. Raw: regime_a_ringorder_bench.json.
    VARIANTS = [("bank", 4096), ("mm0", 16384), ("agg", 0)]  # agg = mask 0 (default); bank/mm0 = diagnostics
    out = []
    for grp, label, M, K, N, explicit in RING_SHAPES:
        cfg = _cfg_for(M, K, N, explicit)
        ideal = ideal_us(M, K, N)
        runs = {v: [] for v, _ in VARIANTS}
        for _r in range(relaunches):  # interleaved
            for v, mask in VARIANTS:
                runs[v].append(run_one(M, K, N, cfg, mask))
        per = {}
        for v, mask in VARIANTS:
            oks = [x for x in runs[v] if x.get("ok") and x["wall_us"]]
            walls = sorted(x["wall_us"] for x in oks)
            med = statistics.median(walls) if walls else None
            per[v] = {
                "mask": mask,
                "walls": walls,
                "med_us": med,
                "n_ok": len(oks),
                "util512_pct": (ideal / med * 100 if med else None),
                "max_rel_err": [x.get("max_rel_err") for x in runs[v]],
                "risc": (oks[0]["risc"] if oks else None),
            }
        bm = per["bank"]["med_us"]
        for v, _ in VARIANTS:
            m = per[v]["med_us"]
            per[v]["vs_bank_pct"] = ((m / bm - 1) * 100) if (bm and m) else None
        # also mm0-relative for agg (the actual decision: agg vs the current production mm0)
        m0 = per["mm0"]["med_us"]
        ag = per["agg"]["med_us"]
        per["agg"]["vs_mm0_pct"] = ((ag / m0 - 1) * 100) if (m0 and ag) else None
        # route cost from the agg run's RINGCOST (prints bank/greedy/mm0/agg); fall back to the mm0 run.
        rc = next((x["ringcost"] for x in runs["agg"] if x.get("ringcost")), None) or next(
            (x["ringcost"] for x in runs["mm0"] if x.get("ringcost")), []
        )
        route = _ring_agg(rc)
        out.append(
            {
                "group": grp,
                "label": label,
                "M": M,
                "K": K,
                "N": N,
                "cfg": list(cfg),
                "ideal_us": ideal,
                "route_cost": route,
                "ringcost_groups": rc,
                "bank": per["bank"],
                "mm0": per["mm0"],
                "agg": per["agg"],
            }
        )
        print(
            f"[ring/{grp}] {label:20} cfg={cfg} ideal={ideal:.1f}  "
            f"bank={bm if bm is None else round(bm,1)} "
            f"mm0={m0 and round(m0,1)}({per['mm0']['vs_bank_pct'] and round(per['mm0']['vs_bank_pct'],1)}%) "
            f"agg={ag and round(ag,1)}({per['agg']['vs_bank_pct'] and round(per['agg']['vs_bank_pct'],1)}%) "
            f"agg_vs_mm0={per['agg']['vs_mm0_pct'] and round(per['agg']['vs_mm0_pct'],1)}%  "
            f"route[aggmax bank={route['bank']['max_edge']} mm0={route['mm0']['max_edge']} "
            f"agg={route['agg']['max_edge']} | aggtot bank={route['bank']['total_hops']} "
            f"mm0={route['mm0']['total_hops']} agg={route['agg']['total_hops']}]",
            flush=True,
        )
        json.dump(out, open(f"{HERE}/regime_a_ringorder_bench.json", "w"), indent=2)
    print("RINGORDER DONE", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    {
        "smoke": smoke,
        "matrix": matrix,
        "mscale": mscale,
        "scatter": scatter,
        "variants": variants,
        "progressive": progressive,
        "pipelined": pipelined,
        "ringorder": ringorder,
    }[mode]()
