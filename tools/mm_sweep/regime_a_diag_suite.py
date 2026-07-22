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

# (name, mask). The skip-in0/in1-read/forward ablations (bits 0/1/2) were removed after the delivery study
# concluded the ring is optimal; only the still-used no-reduce ablation (bit 3) remains alongside full (0).
ABLATIONS = [
    ("full", 0),
    ("noreduce", 8),
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
        TT_MM_PLACECOST="1",  # emit the factory's PLACECOST placement-cost line (placement diag); harmless otherwise
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
    placecost = []  # PLACECOST lines (factory, placement diag): per-group reader dist + reader->slave fwd hops
    ringcost = []  # RINGCOST lines (factory, ring-order diag): per-group bank/greedy/opt max+total edge cost
    for line in r.stdout.splitlines():
        if "DIAGPCC" in line:
            maxrel = float(line.split("max_rel_err=")[1])
        elif "PLACECOST" in line:
            g = {}
            for tok in line.split():
                if tok.startswith(("rdr2tgt=", "maxfwd=")):
                    k, v = tok.split("=")
                    g[k] = int(v)
            if g:
                placecost.append(g)
        elif "RINGCOST" in line:
            g = {}
            for tok in line.split():
                if tok.startswith(("group=", "wnoc=", "sel=", "Sm=", "sel_perring=")):
                    k, v = tok.split("=", 1)
                    g[k] = v
                for od in ("bank", "mm0", "maxedge", "total", "pareto"):
                    # token form: <od>[perm]=aggmax:aggtot:maxringtot
                    if tok.startswith(od + "[") and "]=" in tok:
                        vals = tok.split("]=", 1)[1].split(":")
                        if len(vals) == 3:
                            g[od + "_aggmax"] = int(vals[0])
                            g[od + "_aggtot"] = int(vals[1])
                            g[od + "_maxringtot"] = int(vals[2])
            if "group" in g:
                ringcost.append(g)
    # mask 0 (public path) and the correctness-preserving diagnostics (full-wait / barrier-drain / ring-order /
    # placement / in1 A/B) are correctness-checked by the gtest -> require the PASS; the pure ablations (e.g.
    # no-reduce) produce garbage and are NOT checked. Must match the gtest's constant-input check set.
    _in1preserve = (1 << 22) | (1 << 25)  # in1-delivery A/B diagnostics (correctness-preserving)
    _fc = 256  # DIAG_FORCE_CHAIN (chain; correctness-preserving). Strip it before the correctness-set check.
    m = mask & ~_fc
    checked = m in (0, 64, 128, 1024, 2048, 4096, 16384, 65536, 262144, 524288, 2097152) or (
        m != 0 and (m & ~_in1preserve) == 0
    )  # 64 = DIAG_REDTREE, 128 = DIAG_RSCATTER (both reassociate; constant-input still sums to K)
    return {
        "cfg": list(cfg),
        "mask": mask,
        "ok": bool(wall_us) and (not checked or passed),
        "cls": "ok",
        "wall_us": wall_us,
        "risc": risc,
        "max_rel_err": maxrel,
        "ringcost": ringcost,
        "placecost": placecost,
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
        ("noreduce", 8),
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


# Sm>1 shapes (explicit cfg) exercise the objective differences; Sm=1 controls confirm parity + noise floor.
RING_SHAPES = [
    ("target", "256x2048x1024_sm2", 256, 2048, 1024, None),  # auto -> (1,4,2,2,2), Sm=2 (production)
    ("sm2", "128x6144x4608_sm2", 128, 6144, 4608, (1, 6, 2, 2, 1)),  # Sm=2 wide
    ("sm4", "256x2048x1024_sm4", 256, 2048, 1024, (1, 1, 4, 2, 2)),  # Sm=4 synthetic (4 mm-rings)
    ("sm4", "256x6144x4608_sm4", 256, 6144, 4608, (1, 3, 4, 2, 1)),  # 2nd feasible Sm=4 (wide, Pk3)
    ("sm3", "256x2048x1024_sm3", 256, 2048, 1024, (1, 1, 3, 2, 2)),  # Sm=3 (balanced M-split)
    ("control", "256x6144x768", 256, 6144, 768, None),  # Sm=1 primary
    ("control", "256x6144x4608", 256, 6144, 4608, None),  # Sm=1 wide-N
    ("control", "256x6144x2304", 256, 6144, 2304, None),  # Sm=1 wide-N
]

RING_OBJ = [
    ("bank", 4096),
    ("mm0", 16384),
    ("maxedge", 262144),
    ("total", 65536),
    ("pareto", 0),  # production default
]


def _ring_agg(ringcost):
    # op-level route cost per objective, aggregated ACROSS the (kk,nn) ring groups: worst group-aggregate
    # max-edge (max over groups) + sum of group-aggregate total hops + worst group maxringtot.
    agg = {}
    for od in ("bank", "mm0", "maxedge", "total", "pareto"):
        mx = [g[od + "_aggmax"] for g in ringcost if (od + "_aggmax") in g]
        tt = [g[od + "_aggtot"] for g in ringcost if (od + "_aggtot") in g]
        mr = [g[od + "_maxringtot"] for g in ringcost if (od + "_maxringtot") in g]
        agg[od] = {
            "max_edge": (max(mx) if mx else None),
            "total_hops": (sum(tt) if tt else None),
            "max_ring_total": (max(mr) if mr else None),
        }
    return agg


def ringorder(relaunches=3):
    # A/B over ring-order objectives (bank / mm0 / maxedge=default / maxring / total / pareto), INTERLEAVED
    # relaunches. Route cost (op-aggregate, from the factory RINGCOST) + wall/%change vs bank AND vs mm0 /
    # util / per-RISC / PCC / selected perms. Raw: regime_a_ringorder_bench.json.
    VARIANTS = RING_OBJ
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
        m0 = per["mm0"]["med_us"]
        for v, _ in VARIANTS:
            m = per[v]["med_us"]
            per[v]["vs_bank_pct"] = ((m / bm - 1) * 100) if (bm and m) else None
            per[v]["vs_mm0_pct"] = ((m / m0 - 1) * 100) if (m0 and m) else None
        # route cost from any run that emitted RINGCOST (maxedge=default prints all candidates).
        rc = None
        for v, _ in VARIANTS:
            rc = next((x["ringcost"] for x in runs[v] if x.get("ringcost")), None)
            if rc:
                break
        route = _ring_agg(rc or [])
        rec = {
            "group": grp,
            "label": label,
            "M": M,
            "K": K,
            "N": N,
            "cfg": list(cfg),
            "ideal_us": ideal,
            "route_cost": route,
            "ringcost_groups": rc,
        }
        for v, _ in VARIANTS:
            rec[v] = per[v]
        out.append(rec)
        summ = " ".join(
            f"{v}={per[v]['med_us'] and round(per[v]['med_us'],1)}"
            f"(b{per[v]['vs_bank_pct'] and round(per[v]['vs_bank_pct'],1)}"
            f"/m{per[v]['vs_mm0_pct'] and round(per[v]['vs_mm0_pct'],1)})"
            for v, _ in VARIANTS
        )
        print(f"[ring/{grp}] {label:20} cfg={cfg} ideal={ideal:.1f} (pct = %vs_bank / %vs_mm0)\n    {summ}", flush=True)
        json.dump(out, open(f"{HERE}/regime_a_ringorder_bench.json", "w"), indent=2)
    print("RINGORDER DONE", flush=True)


PLACE_SHAPES = [
    ("target", "256x2048x1024_sm2", 256, 2048, 1024, None),  # production Sm=2 primary
    ("sm2", "128x6144x4608_sm2", 128, 6144, 4608, (1, 6, 2, 2, 1)),
    ("sm3", "256x2048x1024_sm3", 256, 2048, 1024, (1, 1, 3, 2, 2)),
    ("sm4", "256x2048x1024_sm4", 256, 2048, 1024, (1, 1, 4, 2, 2)),
    ("sm4", "256x6144x4608_sm4", 256, 6144, 4608, (1, 3, 4, 2, 1)),
    ("control", "256x6144x768_sm1", 256, 6144, 768, None),  # Sm=1: placement is a no-op (parity sanity)
    ("control", "256x6144x4608_sm1", 256, 6144, 4608, None),
    ("control", "32x6144x4608_sm1", 32, 6144, 4608, None),
    ("control", "64x6144x4608_sm1", 64, 6144, 4608, None),
    ("control", "128x6144x4608_sm1", 128, 6144, 4608, None),
]
PLACE_VARIANTS = [("current", 2097152), ("readers_first", 524288), ("in1_near", 0)]  # in1_near = default


def placement(relaunches=3):
    # A/B over M-split worker placement (current / readers_first / in1_near), INTERLEAVED relaunches.
    # Wall/%change vs current + per-op max reader->slave forward hops (from PLACECOST). Raw:
    # regime_a_placement_bench.json.
    out = []
    for grp, label, M, K, N, explicit in PLACE_SHAPES:
        cfg = _cfg_for(M, K, N, explicit)
        ideal = ideal_us(M, K, N)
        runs = {v: [] for v, _ in PLACE_VARIANTS}
        for _r in range(relaunches):
            for v, mask in PLACE_VARIANTS:
                runs[v].append(run_one(M, K, N, cfg, mask))
        per = {}
        for v, mask in PLACE_VARIANTS:
            oks = [x for x in runs[v] if x.get("ok") and x["wall_us"]]
            walls = sorted(x["wall_us"] for x in oks)
            med = statistics.median(walls) if walls else None
            pcs = next((x["placecost"] for x in runs[v] if x.get("placecost")), [])
            per[v] = {
                "mask": mask,
                "walls": walls,
                "med_us": med,
                "n_ok": len(oks),
                "util512_pct": (ideal / med * 100 if med else None),
                "max_rel_err": [x.get("max_rel_err") for x in runs[v]],
                "risc": (oks[0]["risc"] if oks else None),
                "op_maxfwd": (max((g["maxfwd"] for g in pcs), default=None)),
                "op_sumfwd": (sum(g["maxfwd"] for g in pcs) if pcs else None),
                "op_maxrdr2tgt": (max((g["rdr2tgt"] for g in pcs), default=None)),
            }
        cm = per["current"]["med_us"]
        for v, _ in PLACE_VARIANTS:
            m = per[v]["med_us"]
            per[v]["vs_current_pct"] = ((m / cm - 1) * 100) if (cm and m) else None
        rec = {"group": grp, "label": label, "M": M, "K": K, "N": N, "cfg": list(cfg), "ideal_us": ideal}
        for v, _ in PLACE_VARIANTS:
            rec[v] = per[v]
        out.append(rec)
        summ = " ".join(
            f"{v}={per[v]['med_us'] and round(per[v]['med_us'],1)}"
            f"({per[v]['vs_current_pct'] and round(per[v]['vs_current_pct'],1)}%,maxfwd={per[v]['op_maxfwd']})"
            for v, _ in PLACE_VARIANTS
        )
        print(f"[place/{grp}] {label:20} cfg={cfg} ideal={ideal:.1f}  {summ}", flush=True)
        json.dump(out, open(f"{HERE}/regime_a_placement_bench.json", "w"), indent=2)
    print("PLACEMENT DONE", flush=True)


# Part-5 picker re-sweep: with in1_near making M-split cheaper, re-check whether any Mt>=4 production
# shape's best Sm changed. Keyed (M,K,N); cfg tuple order = (Ns,Pk,Sm,kb,nsb). All run at mask=0 (in1_near
# default). For each shape we test the picker's current cfg plus core-budget-matched Sm variants (Pk*Sm
# held ~constant, same Ns/kb/nsb) that are feasible for the shape's Mt.
PICKER_RESWEEP_SHAPES = [
    (128, 2304, 6144),  # Mt=4 picker Sm=1 (Ns2,Pk3)
    (128, 6144, 768),  # Mt=4 picker Sm=1 (Pk12)
    (128, 6144, 2304),  # Mt=4 picker Sm=1 (Pk12)
    (128, 6144, 4608),  # Mt=4 picker Sm=1 (Pk12) -- placement test shape, in1_near -3.5% at Sm=2
    (128, 15360, 768),  # Mt=4 picker Sm=1 (Pk6)
    (256, 2048, 1024),  # Mt=8 picker Sm=2 (production primary) -- confirm Sm=2 still beats Sm1/Sm4
    (512, 6144, 1536),  # Mt=16 picker Sm=1 (Pk12)
]


def _sm_candidates(M, K, N):
    """Picker cfg + core-budget-matched Sm variants (Pk*Sm ~ const, same Ns/kb/nsb), feasible for Mt."""
    Ns, Pk, Sm, kb, nsb = rb.auto_config(M, K, N)
    Mt = cdiv(M, 32)
    budget = Pk * Sm
    cands = [(f"picker_sm{Sm}", (Ns, Pk, Sm, kb, nsb))]
    for sm_c in (1, 2, 4):
        if sm_c == Sm or Mt % sm_c != 0:
            continue
        pk_c = max(1, budget // sm_c)
        cands.append((f"sm{sm_c}", (Ns, pk_c, sm_c, kb, nsb)))
    return cands


def pickerresweep(relaunches=3):
    # For each Mt>=4 production shape, INTERLEAVED relaunches of the picker cfg vs core-budget-matched Sm
    # variants under the in1_near default. Report median kernel us + %vs picker; flag any Sm-flip candidate
    # that clears a stable ~2% gain. Raw: regime_a_picker_resweep.json.
    out = []
    for M, K, N in PICKER_RESWEEP_SHAPES:
        cands = _sm_candidates(M, K, N)
        ideal = ideal_us(M, K, N)
        runs = {name: [] for name, _ in cands}
        for _r in range(relaunches):
            for name, cfg in cands:
                runs[name].append(run_one(M, K, N, cfg, 0))
        per = {}
        for name, cfg in cands:
            oks = [x for x in runs[name] if x.get("ok") and x["wall_us"]]
            walls = sorted(x["wall_us"] for x in oks)
            med = statistics.median(walls) if walls else None
            per[name] = {
                "cfg": list(cfg),
                "walls": walls,
                "med_us": med,
                "n_ok": len(oks),
                "util512_pct": (ideal / med * 100 if med else None),
                "max_rel_err": [x.get("max_rel_err") for x in runs[name]],
            }
        pick_name = cands[0][0]
        pm = per[pick_name]["med_us"]
        for name, _ in cands:
            m = per[name]["med_us"]
            per[name]["vs_picker_pct"] = ((m / pm - 1) * 100) if (pm and m) else None
        rec = {"M": M, "K": K, "N": N, "ideal_us": ideal, "picker": pick_name, "cands": per}
        out.append(rec)
        summ = " ".join(
            f"{name}={per[name]['med_us'] and round(per[name]['med_us'],1)}"
            f"({per[name]['vs_picker_pct'] and round(per[name]['vs_picker_pct'],1)}%)"
            for name, _ in cands
        )
        print(f"[resweep] {M}x{K}x{N:5} ideal={ideal:.1f}  {summ}", flush=True)
        json.dump(out, open(f"{HERE}/regime_a_picker_resweep.json", "w"), indent=2)
    print("PICKER RESWEEP DONE", flush=True)


# in1-delivery experiment: forward-order (write->signal->flush) + CB1 depth 2/8 + coalesced read, each vs
# the mask-0 default, on the shapes' NEW picker-winning configs. Then the winning combination.
# After adoption, mask 0 = new fast default (fwd-signal-first + coalesce). The 1<<22 / 1<<25 flags select the
# OLD behaviour (A/B baselines), so they read as POSITIVE % (reverting to the old order costs that much).
IN1EXP_MASKS = [
    ("base", 0),
    ("old_fwd_flush_first", 1 << 22),  # Sm>1 only (no-op at Sm=1)
    ("old_no_coalesce", 1 << 25),
]
IN1EXP_SHAPES = [
    ("primary", 128, 2048, 512),
    ("primary", 256, 2048, 512),
    ("primary", 256, 2048, 1536),
    ("primary", 256, 2048, 1024),
    ("ctl_bw", 32, 15360, 768),  # bandwidth-bound negative control (in1 read -70%)
    ("ctl_wideN", 128, 2304, 6144),  # wide-N Sm=1
    ("ctl_wideN", 256, 6144, 4608),  # wide-N (Sm2)
    ("ctl_in0fwd", 64, 6144, 9216),  # deep-K, in0-ring-forward heavy
]


def _in1exp_run(M, K, N, cfg, masks, relaunches):
    runs = {n: [] for n, _ in masks}
    for _r in range(relaunches):
        for n, mask in masks:
            runs[n].append(run_one(M, K, N, cfg, mask))
    per = {}
    for n, mask in masks:
        oks = [x for x in runs[n] if x.get("ok") and x["wall_us"]]
        walls = sorted(x["wall_us"] for x in oks)
        per[n] = {
            "mask": mask,
            "med_us": (statistics.median(walls) if walls else None),
            "walls": walls,
            "n_ok": len(oks),
            "max_rel_err": max(
                (x.get("max_rel_err") for x in runs[n] if x.get("max_rel_err") is not None), default=None
            ),
        }
    bm = per["base"]["med_us"]
    for n, _ in masks:
        m = per[n]["med_us"]
        per[n]["vs_base_pct"] = ((m / bm - 1) * 100) if (bm and m) else None
    return per


def in1exp(relaunches=3):
    # Phase 1: each in1 diagnostic independently vs mask-0 on the NEW winning config. Phase 2: the winning
    # combination (per-shape best-improving individuals). Reports median wall, %vs base, PCC max_rel_err.
    out = []
    for grp, M, K, N in IN1EXP_SHAPES:
        cfg = tuple(rb.auto_config(M, K, N))
        Sm = cfg[2]
        masks = [(n, m) for n, m in IN1EXP_MASKS if not (n == "fwd_sig_first" and Sm == 1)]
        per = _in1exp_run(M, K, N, cfg, masks, relaunches)
        # winning combination = all individually-improving (>1%, PCC-ok) correctness-preserving diagnostics
        combo_bits, combo_names = 0, []
        for n, _ in masks:
            if n == "base":
                continue
            v = per[n]["vs_base_pct"]
            mre = per[n]["max_rel_err"]
            if v is not None and v < -1.0 and (mre is None or mre < 0.02):
                combo_bits |= per[n]["mask"]
                combo_names.append(n)
        if combo_bits and len(combo_names) >= 2:
            cr = _in1exp_run(M, K, N, cfg, [("base", 0), ("combo", combo_bits)], relaunches)
            per["combo"] = cr["combo"]
            per["combo"]["names"] = combo_names
        rec = {"group": grp, "M": M, "K": K, "N": N, "cfg": list(cfg), "Sm": Sm, "per": per}
        out.append(rec)
        json.dump(out, open(f"{HERE}/regime_a_in1exp.json", "w"), indent=2)
        summ = " ".join(
            f"{n}={per[n]['med_us'] and round(per[n]['med_us'],1)}({per[n]['vs_base_pct'] and round(per[n]['vs_base_pct'],1)}%"
            f"{',pcc!' if (per[n].get('max_rel_err') or 0) >= 0.02 else ''})"
            for n in per
        )
        print(f"[in1exp/{grp}] {M}x{K}x{N} cfg={list(cfg)} {summ}", flush=True)
    print("IN1EXP DONE", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    {
        "smoke": smoke,
        "matrix": matrix,
        "mscale": mscale,
        "progressive": progressive,
        "pipelined": pipelined,
        "ringorder": ringorder,
        "placement": placement,
        "pickerresweep": pickerresweep,
        "in1exp": in1exp,
    }[mode]()
