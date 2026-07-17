#!/usr/bin/env python3
# 8-hour RESUMABLE Mt<=8 re-baseline + picker-eval + causal-ablation campaign for
# ttnn.experimental.regime_a_matmul on Blackhole. Produces EVIDENCE + CANDIDATES only.
# NO production kernel or picker changes (all measurement / offline analysis).
#
# Every measurement goes through rb.run_cfg (hang-safe subprocess, tt-smi reset on hang) backed by a
# dedicated CacheStore (regime_a_campaign_cache.json) that MERGES + writes ATOMICALLY per config -> the
# cache IS the checkpoint: a crash/hang/kill loses at most the in-flight config, resume skips the rest.
#
# Stages (argv): baseline | picker | select | ablate | aggregate | all
#   baseline  -> auto-picker on the full ~50-shape Mt<=8 corpus, full metrics       (Stage 1)
#   picker    -> bounded 25-40 candidate set per shape, picker-gap table            (Stage 2)
#   select    -> rank + pick <=6 bottom performers                                  (Stage 3)
#   ablate    -> causal diag ablations (via regime_a_diag_suite) on the picks       (Stage 4)
#   aggregate -> refreshed corpus report + picker-gap + ablation + recommendation   (Stage 5)
#   all       -> baseline;picker;select;ablate;aggregate with a ~7h launch deadline
import json, os, sys, time, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# picker_v2 opens fluxltx_regimeA_sweep.json via a bare relative path at import time; run the main process
# from the tools dir so that (and any other relative import-time open) resolves. Device subprocesses set
# their own cwd=ROOT internally (run_cfg / run_one), so this does not affect measurements.
os.chdir(HERE)
import regime_a_bench as rb
import regime_a_diag_suite as ds
import picker_v2 as pv  # cost model (cost, bestP) + measured bwp table (by/best)

CACHE = f"{HERE}/regime_a_campaign_cache.json"
STATE = f"{HERE}/regime_a_campaign_state.json"
S1 = f"{HERE}/regime_a_campaign_baseline.json"
S2 = f"{HERE}/regime_a_campaign_picker.json"
S3 = f"{HERE}/regime_a_campaign_select.json"
S4 = f"{HERE}/regime_a_campaign_ablate.json"
FREQ, PEAK512, TB = rb.FREQ, rb.PEAK512, rb.TB
cdiv = rb.cdiv
LAUNCH_DEADLINE_S = 7 * 3600  # stop launching new measurements ~7h in; leave time to aggregate


def campaign_cache():
    """Resumable cache that NEVER seeds from the old regime_a_bench.json OUT: rb.load_cache seeds stale
    pre-topology-change records when the path is absent, which would silently poison this fresh
    re-baseline. Pre-creating CACHE as {} suppresses that seeding while staying fully resumable."""
    if not os.path.exists(CACHE):
        json.dump({}, open(CACHE, "w"))
    return rb.load_cache(CACHE)


# ---------------------------------------------------------------- corpus (Mt<=8 => M<=256)
# Every Mt<=8 shape currently in REGIME_A_BENCH_REPORT.md.
REPORT_MT8 = [
    (32, 2048, 512),
    (32, 2048, 1536),
    (32, 2048, 2048),
    (32, 256, 6144),
    (32, 6080, 4640),
    (32, 6100, 4608),
    (32, 6144, 1536),
    (32, 6144, 2304),
    (32, 6144, 3072),
    (32, 6144, 4600),
    (32, 6144, 4608),
    (32, 6144, 6144),
    (32, 6144, 9216),
    (48, 6144, 4608),
    (64, 15360, 1536),
    (64, 4608, 6144),
    (64, 6080, 4640),
    (64, 6144, 1536),
    (64, 6144, 4608),
    (64, 6144, 9216),
    (128, 15360, 768),
    (128, 2304, 6144),
    (128, 6080, 4640),
    (128, 6144, 2304),
    (128, 6144, 4608),
    (128, 6144, 768),
    (256, 2048, 1024),
    (256, 6080, 4640),
]
# balanced-tail shapes (rb driver list, Mt<=8 only)
TAIL = [
    (32, 6080, 4640),
    (64, 6080, 4640),
    (128, 6080, 4640),
    (256, 6080, 4640),
    (32, 6144, 4600),
    (32, 6100, 4608),
    (48, 6144, 4608),
]
# structured M-scaling matrix
MATRIX_M = [32, 64, 128, 256]
MATRIX_KN = [
    (2048, 512),
    (2048, 1024),
    (2048, 1536),
    (2048, 2048),
    (6144, 768),
    (6144, 1536),
    (6144, 2304),
    (6144, 4608),
    (6144, 6144),
    (15360, 768),
    (15360, 1536),
    (2304, 6144),
]


def corpus():
    """dict shape -> sorted list of categories (dedup across report/tail/matrix)."""
    cats = {}
    for s in REPORT_MT8:
        cats.setdefault(s, set()).add("report")
    for s in TAIL:
        cats.setdefault(s, set()).add("tail")
    for M in MATRIX_M:
        for kn in MATRIX_KN:
            cats.setdefault((M, kn[0], kn[1]), set()).add("matrix")
    return {s: sorted(c) for s, c in sorted(cats.items())}


# ---------------------------------------------------------------- derived metrics
def _sched_tiles(M, K, N, cfg):
    """Scheduled (padded) tile dims the op actually moves for this cfg."""
    Ns, Pk, Sm, kb, nsb = cfg
    pm = rb.plan_metrics(M, K, N, cfg)
    Kt_sched = pm["Ktl"] * Pk
    Mt_sched = pm["Mblk"] * Sm
    Nt_sched = pm["N_slice"] * Ns * 8
    return Mt_sched, Kt_sched, Nt_sched, pm


def enrich(M, K, N, r):
    """Add DRAM-ideal, wall/ideal, excess us, delivered (scheduled/padded) BW, padding ratio to an ok rec."""
    cfg = tuple(r["cfg"])
    Mt_s, Kt_s, Nt_s, pm = _sched_tiles(M, K, N, cfg)
    Mt, Kt, Nt = cdiv(M, 32), cdiv(K, 32), cdiv(N, 32)
    ideal = pm["ideal_us_512"]
    med = r["us_med"]
    delivered_bytes = (Mt_s * Kt_s + Kt_s * Nt_s + Mt_s * Nt_s) * TB
    logical = pm["logical_bytes"]
    out = dict(r)
    out.update(
        cls="ok",
        ideal_us=ideal,
        wall_over_ideal=(med / ideal if ideal else None),
        excess_us=(med - ideal),
        delivered_gbps=(delivered_bytes / (med / 1e6) / 1e9 if med else None),
        sched_over_valid=(delivered_bytes / logical if logical else None),
        sched_tiles=[Mt_s, Kt_s, Nt_s],
        valid_tiles=[Mt, Kt, Nt],
        l1_pct=pm["l1_pct"],
        N_bpc=pm["N_bpc"],
    )
    return out


# ---------------------------------------------------------------- Stage 1: baseline
def baseline(cache=None, deadline=None):
    cache = cache or campaign_cache()
    cor = corpus()
    out = []
    for i, (shape, cats) in enumerate(cor.items()):
        M, K, N = shape
        if deadline and time.time() > deadline:
            print(f"[baseline] launch deadline hit at {i}/{len(cor)}", flush=True)
            break
        auto = tuple(rb.auto_config(M, K, N))
        r = rb.run_cfg(M, K, N, None, cache)  # product / auto path (public op)
        rec = {"M": M, "K": K, "N": N, "Mt": cdiv(M, 32), "cats": cats, "auto_cfg": list(auto)}
        if rb._ok(r):
            rec.update(enrich(M, K, N, r))
        else:
            rec["cls"] = r.get("cls", "?")
            rec["fail"] = {k: r.get(k) for k in ("cls", "reason", "returncode", "pcc")}
        out.append(rec)
        json.dump(out, open(S1, "w"), indent=2)
        s = (
            f"{rec.get('us_med', 0):.1f}us {rec.get('pct512', 0):.0f}%512 w/i={rec.get('wall_over_ideal', 0):.2f} "
            f"exc={rec.get('excess_us', 0):.1f} cores={rec.get('cores', '?')} pcc={rec.get('pcc', 0):.4f}"
            if rec.get("cls") == "ok"
            else f"[{rec['cls']}]"
        )
        print(f"[baseline {i+1}/{len(cor)}] {M}x{K}x{N} Mt{rec['Mt']} cfg={list(auto)} {s}", flush=True)
    print(f"BASELINE DONE ({len(out)} shapes) -> {S1}", flush=True)
    return out


# ---------------------------------------------------------------- Stage 2: picker candidates
def _prev_measured(shape, topk=8):
    """Prev-measured configs for this shape from the 3262-config oracle, best bwp first (Sm-diverse)."""
    tbl = pv.by.get(shape, {})
    return [c for c, _ in sorted(tbl.items(), key=lambda kv: -kv[1])][:topk]


def _candidates(M, K, N, target=32):
    """Bounded candidate set: picker + prev winner(s) + cost-model top + best-per-(Pk,Ns,Sm) + kb/nsb
    neighbours. Deduped, feasibility-filtered, capped ~target with the principled groups prioritised."""
    shape = (M, K, N)
    feas = [tuple(c) for c in rb.enumerate_feasible(M, K, N)]
    feas_set = set(feas)

    def ok(c):
        c = tuple(c)
        return c in feas_set or rb.planner_feasible(M, K, N, c)[0]

    auto = tuple(rb.auto_config(M, K, N))

    def _cost(c):
        try:
            v = pv.cost(M, K, N, c, pv.bestP)  # None plan (L1-infeasible under pv's model) -> deprioritise
            return v if v is not None else 1e18
        except (TypeError, KeyError, ZeroDivisionError):
            return 1e18

    ranked = sorted(feas, key=_cost)
    # best per distinct (Pk,Ns,Sm) factorization (min cost rep), then the cheapest ~12 factorizations
    per_fact = {}
    for c in ranked:  # ranked asc by cost -> first seen per factorization is its best rep
        Ns, Pk, Sm, kb, nsb = c
        per_fact.setdefault((Pk, Ns, Sm), c)
    fact_reps = sorted(per_fact.values(), key=_cost)[:12]
    # priority order: principled groups first, neighbours fill last; hard cap at `target`.
    groups = [
        ("picker", [auto]),
        ("prev", _prev_measured(shape)),
        ("costtop", ranked[:12]),
        ("factorization", fact_reps),
        ("neighbors", rb.neighbors(auto)),
    ]
    picked, seen, prov = [], set(), {}
    for gname, cs in groups:
        for c in cs:
            c = tuple(c)
            if c in seen or not ok(c) or len(picked) >= target:
                continue
            seen.add(c)
            picked.append(c)
            prov[c] = gname
    return picked, prov, auto


def picker(cache=None, deadline=None):
    cache = cache or campaign_cache()
    cor = corpus()
    out = []
    for i, (shape, cats) in enumerate(cor.items()):
        M, K, N = shape
        if deadline and time.time() > deadline:
            print(f"[picker] launch deadline hit at {i}/{len(cor)}", flush=True)
            break
        cands, prov, auto = _candidates(M, K, N)
        recs = []
        for c in cands:
            if deadline and time.time() > deadline:
                break
            r = rb.run_cfg(M, K, N, c, cache)
            if rb._ok(r):
                recs.append(
                    {
                        "cfg": list(c),
                        "prov": prov[c],
                        "us_med": r["us_med"],
                        "pct512": r["pct512"],
                        "cores": r["cores"],
                        "pcc": r["pcc"],
                    }
                )
        auto_r = rb.run_cfg(M, K, N, tuple(auto), cache)
        auto_us = auto_r["us_med"] if rb._ok(auto_r) else None
        winner = min(recs, key=lambda x: x["us_med"]) if recs else None
        gap = ((auto_us / winner["us_med"] - 1) * 100) if (winner and auto_us) else None
        rec = {
            "M": M,
            "K": K,
            "N": N,
            "Mt": cdiv(M, 32),
            "cats": cats,
            "auto_cfg": list(auto),
            "auto_us": auto_us,
            "n_cands": len(recs),
            "winner": winner,
            "picker_gap_pct": gap,
            "flag_expand": bool(gap is not None and gap > 3.0),
            "cands": sorted(recs, key=lambda x: x["us_med"])[:12],  # keep top-12 for the report
        }
        out.append(rec)
        json.dump(out, open(S2, "w"), indent=2)
        wtxt = f"win={winner['us_med']:.1f}us cfg={winner['cfg']}({winner['prov']})" if winner else "win=-"
        print(
            f"[picker {i+1}/{len(cor)}] {M}x{K}x{N} n={len(recs)} auto={auto_us and round(auto_us,1)} "
            f"{wtxt} gap={gap and round(gap,1)}%{' *EXPAND' if rec['flag_expand'] else ''}",
            flush=True,
        )
    print(f"PICKER DONE -> {S2}", flush=True)
    return out


# ---------------------------------------------------------------- Stage 3: select bottom performers
def select():
    base = json.load(open(S1))
    pick = {(r["M"], r["K"], r["N"]): r for r in json.load(open(S2))} if os.path.exists(S2) else {}
    ok = [r for r in base if r.get("cls") == "ok"]
    for r in ok:
        pg = pick.get((r["M"], r["K"], r["N"]), {})
        r["picker_gap_pct"] = pg.get("picker_gap_pct")
    # four independent rankings
    rk = {
        "wall_over_ideal": sorted(ok, key=lambda r: -(r.get("wall_over_ideal") or 0)),
        "excess_us": sorted(ok, key=lambda r: -(r.get("excess_us") or 0)),
        "low_delivered_bw": sorted(
            [r for r in ok if (r.get("ideal_us") or 0) >= 20], key=lambda r: (r.get("delivered_gbps") or 1e9)
        ),
        "picker_gap": sorted(
            [r for r in ok if r.get("picker_gap_pct") is not None], key=lambda r: -(r.get("picker_gap_pct") or 0)
        ),
    }
    # union of the top-4 of each ranking, then pick <=6 with M-diversity, dropping tiny-ideal dispatch noise
    cand = []
    for name, lst in rk.items():
        for r in lst[:4]:
            cand.append((name, r))
    seen, chosen, bymt = set(), [], {}
    for name, r in cand:
        key = (r["M"], r["K"], r["N"])
        # skip only genuine dispatch-noise: WALL < 8us (fixed launch overhead dominates). Keying on ideal
        # would wrongly drop the worst real shapes (e.g. 256x2048x512: wall 21.6us, ideal 6.7us, 31%512).
        if key in seen or (r.get("us_med") or 0) < 8:
            continue
        if len(chosen) >= 6:
            break
        # M-diversity: cap 2 per Mt until we've covered the range
        mt = r["Mt"]
        if bymt.get(mt, 0) >= 2 and len(set(bymt) | {mt}) < 4:
            continue
        seen.add(key)
        bymt[mt] = bymt.get(mt, 0) + 1
        chosen.append(
            {
                "M": r["M"],
                "K": r["K"],
                "N": r["N"],
                "Mt": r["Mt"],
                "auto_cfg": r["auto_cfg"],
                "reasons": [n for n, rr in cand if (rr["M"], rr["K"], rr["N"]) == key],
                "wall_over_ideal": r.get("wall_over_ideal"),
                "excess_us": r.get("excess_us"),
                "delivered_gbps": r.get("delivered_gbps"),
                "picker_gap_pct": r.get("picker_gap_pct"),
                "us_med": r.get("us_med"),
                "pct512": r.get("pct512"),
            }
        )
    rankings = {
        k: [{"shape": f"{r['M']}x{r['K']}x{r['N']}", "val": v(r)} for r in lst[:8]]
        for k, (lst, v) in {
            "wall_over_ideal": (rk["wall_over_ideal"], lambda r: round(r.get("wall_over_ideal") or 0, 2)),
            "excess_us": (rk["excess_us"], lambda r: round(r.get("excess_us") or 0, 1)),
            "low_delivered_bw": (rk["low_delivered_bw"], lambda r: round(r.get("delivered_gbps") or 0, 0)),
            "picker_gap": (rk["picker_gap"], lambda r: round(r.get("picker_gap_pct") or 0, 1)),
        }.items()
    }
    res = {"rankings": rankings, "selected": chosen}
    json.dump(res, open(S3, "w"), indent=2)
    print("SELECTED bottom performers:", flush=True)
    for c in chosen:
        print(
            f"  {c['M']}x{c['K']}x{c['N']} Mt{c['Mt']} w/i={c['wall_over_ideal']:.2f} exc={c['excess_us']:.1f}us "
            f"bw={c['delivered_gbps']:.0f} gap={c['picker_gap_pct']} reasons={c['reasons']}",
            flush=True,
        )
    return res


# ---------------------------------------------------------------- Stage 4: causal ablations
# (name, diag mask) for the production-kernel diagnostics. Masks per regime_a_matmul_config.hpp.
ABLATE = [
    ("skip_in1_read", 1),
    ("skip_in0_read", 2),
    ("skip_in0_fwd", 4),
    ("no_reduce", 8),
    ("full_in0_wait", 1024),  # DIAG_FULL_IN0_WAIT (default = progressive)
    ("old_barrier_drain", 2048),  # DIAG_BARRIER_DRAIN (default = pipelined)
    ("bank_ring", 4096),  # RING_BANK (default = PARETO)
    ("place_current", 2097152),  # DIAG_PLACE_CURRENT (Sm>1 only; default = IN1_NEAR)
]


def ablate(deadline=None):
    sel = json.load(open(S3))["selected"]
    out = []
    for r in sel:
        M, K, N = r["M"], r["K"], r["N"]
        cfg = tuple(r["auto_cfg"])  # (Ns,Pk,Sm,kb,nsb)
        Sm = cfg[2]
        masks = [("full", 0)] + [(n, m) for n, m in ABLATE if not (n == "place_current" and Sm == 1)]
        res = {}
        for name, mask in masks:
            if deadline and time.time() > deadline:
                break
            runs = [ds.run_one(M, K, N, cfg, mask) for _ in range(2)]  # 2 interleaved-ish relaunches
            oks = [x for x in runs if x.get("ok") and x["wall_us"]]
            med = statistics.median([x["wall_us"] for x in oks]) if oks else None
            risc = oks[0]["risc"] if oks else None
            res[name] = {"mask": mask, "med_us": med, "risc": risc, "n_ok": len(oks)}
        full = res.get("full", {}).get("med_us")
        deltas = {n: ((v["med_us"] / full - 1) * 100 if (full and v.get("med_us")) else None) for n, v in res.items()}
        rec = {
            "M": M,
            "K": K,
            "N": N,
            "Mt": r["Mt"],
            "cfg": list(cfg),
            "full_us": full,
            "abl": res,
            "deltas_pct": deltas,
        }
        out.append(rec)
        json.dump(out, open(S4, "w"), indent=2)
        print(
            f"[ablate] {M}x{K}x{N} full={full and round(full,1)}us "
            + " ".join(f"{n}={deltas[n] and round(deltas[n],0)}%" for n in deltas if n != "full"),
            flush=True,
        )
    print(f"ABLATE DONE -> {S4}", flush=True)
    return out


# ---------------------------------------------------------------- Stage 2b: expand worst picker misses
S5 = f"{HERE}/regime_a_campaign_expand.json"
# (M,K,N,mode): the two most important picker misses. 'full' = whole feasible space (cheap shape);
# 'bounded' = cost-ranked top of the kb<=4 sub-domain (expensive shape, full exhaustive won't fit budget).
EXPAND = [
    (256, 2048, 512, "full"),  # worst abs performer + 24.9% gap; small -> exhaustive
    (256, 15360, 1536, "bounded"),  # biggest gap 26.2%; ~185us/cfg -> bounded cost-ranked expansion
]


def expand(deadline=None):
    cache = campaign_cache()  # already holds every Stage-2 candidate -> those return instantly
    out = []
    for M, K, N, mode in EXPAND:
        auto = tuple(rb.auto_config(M, K, N))
        if mode == "full":
            cfgs = [tuple(c) for c in rb.enumerate_feasible(M, K, N)]
        else:
            allc = [tuple(c) for c in rb.enumerate_feasible(M, K, N, kb_set=(1, 2, 4))]

            def _c(c):
                try:
                    v = pv.cost(M, K, N, c, pv.bestP)
                    return v if v is not None else 1e18
                except (TypeError, KeyError, ZeroDivisionError):
                    return 1e18

            allc.sort(key=_c)
            cfgs = allc[:60]
        if auto not in cfgs:
            cfgs = [auto] + cfgs
        res, n_run = [], 0
        for c in cfgs:
            if deadline and time.time() > deadline:
                print(f"[expand] {M}x{K}x{N} deadline hit after {n_run}/{len(cfgs)}", flush=True)
                break
            cached = f"{M}x{K}x{N}:" + ",".join(map(str, c)) in cache
            r = rb.run_cfg(M, K, N, c, cache)
            n_run += 0 if cached else 1
            if rb._ok(r):
                res.append({"cfg": list(c), "us_med": r["us_med"], "pct512": r["pct512"], "cores": r["cores"]})
        res.sort(key=lambda x: x["us_med"])
        auto_r = rb.run_cfg(M, K, N, auto, cache)
        auto_us = auto_r["us_med"] if rb._ok(auto_r) else None
        best = res[0] if res else None
        gap = ((auto_us / best["us_med"] - 1) * 100) if (best and auto_us) else None
        rec = {
            "M": M,
            "K": K,
            "N": N,
            "mode": mode,
            "n_configs": len(cfgs),
            "n_measured": len(res),
            "auto_cfg": list(auto),
            "auto_us": auto_us,
            "best": best,
            "true_gap_pct": gap,
            "top5": res[:5],
        }
        out.append(rec)
        json.dump(out, open(S5, "w"), indent=2)
        print(
            f"[expand] {M}x{K}x{N} {mode} n={len(res)}/{len(cfgs)} auto={auto_us and round(auto_us,1)} "
            f"best={best['cfg']}@{best['us_med']:.1f}us gap={gap and round(gap,1)}%",
            flush=True,
        )
    print(f"EXPAND DONE -> {S5}", flush=True)
    return out


# ---------------------------------------------------------------- Stage 2c: stability of proposed changes
S6 = f"{HERE}/regime_a_campaign_stability.json"


def stability(relaunches=3, min_gap=3.0, topk=14, deadline=None):
    """Re-measure auto vs the Stage-2 winner for the flagged (>min_gap) shapes with INTERLEAVED cross-launch
    relaunches (Stage 2 measured each candidate once; this establishes whether the gap clears launch noise).
    Uses ds.run_one at mask 0 (public path, no cache) so every relaunch is an independent measurement."""
    pick = json.load(open(S2))
    flagged = [r for r in pick if r.get("winner") and (r.get("picker_gap_pct") or 0) >= min_gap]
    flagged.sort(key=lambda r: -(r["picker_gap_pct"] or 0))
    flagged = flagged[:topk]
    out = []
    for r in flagged:
        M, K, N = r["M"], r["K"], r["N"]
        auto, win = tuple(r["auto_cfg"]), tuple(r["winner"]["cfg"])
        if deadline and time.time() > deadline:
            print(f"[stability] deadline hit before {M}x{K}x{N}", flush=True)
            break
        runs = {"auto": [], "winner": []}
        for _ in range(relaunches):
            for name, cfg in (("auto", auto), ("winner", win)):
                x = ds.run_one(M, K, N, cfg, 0)
                if x.get("ok") and x.get("wall_us"):
                    runs[name].append(x["wall_us"])
        aw, ww = sorted(runs["auto"]), sorted(runs["winner"])
        am = statistics.median(aw) if aw else None
        wm = statistics.median(ww) if ww else None
        gap = ((am / wm - 1) * 100) if (am and wm) else None

        def spr(v):
            return ((max(v) - min(v)) / min(v) * 100) if v else None

        rec = {
            "M": M,
            "K": K,
            "N": N,
            "auto_cfg": list(auto),
            "winner_cfg": list(win),
            "auto_walls": aw,
            "winner_walls": ww,
            "auto_med": am,
            "winner_med": wm,
            "stable_gap_pct": gap,
            "auto_spread_pct": spr(aw),
            "winner_spread_pct": spr(ww),
            "single_run_gap_pct": r.get("picker_gap_pct"),
            "stable": bool(gap is not None and gap > 3.0 and gap > (spr(ww) or 0) + (spr(aw) or 0)),
        }
        out.append(rec)
        json.dump(out, open(S6, "w"), indent=2)
        print(
            f"[stability] {M}x{K}x{N} auto={am and round(am,1)} win={wm and round(wm,1)} "
            f"stable_gap={gap and round(gap,1)}% (single={r.get('picker_gap_pct'):.0f}%) "
            f"spread a/w={spr(aw) and round(spr(aw),0)}/{spr(ww) and round(spr(ww),0)}% "
            f"{'STABLE' if rec['stable'] else 'noisy'}",
            flush=True,
        )
    print(f"STABILITY DONE -> {S6}", flush=True)
    return out


# ---------------------------------------------------------------- Stage 5: aggregate report
REPORT = f"{HERE}/REGIME_A_MT8_CAMPAIGN_REPORT.md"


def _risc_str(pr):
    if not pr:
        return "-"
    return "/".join(f"{k[0]}{round(v,1)}" for k, v in sorted(pr.items()))  # B../N../T..


def _classify(deltas):
    """Heuristic bottleneck label from ablation deltas (negative = that component's cost as upper bound)."""
    d = {k: v for k, v in (deltas or {}).items() if v is not None and k != "full"}
    if not d:
        return "?"
    # component-cost ablations (negative delta = time removed): pick the biggest cost
    costs = {k: -d[k] for k in ("skip_in1_read", "skip_in0_read", "skip_in0_fwd", "no_reduce") if k in d}
    tags = []
    if costs:
        big = max(costs, key=costs.get)
        m = {
            "skip_in1_read": "in1-delivery",
            "skip_in0_read": "in0-delivery",
            "skip_in0_fwd": "in0-forwarding",
            "no_reduce": "reduction/sync",
        }
        if costs[big] > 5:
            tags.append(m[big] + f"({costs[big]:.0f}%)")
    # scheduling-alternative ablations (positive delta = default already better)
    for k, lbl in (
        ("full_in0_wait", "prog-wait-helps"),
        ("old_barrier_drain", "pipe-drain-helps"),
        ("bank_ring", "pareto-helps"),
        ("place_current", "in1near-helps"),
    ):
        if k in d and d[k] > 3:
            tags.append(f"{lbl}(+{d[k]:.0f}%)")
    return ", ".join(tags) if tags else "compute/feed-or-overhead"


def aggregate():
    base = json.load(open(S1)) if os.path.exists(S1) else []
    pick = json.load(open(S2)) if os.path.exists(S2) else []
    sel = json.load(open(S3)) if os.path.exists(S3) else {}
    abl = json.load(open(S4)) if os.path.exists(S4) else []
    pmap = {(r["M"], r["K"], r["N"]): r for r in pick}
    L = []
    L.append("# Regime-A Mt<=8 re-baseline campaign (evidence only, no production changes)\n")
    L.append(
        f"Corpus: {len(base)} Mt<=8 shapes (REGIME_A_BENCH_REPORT + balanced tails + M-scaling matrix). "
        "Fresh measurements (campaign cache never seeded from the pre-topology-change bench cache). "
        "8 timed iters after 1 warmup; median kernel us; PCC vs torch fp32.\n"
    )

    # ---- Stage 1: full-corpus baseline ----
    L.append("## 1. Full-corpus baseline (auto-picker)\n")
    L.append(
        "| shape | Mt | cfg (Ns,Pk,Sm,kb,nsb) | cores | us | %512 | eff/deliv GB/s | ideal us | "
        "wall/ideal | excess us | per-RISC us | core-spread% | pcc | cls |"
    )
    L.append("|---|--|--|--|--|--|--|--|--|--|--|--|--|--|")
    ok = [r for r in base if r.get("cls") == "ok"]
    for r in sorted(base, key=lambda r: (r["Mt"], r["K"], r["N"])):
        if r.get("cls") != "ok":
            L.append(
                f"| {r['M']}x{r['K']}x{r['N']} | {r['Mt']} | {r['auto_cfg']} | | | | | | | | | | | **{r.get('cls')}** |"
            )
            continue
        L.append(
            f"| {r['M']}x{r['K']}x{r['N']} | {r['Mt']} | {r['auto_cfg']} | {r['cores']} | {r['us_med']:.1f} | "
            f"{r['pct512']:.0f} | {r['eff_gbps']:.0f}/{r['delivered_gbps']:.0f} | {r['ideal_us']:.1f} | "
            f"{r['wall_over_ideal']:.2f} | {r['excess_us']:.1f} | {_risc_str(r.get('per_risc_us'))} | "
            f"{(r.get('core_spread_pct') or 0):.0f} | {r['pcc']:.4f} | ok |"
        )
    if ok:
        import statistics as st

        L.append(
            f"\n**Summary:** median %512 = {st.median([r['pct512'] for r in ok]):.0f}%, "
            f"median wall/ideal = {st.median([r['wall_over_ideal'] for r in ok]):.2f}; "
            f"{sum(1 for r in base if r.get('cls')!='ok')} non-ok.\n"
        )

    # ---- Stage 2: picker-gap / proposed-change table ----
    L.append("## 2. Picker quality (bounded 25-40 candidates/shape)\n")
    L.append(
        "Proposed-change table: shapes where a measured candidate beat the auto-picker. "
        "'*' = gap >3% (search was flagged for expansion). NO picker edits made.\n"
    )
    L.append("| shape | Mt | picker cfg | picker us | winner cfg (prov) | winner us | gap% | n_cand |")
    L.append("|---|--|--|--|--|--|--|--|")
    gaps = [r for r in pick if r.get("winner") and r.get("picker_gap_pct") is not None]
    for r in sorted(gaps, key=lambda r: -(r["picker_gap_pct"] or 0)):
        w = r["winner"]
        star = " *" if r.get("flag_expand") else ""
        L.append(
            f"| {r['M']}x{r['K']}x{r['N']} | {r['Mt']} | {r['auto_cfg']} | "
            f"{r['auto_us'] and round(r['auto_us'],1)} | {w['cfg']} ({w['prov']}) | {w['us_med']:.1f} | "
            f"{r['picker_gap_pct']:.1f}{star} | {r['n_cands']} |"
        )
    big = [r for r in gaps if (r["picker_gap_pct"] or 0) > 3]
    L.append(
        f"\n**{len(big)} shapes** with >3% picker gap"
        + (
            ": "
            + ", ".join(
                f"{r['M']}x{r['K']}x{r['N']}({r['picker_gap_pct']:.0f}%)"
                for r in sorted(big, key=lambda r: -(r["picker_gap_pct"] or 0))[:12]
            )
            if big
            else ""
        )
        + ".\n"
    )

    # ---- Stage 3: bottom performers ----
    if sel:
        L.append("## 3. Bottom performers (multi-criteria)\n")
        for name, lst in sel.get("rankings", {}).items():
            L.append(f"- **{name}** top: " + ", ".join(f"{x['shape']}={x['val']}" for x in lst[:6]))
        L.append("\nSelected for ablation (<=6, M-diverse, dispatch-noise dropped):")
        for c in sel.get("selected", []):
            L.append(
                f"- `{c['M']}x{c['K']}x{c['N']}` Mt{c['Mt']} wall/ideal={c['wall_over_ideal']:.2f} "
                f"excess={c['excess_us']:.1f}us deliv={c['delivered_gbps']:.0f}GB/s gap={c['picker_gap_pct']} "
                f"({', '.join(sorted(set(c['reasons']))) })"
            )
        L.append("")

    # ---- Stage 4: ablations ----
    if abl:
        L.append("## 4. Causal ablations on bottom performers\n")
        L.append(
            "Deltas vs full (mask 0). Component-skip (negative) = that component's cost as an UPPER "
            "BOUND (scheduling confounds preserved). Scheduling-alt (positive) = current default wins.\n"
        )
        L.append(
            "| shape | cfg | full us | skip in1 | skip in0 | skip fwd | no reduce | full-wait | "
            "old-drain | bank-ring | place-cur | bottleneck |"
        )
        L.append("|---|--|--|--|--|--|--|--|--|--|--|--|")
        for r in abl:
            d = r["deltas_pct"]

            def g(k):
                v = d.get(k)
                return f"{v:+.0f}%" if v is not None else "-"

            L.append(
                f"| {r['M']}x{r['K']}x{r['N']} | {r['cfg']} | {r['full_us'] and round(r['full_us'],1)} | "
                f"{g('skip_in1_read')} | {g('skip_in0_read')} | {g('skip_in0_fwd')} | {g('no_reduce')} | "
                f"{g('full_in0_wait')} | {g('old_barrier_drain')} | {g('bank_ring')} | {g('place_current')} | "
                f"{_classify(d)} |"
            )
        L.append("")

    # ---- Stage 2b: exhaustive expansion of worst misses ----
    exp = json.load(open(S5)) if os.path.exists(S5) else []
    if exp:
        L.append("## 2b. Exhaustive/expanded sweep of the worst picker misses\n")
        L.append("| shape | mode | configs | measured | auto cfg | auto us | true-best cfg | best us | true gap% |")
        L.append("|---|--|--|--|--|--|--|--|--|")
        for r in exp:
            b = r.get("best") or {}
            L.append(
                f"| {r['M']}x{r['K']}x{r['N']} | {r['mode']} | {r['n_configs']} | {r['n_measured']} | "
                f"{r['auto_cfg']} | {r['auto_us'] and round(r['auto_us'],1)} | {b.get('cfg')} | "
                f"{b.get('us_med') and round(b['us_med'],1)} | {r['true_gap_pct'] and round(r['true_gap_pct'],1)} |"
            )
        L.append("")

    # ---- Stage 2c: stability of proposed changes (cross-launch relaunches) ----
    stab = json.load(open(S6)) if os.path.exists(S6) else []
    if stab:
        L.append("## 2c. Proposed picker changes — stability (interleaved cross-launch relaunches)\n")
        L.append(
            "Auto vs Stage-2 winner, medians over independent relaunches; spread = (max-min)/min per arm. "
            "STABLE = stable gap >3% and exceeding combined launch spread. **No picker edits made** — "
            "this is the proposed-change table for review.\n"
        )
        L.append(
            "| shape | auto cfg | proposed cfg | auto us | proposed us | stable gap% | single-run gap% | "
            "spread a/w % | verdict |"
        )
        L.append("|---|--|--|--|--|--|--|--|--|")
        for r in sorted(stab, key=lambda r: -(r.get("stable_gap_pct") or 0)):
            L.append(
                f"| {r['M']}x{r['K']}x{r['N']} | {r['auto_cfg']} | {r['winner_cfg']} | "
                f"{r['auto_med'] and round(r['auto_med'],1)} | {r['winner_med'] and round(r['winner_med'],1)} | "
                f"{r['stable_gap_pct'] and round(r['stable_gap_pct'],1)} | "
                f"{r['single_run_gap_pct'] and round(r['single_run_gap_pct'],1)} | "
                f"{r['auto_spread_pct'] and round(r['auto_spread_pct'],0)}/"
                f"{r['winner_spread_pct'] and round(r['winner_spread_pct'],0)} | "
                f"{'**STABLE**' if r.get('stable') else 'noisy'} |"
            )
        nst = sum(1 for r in stab if r.get("stable"))
        L.append(f"\n**{nst}/{len(stab)}** proposed changes are STABLE (clear launch noise).\n")

    # ---- Stage 5: recommendation ----
    L.append("## 5. Recommended next optimization (causal)\n")
    # data-driven facts
    smflip = [
        r
        for r in pick
        if r.get("winner") and (r["picker_gap_pct"] or 0) > 3 and r["winner"]["cfg"][2] > r["auto_cfg"][2]
    ]
    nsbfix = [
        r
        for r in pick
        if r.get("winner")
        and (r["picker_gap_pct"] or 0) > 3
        and r["winner"]["cfg"][2] == r["auto_cfg"][2] == 1
        and (r["winner"]["cfg"][4] != r["auto_cfg"][4] or r["winner"]["cfg"][3] != r["auto_cfg"][3])
    ]
    L.append(
        f"**Efficiency vs Mt** (median %512): "
        + ", ".join(
            f"Mt{mt}={statistics.median([r['pct512'] for r in ok if r['Mt']==mt]):.0f}%"
            for mt in (1, 2, 4, 8)
            if [r for r in ok if r["Mt"] == mt]
        )
        + " — efficiency falls sharply into the Mt=8 regime.\n"
    )
    L.append(
        "**Primary finding — the picker's cost-model fallback systematically mis-picks on the M-scaling "
        "shapes (which use the fallback, not the 20-shape lookup table):**\n"
        f"1. **Prefer M-split (Sm>1) over deep split-K on low-arithmetic-intensity Mt>=4 shapes.** "
        f"{len(smflip)} flagged shapes want a higher Sm than the picker chose (e.g. 256x2048x512 Sm1->Sm3 "
        "+25%, 256x15360x1536 Sm1->Sm2 +26%). Ablation root cause: at high Pk the split-K **reduction** "
        "dominates (no_reduce -26..-38% on the Pk8 shapes) and in0-**forwarding** is heavy (skip_in0_fwd "
        "-19..-30%); trading K-split for M-split (now cheap post-in1_near) removes both. This is the single "
        "biggest lever.\n"
        f"2. **Prefer kb>=2 and nsb>=3 on deep-K / wide shapes.** {len(nsbfix)} flagged Sm=1 shapes want "
        "deeper K-blocks / more N sub-blocking than the fallback's kb=1,nsb=1 (e.g. 32x15360x768 "
        "[1,12,1,1,1]->[1,6,1,2,3] +24%: ablation shows this shape is **in1-read bound (skip_in1_read "
        "-73%)** and kb>=2 cuts per-block read overhead; 256x2304x6144 nsb 1->3 +22%).\n"
        "\n**These are PICKER (offline cost-model) changes, not kernel changes** — the fixes are already "
        "expressible in the existing config space. Recommend: (a) add an M-split branch to the cost-model "
        "for low-AI Mt>=4, (b) raise the kb/nsb floor in the fallback, (c) extend the lookup table to the "
        "M-scaling shapes with the measured winners. Land only after a stability + full-corpus regression "
        "pass (this campaign did not edit the picker).\n"
        "\n**CB1-depth experiment (optimization #5): NOT warranted by the evidence.** in1 backpressure is "
        "not the exposed bottleneck — the in1-bound shape (32x15360x768) is read-*volume* bound (fixed by "
        "kb>=2, a config choice), and the low-AI shapes are reduction/forwarding bound (skip_in1_read only "
        "-3..-12% there). A deeper CB1 would not help a shape whose critical path IS the DRAM read. "
        "Deferred unless a future shape shows in1 backpressure with compute headroom.\n"
        "\n**Padding (optimization #6):** balanced-tail shapes show sched/valid ~1.0-1.06 (small); the "
        "Section-1 table's delivered-vs-effective BW gap is minor, so skipping padded compute is low-value "
        "for this corpus.\n"
    )

    open(REPORT, "w").write("\n".join(L) + "\n")
    print(f"AGGREGATE DONE -> {REPORT}", flush=True)
    return REPORT


def state_start():
    st = {"start": time.time()}
    if os.path.exists(STATE):
        try:
            st = json.load(open(STATE))
        except Exception:
            pass
    st.setdefault("start", time.time())
    json.dump(st, open(STATE, "w"))
    return st["start"]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if mode == "all":
        start = state_start()
        cache = campaign_cache()
        # Reserve time so select+ablate always run: baseline uncapped (cheap+top priority), picker until
        # 6h, ablate until 6.9h. Aggregate (host-side, no device) runs afterwards regardless.
        baseline(cache, start + LAUNCH_DEADLINE_S)
        picker(cache, start + 6 * 3600)
        select()
        ablate(start + int(6.9 * 3600))
        aggregate()
        print("CAMPAIGN DONE (all phases + aggregate)", flush=True)
    elif mode == "expand":
        start = state_start()
        expand(start + LAUNCH_DEADLINE_S)
    elif mode == "stability":
        start = state_start()
        stability(deadline=start + LAUNCH_DEADLINE_S)
    else:
        {"baseline": baseline, "picker": picker, "select": select, "ablate": ablate, "aggregate": aggregate}[mode]()
