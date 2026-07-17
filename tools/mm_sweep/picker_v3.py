#!/usr/bin/env python3
# Improved regime_a_matmul FALLBACK cost model (picker v3). Over v2 it:
#   - ENUMERATES Sm>1 (v2 fallback was Sm=1 only) -> can pick M-split.
#   - models split-K REDUCTION cost (penalise large Pk) and M-split in1-FORWARDING cost (penalise large Sm),
#     so the reduction/forwarding trade the campaign exposed is represented.
#   - models per-BLOCK read startup (nblocks ~ (Ktl/kb)*Nbpc) so small kb (more blocks) is penalised ->
#     better kb/nsb ranking.
# Trained on the Mt<=8 campaign measurements (regime_a_campaign_cache.json) with a held-out validation
# subset. Prints trained params (for the C++ port) + train/val regret vs the current picker.
#
# Regret(shape) = measured_us[ argmin_cost over MEASURED configs ] / best measured_us  (>=1.0, lower better).
# Restricting the argmin to measured configs mirrors picker_v2.evalP (we can only score what we measured;
# the campaign's candidate sets were cost-guided + factorization-diverse so they cover the good region).
import json, os, re, math, itertools, statistics, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regime_a_bench as rb

HERE = os.path.dirname(os.path.abspath(__file__))
cdiv = rb.cdiv


# ---------------------------------------------------------------- training data (measured us per cfg)
def load_measured(cache_path=None):
    c = json.load(open(cache_path or f"{HERE}/regime_a_campaign_cache.json"))
    by, auto_us = defaultdict(dict), {}
    for k, v in c.items():
        if not isinstance(v, dict) or v.get("cls") != "ok":
            continue
        m = re.match(r"(\d+)x(\d+)x(\d+):(.+)", k)
        if not m:
            continue
        M, K, N = int(m.group(1)), int(m.group(2)), int(m.group(3))
        tag = m.group(4)
        if tag == "auto":
            auto_us[(M, K, N)] = v["us_med"]
        else:
            by[(M, K, N)][tuple(int(x) for x in tag.split(","))] = v["us_med"]
    return by, auto_us


def load_current(by, auto_us):
    """Ground-truth DEPLOYED picks/us: the baseline S1 auto_cfg (measured via the real C++ auto_select_config).
    Falls back to rb.auto_config (verified identical to S1) if S1 is absent."""
    cur_pick, cur_us = {}, {}
    s1p = f"{HERE}/regime_a_campaign_baseline.json"
    if os.path.exists(s1p):
        for r in json.load(open(s1p)):
            if r.get("cls") == "ok":
                s = (r["M"], r["K"], r["N"])
                cur_pick[s] = tuple(r["auto_cfg"])
                cur_us[s] = r["us_med"]
    for s in by:
        if s not in cur_pick:
            cur_pick[s] = tuple(rb.auto_config(*s))
            cur_us[s] = by[s].get(cur_pick[s], auto_us.get(s))
    return cur_pick, cur_us


# ---------------------------------------------------------------- geometry + parametric cost
def geo(M, K, N, cfg):
    """Geometry mirroring the C++ pick_plan; returns None if infeasible."""
    if not rb.planner_feasible(M, K, N, cfg)[0]:
        return None
    Ns, Pk, Sm, kb, nsb = cfg
    pm = rb.plan_metrics(M, K, N, cfg)
    Kt, Nt = cdiv(K, 32), cdiv(N, 32)
    wasteK = pm["k_pad_tiles"] / Kt
    Nt_sched = pm["N_slice"] * Ns * 8
    wasteN = Nt_sched / Nt - 1.0
    return dict(
        cores=pm["cores"],
        Ktl=pm["Ktl"],
        Mblk=pm["Mblk"],
        Nown=pm["N_own"],
        Nbpc=pm["N_bpc"],
        wasteK=wasteK,
        wasteN=wasteN,
        Kt=Kt,
        Nt=Nt,
    )


_GEO = {}


def geo_m(M, K, N, cfg):
    """Memoised geo() so training (millions of cost() calls) does arithmetic only, not plan_metrics."""
    k = (M, K, N, cfg)
    if k not in _GEO:
        _GEO[k] = geo(M, K, N, cfg)
    return _GEO[k]


def cost(M, K, N, cfg, P, g=None):
    g = g or geo_m(M, K, N, cfg)
    if g is None:
        return 1e18
    Ns, Pk, Sm, kb, nsb = cfg
    readT = g["Kt"] * g["Nt"] / min(g["cores"], P["Csat"])
    comp_pc = g["Mblk"] * g["Nown"] * g["Ktl"]
    area = min(g["Mblk"] * nsb, P["acap"])
    kbe = min(kb, P["kbcap"])
    compT = comp_pc / ((kbe / (kbe + P["kk"])) * (area / (area + P["aa"])))
    ovlT = P["ovl"] * comp_pc / g["Nbpc"]
    nblocks = (g["Ktl"] / kb) * g["Nbpc"]  # per-block read startup count (small kb -> more)
    startT = P["start"] * nblocks
    reduceT = P["rk"] * max(Pk - 1, 0) * (g["Mblk"] * g["Nown"])  # split-K reduction over out tiles/core
    fwdT = P["fm"] * max(Sm - 1, 0) * (g["Ktl"] * g["Nown"])  # in1 forwarding volume (reader shard)
    base = max(readT, compT) + ovlT + startT + reduceT + fwdT
    return base * (1.0 + P["wst"] * (g["wasteK"] + g["wasteN"]))


# ---------------------------------------------------------------- eval / train
def pick_over(cfgs, M, K, N, P):
    return min(cfgs, key=lambda c: cost(M, K, N, c, P))


def pick_hyst(cfgs, M, K, N, P, margin, anchor):
    """HYSTERESIS fallback: anchor on the deployed Sm=1 pick (`anchor`); switch to the best Sm>1 candidate
    ONLY if (a) the shape is NARROW-N (Nband<=nband_max) -- where N-split can't supply parallelism so
    M-split is the lever (the campaign's clean win/loss discriminator), and (b) its reduction/forwarding-
    aware cost beats the anchor's by `margin`. Otherwise pick==anchor (deployed) -> zero regression."""
    if cdiv(N, 32) and cdiv(cdiv(N, 32), 8) > P.get("nband_max", 99):
        return anchor
    smG = [c for c in cfgs if c[2] > 1]
    if not smG:
        return anchor
    bestG = min(smG, key=lambda c: cost(M, K, N, c, P))
    if cost(M, K, N, bestG, P) < cost(M, K, N, anchor, P) * (1.0 - margin):
        return bestG
    return anchor


def regret(by, shapes, P):
    """Geomean + worst regret over `shapes`, choosing among each shape's MEASURED configs."""
    regs = []
    for s in shapes:
        meas = by[s]
        if len(meas) < 2:
            continue
        best = min(meas.values())
        pick = pick_over(list(meas), *s, P)
        regs.append(meas[pick] / best)
    gm = math.exp(sum(math.log(r) for r in regs) / len(regs))
    return gm, max(regs), regs


# Base params PINNED to the deployed C++ cost model. The hysteresis fallback anchors on the deployed Sm=1
# pick, so Sm=1 behaviour is byte-identical; only (rk, fm, margin) — the Sm>1 decision — is trained.
GRID = dict(
    Csat=[24],
    kk=[0.5],
    aa=[2.0],
    acap=[6],
    kbcap=[2],
    ovl=[1.0],
    wst=[0.5],
    start=[0.0],
    rk=[0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8],  # split-K reduction weight (favours M-split at high Pk)
    fm=[0.0, 0.005, 0.01, 0.02, 0.04, 0.08],  # in1 forwarding weight (penalises over-splitting M)
    margin=[0.03, 0.05, 0.08, 0.12, 0.18],  # hysteresis: Sm>1 must beat the anchor cost by this
    nband_max=[2, 4, 6, 8, 12],  # NARROW-N guard: Sm>1 only for cdiv(Nt,8) <= this
)


def eval_hyst(by, cur_pick, cur_us, shapes, P):
    """Evaluate the hysteresis fallback over `shapes`. Returns counts + magnitudes. A shape only deviates
    from the deployed pick when the Sm>1 branch fires -> regressions come solely from bad firings."""
    margin = P["margin"]
    n_reg = n_win = 0
    reg_mag = win_mag = 0.0
    regs, fired = [], []
    for s in shapes:
        meas = by[s]
        if len(meas) < 2 or s not in cur_pick:
            continue
        anchor = cur_pick[s]
        pick = pick_hyst(list(meas), *s, P, margin, anchor)
        best = min(meas.values())
        u = meas.get(pick, cur_us[s])
        uc = cur_us[s]
        regs.append(u / best if best else 1.0)
        if pick != anchor:
            fired.append((s, anchor, cur_us[s], pick, u))
            if uc and u > uc * 1.03:
                n_reg += 1
                reg_mag += u / uc - 1.0
            elif uc and u < uc * 0.97:
                n_win += 1
                win_mag += uc / u - 1.0
    gm = math.exp(sum(math.log(r) for r in regs) / len(regs)) if regs else 1.0
    return dict(n_reg=n_reg, reg_mag=reg_mag, n_win=n_win, win_mag=win_mag, gm=gm, fired=fired)


def train(by, cur_pick, cur_us, train_shapes):
    """Lexicographic: (min hard-regressions, MAX win magnitude, min geomean regret)."""
    keys = list(GRID)
    best_key, bestP = (10**9, 0.0, 10**9), None
    for vals in itertools.product(*[GRID[k] for k in keys]):
        P = dict(zip(keys, vals))
        e = eval_hyst(by, cur_pick, cur_us, train_shapes, P)
        key = (e["n_reg"], -round(e["win_mag"], 4), e["gm"])
        if key < best_key:
            best_key, bestP = key, P
    return bestP, best_key


def emit_table(by, cur_pick, cur_us, min_gain=0.03):
    """Propose lookup-table entries: for every shape whose measured-best beats the deployed pick by
    >min_gain, emit (Mt,Kt,Nt)->best_cfg, tagged by confirmation source (stability / expand / validate /
    single-run). This is the ROBUST no-regression win mechanism (measured winners, not cost-model picks)."""
    # confirmation sources
    stab = {}
    p6 = f"{HERE}/regime_a_campaign_stability.json"
    if os.path.exists(p6):
        for r in json.load(open(p6)):
            if r.get("stable") and (r.get("stable_gap_pct") or 0) > 3:
                stab[(r["M"], r["K"], r["N"])] = tuple(r["winner_cfg"])
    exp = {}
    p5 = f"{HERE}/regime_a_campaign_expand.json"
    if os.path.exists(p5):
        for r in json.load(open(p5)):
            if r.get("best"):
                exp[(r["M"], r["K"], r["N"])] = tuple(r["best"]["cfg"])
    valc = {}
    pv = f"{HERE}/regime_a_campaign_validate.json"
    if os.path.exists(pv):
        for r in json.load(open(pv)):
            if r.get("cand_gain_pct", 0) and r["cand_gain_pct"] > 3:
                valc[(r["M"], r["K"], r["N"])] = tuple(r["cand_cfg"])
    rows = []
    for s in sorted(by):
        M, K, N = s
        meas = by[s]
        if s not in cur_us or not cur_us[s]:
            continue
        best_cfg = min(meas, key=meas.get)
        best_us = meas[best_cfg]
        # prefer a confirmed cfg (expand true-best / validate / stability) over the raw sampled best
        cfg = exp.get(s) or valc.get(s) or stab.get(s) or best_cfg
        us = meas.get(cfg, best_us)
        gain = cur_us[s] / us - 1.0
        if gain <= min_gain:
            continue
        src = "expand" if s in exp else "validate" if s in valc else "stability" if s in stab else "single"
        rows.append((gain, s, cur_pick[s], cur_us[s], cfg, us, src))
    rows.sort(key=lambda r: -r[0])
    print("\n=== PROPOSED LOOKUP-TABLE ENTRIES (measured winners, gain>3% vs deployed) ===")
    print("C++ kTable  (Mt,Kt,Nt) -> {Pk,Ns,Sm,kb,nsb}   [cfg tuple is (Ns,Pk,Sm,kb,nsb)]")
    for gain, s, cp, cu, cfg, us, src in rows:
        Mt, Kt, Nt = cdiv(s[0], 32), cdiv(s[1], 32), cdiv(s[2], 32)
        Ns, Pk, Sm, kb, nsb = cfg
        print(
            f"  {{{{{Mt}, {Kt}, {Nt}}}, {{{Pk}, {Ns}, {Sm}, {kb}, {nsb}}}}},  "
            f"// {s[0]}x{s[1]}x{s[2]} {cu:.1f}->{us:.1f}us +{gain*100:.0f}% [{src}]"
        )
    json.dump(
        [[list(s), list(cfg), cu, us, round(gain * 100, 1), src] for gain, s, cp, cu, cfg, us, src in rows],
        open(f"{HERE}/picker_v3_table.json", "w"),
        indent=2,
    )
    print(f"\n{len(rows)} entries -> picker_v3_table.json")
    return rows


if __name__ == "__main__":
    by, auto_us = load_measured()
    cur_pick, cur_us = load_current(by, auto_us)
    if len(sys.argv) > 1 and sys.argv[1] == "table":
        emit_table(by, cur_pick, cur_us)
        sys.exit(0)
    shapes = sorted([s for s in by if len(by[s]) >= 2 and s in cur_pick])
    val = [s for i, s in enumerate(shapes) if i % 3 == 2]  # deterministic 1/3 holdout
    trn = [s for i, s in enumerate(shapes) if i % 3 != 2]
    print(f"shapes={len(shapes)} train={len(trn)} val={len(val)}", flush=True)

    bestP, tkey = train(by, cur_pick, cur_us, trn)
    print(f"\nTRAINED PARAMS (base pinned to deployed): rk={bestP['rk']} fm={bestP['fm']} margin={bestP['margin']}")
    for label, sh in (("train", trn), ("val", val), ("all", shapes)):
        e = eval_hyst(by, cur_pick, cur_us, sh, bestP)
        print(
            f"  {label:5}: fired={len(e['fired'])} wins(>3%)={e['n_win']} (mag {e['win_mag']*100:.0f}%) "
            f"REGRESSIONS(>3%)={e['n_reg']} (mag {e['reg_mag']*100:.0f}%) geomean_regret={e['gm']:.4f}"
        )

    e = eval_hyst(by, cur_pick, cur_us, shapes, bestP)
    print("\n=== hysteresis fallback firings (Sm>1 chosen over deployed anchor) ===")
    for s, anchor, ua, pick, u in sorted(e["fired"], key=lambda x: (x[2] / x[4] if x[4] else 0)):
        tag = "VAL" if s in val else "trn"
        gain = (ua / u - 1) * 100 if u else 0
        flag = " <-REGRESSION" if u > ua * 1.03 else (" WIN" if u < ua * 0.97 else "")
        print(
            f"  [{tag}] {s[0]}x{s[1]}x{s[2]} anchor={list(anchor)}@{ua:.1f} -> Sm>1={list(pick)}@{u:.1f} "
            f"gain={gain:.1f}%{flag}"
        )
    json.dump(
        {
            "params": bestP,
            "train_shapes": [list(s) for s in trn],
            "val_shapes": [list(s) for s in val],
            "fired": [[list(s), list(a), ua, list(p), u] for s, a, ua, p, u in e["fired"]],
        },
        open(f"{HERE}/picker_v3_trained.json", "w"),
        indent=2,
    )
    print("\nWROTE picker_v3_trained.json")
