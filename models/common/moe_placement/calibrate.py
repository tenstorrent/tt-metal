# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Calibrate per-layer expert placements on TRAIN traffic and score every plan on held-out TEST traffic.

Plans (all with the same optimizer and constraint):
  bal{32,64,128,256}_s1, bal128_s2   balanced mix; data-size ablation and seed stability
  lodo_<domain>                     balanced mix with that domain left out
  lolo_<group>                      balanced mix with one language group left out
  indom_<domain>                    that domain only (in-domain upper reference)
  identity                          the original layout; extra plans can be passed in (e.g. a committed preset)
Test sets (TEST requests only): balanced mix, source-uniform mix, one per domain, one per language group with
enough tokens, plus every control/edge capture scored whole.

Metric per (plan, test set), summed over all test chunks and MoE layers:
  peak tiles          busiest chip's routed expert work, in 32-token tiles per expert (the critical path)
  peak_change_pct     100 * (plan peak / identity peak - 1)        negative = less critical-path work
  utilization         mean chip tiles / busiest chip tiles          1.0 = perfectly even
  hops_change_pct     dispatch + combine payload-page hops vs identity
Peak tiles rank plans within one model; they overstate on-device gains by a model-dependent factor
(measured 0.22x for GLM-5.3 and 0.10x for Kimi-K3), so the on-device A/B decides.

Optimizer: per layer and dispatch group, simulated annealing over expert swaps between rows, scoring busiest
chip tiles (1.0) + busiest dispatch link (0.25) + busiest combine link (0.25) + total hops (0.05), each
normalized to the original layout. Starts from the original layout and from a greedy load-balanced layout.
Constraints: "aggregate" (default) - sums over all training chunks may not get worse than the original;
"frame" - no training chunk may get worse (lossless per chunk; keeps little of the gain); "cap10" -
aggregate plus no chunk worse than the original by more than 10%.
"""

from __future__ import annotations

import csv
import json
import time
import zlib
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from . import traffic
from .corpus import DOMAINS
from .spec import MoeSpec
from .transport import contributions

FULL_PLANS = "full"
_CTX: dict = {}  # shared with forked workers


def _state(totals, tiles, d, c, m, rows, gs):
    load = np.stack([totals[:, m == r].sum(1) for r in range(rows)], 1)
    tl = np.stack([tiles[:, m == r].sum(1) for r in range(rows)], 1)
    return load, tl, d[:, np.arange(gs), m].sum(1), c[:, np.arange(gs), m].sum(1)


def optimize(counts: np.ndarray, rng, spec: MoeSpec, steps: int, constraint: str = "aggregate") -> np.ndarray:
    """counts [chunks, rows, experts] (one layer) -> owners[e]."""
    rows, groups, epc, gs = spec.rows, spec.groups, spec.experts_per_chip, spec.group_size
    counts = counts.astype(np.int64)
    owners = spec.identity_owners().copy()
    totals_all = counts.sum(1)
    tiles_all = (totals_all + 31) // 32
    f = counts.shape[0]
    raw_limit = totals_all.reshape(f, groups, rows, epc).sum(-1).max((1, 2))
    tile_limit = tiles_all.reshape(f, groups, rows, epc).sum(-1).max((1, 2))
    cap = 1.10 if constraint == "cap10" else None
    for g in range(groups):
        cg = counts[:, :, g * gs : (g + 1) * gs]
        totals = cg.sum(1)
        tiles = (totals + 31) // 32
        dd, cc = contributions(cg, spec.transport, spec.num_links)
        base = np.arange(gs) // epc
        init = _state(totals, tiles, dd, cc, base, rows, gs)
        dp = np.maximum(init[2].max(1), 1)
        cp = np.maximum(init[3].max(1), 1)
        hp = np.maximum((init[2] + init[3]).sum(1), 1)
        tp = np.maximum(init[1].max(1), 1)

        def score(t, d, r):
            return float(
                np.mean(t.max(1) / tp + 0.25 * d.max(1) / dp + 0.25 * r.max(1) / cp + 0.05 * (d + r).sum(1) / hp)
            )

        def caps_ok(l, t):
            if constraint == "frame":
                return bool(np.all(l.max(1) <= raw_limit) and np.all(t.max(1) <= tile_limit))
            ok = l.max(1).sum() <= raw_limit.sum() and t.max(1).sum() <= tile_limit.sum()
            return bool(
                ok and (cap is None or (np.all(l.max(1) <= raw_limit * cap) and np.all(t.max(1) <= tile_limit * cap)))
            )

        def feasible(l, t, d, r):
            if not caps_ok(l, t):
                return False
            if constraint == "frame":
                return bool(
                    np.all(d.max(1) <= dp * 1.02)
                    and np.all(r.max(1) <= cp * 1.02)
                    and np.all((d + r).sum(1) <= hp * 1.02)
                )
            ok = (
                d.max(1).sum() <= dp.sum() * 1.02
                and r.max(1).sum() <= cp.sum() * 1.02
                and (d + r).sum() <= hp.sum() * 1.02
            )
            return bool(ok and (cap is None or (np.all(d.max(1) <= dp * cap) and np.all(r.max(1) <= cp * cap))))

        greedy = np.empty(gs, np.int64)
        occ = np.zeros(rows, np.int64)
        cost = np.zeros(rows)
        costs = tiles.mean(0)
        for e in np.argsort(-costs, kind="stable"):
            r = min((r for r in range(rows) if occ[r] < epc), key=lambda r: (cost[r], r))
            greedy[e] = r
            occ[r] += 1
            cost[r] += costs[e]
        best, best_score = base.copy(), score(*init[1:])
        for start in (base, greedy):
            m = start.copy()
            l, t, d, r = _state(totals, tiles, dd, cc, m, rows, gs)
            if not feasible(l, t, d, r):
                m = base.copy()
                l, t, d, r = _state(totals, tiles, dd, cc, m, rows, gs)
            cur = score(t, d, r)
            if cur < best_score:
                best, best_score = m.copy(), cur
            for step in range(steps):
                i, j = rng.integers(0, gs, 2)
                oi, oj = m[i], m[j]
                if oi == oj:
                    continue
                nl = l.copy()
                dl = totals[:, j] - totals[:, i]
                nl[:, oi] += dl
                nl[:, oj] -= dl
                nt = t.copy()
                dt = tiles[:, j] - tiles[:, i]
                nt[:, oi] += dt
                nt[:, oj] -= dt
                if not caps_ok(nl, nt):
                    continue
                nd = d + dd[:, i, oj] - dd[:, i, oi] + dd[:, j, oi] - dd[:, j, oj]
                nr = r + cc[:, i, oj] - cc[:, i, oi] + cc[:, j, oi] - cc[:, j, oj]
                if not feasible(nl, nt, nd, nr):
                    continue
                prop = score(nt, nd, nr)
                temp = 0.005 * (1 - step / steps) ** 2
                if prop < cur or (temp > 0 and rng.random() < np.exp(min(0, (cur - prop) / temp))):
                    m[i], m[j] = oj, oi
                    l, t, d, r = nl, nt, nd, nr
                    cur = prop
                    if cur < best_score:
                        best, best_score = m.copy(), cur
        owners[g * gs : (g + 1) * gs] = best
    return owners


def _train_task(key):
    name, li = key
    c = _CTX
    t = time.time()
    rng = np.random.default_rng(zlib.crc32(f"{name}:{c['layers'][li]}".encode()))
    return name, li, optimize(c["train"][name][li], rng, c["spec"], c["steps"], c["constraint"]), time.time() - t


def _eval_task(li):
    """All plans x all test sets for one layer -> {(plan, test): sums}, {(plan, test): per-chunk peaks}."""
    c = _CTX
    spec = c["spec"]
    rows, groups, gs = spec.rows, spec.groups, spec.group_size
    res, frames = {}, {}
    for tname, cnt in c["test"].items():
        cl = cnt[li].astype(np.int64)
        per_group = []
        for g in range(groups):
            cg = cl[:, :, g * gs : (g + 1) * gs]
            totals = cg.sum(1)
            tiles = (totals + 31) // 32
            d, cc = contributions(cg, spec.transport, spec.num_links)
            per_group.append((totals, tiles, d, cc))
        id_pk = None
        for pname, plan in c["plans"].items():
            pk = mt = hp = None
            for g, (totals, tiles, d, cc) in enumerate(per_group):
                _, t, dd, ccc = _state(totals, tiles, d, cc, plan[li][g * gs : (g + 1) * gs], rows, gs)
                pk = t.max(1) if pk is None else np.maximum(pk, t.max(1))
                mt = t.mean(1) / groups if mt is None else mt + t.mean(1) / groups
                hp = (dd + ccc).sum(1) if hp is None else hp + (dd + ccc).sum(1)
            if pname == "identity":
                id_pk = pk
            ratio = pk / np.maximum(id_pk, 1)
            frames[pname, tname] = pk.astype(np.float64)
            res[pname, tname] = np.array(
                [
                    pk.sum(),
                    mt.sum(),
                    hp.sum(),
                    (pk > id_pk).sum(),
                    len(pk),
                    ratio.max(),
                    np.maximum(pk - id_pk, 0).sum(),
                ],
                np.float64,
            )
    return res, frames


def train_sets(corpus, spec: MoeSpec, plans) -> tuple[dict, dict]:
    train = [s for s in corpus.samples if s["split"] == "train"]
    domains = [d for d in DOMAINS if d in corpus.domains] + [d for d in corpus.domains if d not in DOMAINS]
    out, comp = {}, {}

    def add(name, pool, n, seed):
        if plans != FULL_PLANS and name not in plans:
            return
        out[name], comp[name] = traffic.build_frames(
            corpus, pool, n, seed, spec.rows, spec.num_experts, spec.chunk_tokens
        )

    for n in (32, 64, 128, 256):
        add(f"bal{n}_s1", train, n, 101)  # same seed: smaller sets are prefixes of the same stream
    add("bal128_s2", train, 128, 202)
    for d in domains:
        i = DOMAINS.index(d) if d in DOMAINS else len(DOMAINS) + domains.index(d)
        add(f"lodo_{d}", [s for s in train if s["domain"] != d], 128, 300 + i)
        add(f"indom_{d}", [s for s in train if s["domain"] == d], 64, 400 + i)
    for g in ("cjk", "indic_thai_arabic"):
        if any(s["group"] == g for s in train):
            add(f"lolo_{g}", [s for s in train if s["group"] != g], 128, 500 + len(g))
    return out, comp


def test_sets(corpus, spec: MoeSpec) -> tuple[dict, dict]:
    test = [s for s in corpus.samples if s["split"] == "test"]
    domains = [d for d in DOMAINS if d in corpus.domains] + [d for d in corpus.domains if d not in DOMAINS]
    out, comp = {}, {}
    ct = spec.chunk_tokens

    def tok_of(pool):
        return sum(s["end"] - s["start"] for s in pool)

    def add(name, pool, n, seed, strata="domain"):
        out[name], comp[name] = traffic.build_frames(corpus, pool, n, seed, spec.rows, spec.num_experts, ct, strata)

    add("mix_balanced", test, 48, 9001)
    add("mix_source_uniform", test, 48, 9002, strata="source")
    for d in domains:
        pool = [s for s in test if s["domain"] == d]
        i = DOMAINS.index(d) if d in DOMAINS else len(DOMAINS) + domains.index(d)
        add(f"domain_{d}", pool, max(4, min(16, tok_of(pool) // ct)), 9100 + i)
    for g in ("english", "european", "cjk", "indic_thai_arabic", "african"):
        pool = [s for s in test if s["group"] == g]
        if tok_of(pool) >= 2 * ct:
            add(f"lang_{g}", pool, min(16, tok_of(pool) // ct), 9200 + len(g))
    for n in sorted(corpus.ids):
        if traffic.is_special(n):
            out[f"seq_{n}"] = traffic.contiguous_frames(corpus.ids[n], spec.rows, spec.num_experts, ct)
            comp[f"seq_{n}"] = dict(chunks=out[f"seq_{n}"].shape[1], contiguous=True)
    return out, comp


def run(
    workdir: Path,
    spec: MoeSpec,
    out: str = "calibration",
    steps: int | None = None,
    constraint: str = "aggregate",
    plans=FULL_PLANS,
    extra_plans: dict | None = None,
    eval_plans: Path | None = None,
    layers=None,
    skip_prefix: int = 0,
    workers: int = 16,
    log=print,
) -> dict:
    """Train and evaluate. ``extra_plans``: name -> per-layer owner arrays (e.g. a committed preset) scored
    alongside. ``eval_plans``: skip training, score the plans stored in that plans_owners.npz instead."""
    t0 = time.time()
    steps = steps or round(
        2000 * spec.group_size / 64
    )  # 2000 for 64 experts per group (GLM-5.3), 7000 for 224 (Kimi-K3)
    od = workdir / out
    od.mkdir(parents=True, exist_ok=True)
    corpus = traffic.load_corpus(workdir, layers, skip_prefix)
    L = len(corpus.layers)
    log(f"{len(corpus.ids)} captures, {len(corpus.samples)} requests, {L} layers, load {time.time() - t0:.0f}s")
    train, comp_train = ({}, {}) if eval_plans else train_sets(corpus, spec, plans)
    test, comp_test = test_sets(corpus, spec)
    split_tokens = {k: sum(s["end"] - s["start"] for s in corpus.samples if s["split"] == k) for k in ("train", "test")}
    (od / "composition.json").write_text(
        json.dumps(dict(split_tokens=split_tokens, sets={**comp_train, **comp_test}), indent=1)
    )
    corpus.ids = None  # only counts are needed from here on
    log(f"frames built {time.time() - t0:.0f}s: {len(train)} train sets, {len(test)} test sets")

    all_plans = {"identity": [spec.identity_owners()] * L}
    for name, per_layer in (extra_plans or {}).items():
        all_plans[name] = list(per_layer)
    if eval_plans:
        z = np.load(eval_plans)
        if z["layers"].tolist() != corpus.layers:
            raise ValueError("plans were calibrated for other layers")
        all_plans.update(
            {k: list(z[k]) for k in z.files if k not in ("layers", "identity") and (plans == FULL_PLANS or k in plans)}
        )
    _CTX.update(
        spec=spec, layers=corpus.layers, train=train, test=test, steps=steps, constraint=constraint, plans=all_plans
    )
    secs = defaultdict(float)
    if train:
        keys = sorted(((n, li) for n in train for li in range(L)), key=lambda k: -train[k[0]].shape[1])
        trained = defaultdict(dict)
        with get_context("fork").Pool(workers) as pool:
            for i, (name, li, o, s) in enumerate(pool.imap_unordered(_train_task, keys)):
                trained[name][li] = o
                secs[name] += s
                if (i + 1) % max(1, len(keys) // 10) == 0:
                    log(f"trained {i + 1}/{len(keys)} ({time.time() - t0:.0f}s)")
        for name in train:
            all_plans[name] = [trained[name][li] for li in range(L)]
        np.savez_compressed(
            od / "plans_owners.npz", layers=np.array(corpus.layers), **{k: np.stack(v) for k, v in all_plans.items()}
        )
        (od / "train_seconds.json").write_text(json.dumps(secs, indent=1))

    tot = defaultdict(lambda: np.zeros(7))
    chunk_pk = defaultdict(float)
    with get_context("fork").Pool(workers) as pool:
        for res, frames in pool.imap_unordered(_eval_task, range(L)):
            for k, v in frames.items():
                chunk_pk[k] = chunk_pk[k] + v
            for k, v in res.items():
                tot[k][[0, 1, 2, 3, 4, 6]] += v[[0, 1, 2, 3, 4, 6]]
                tot[k][5] = max(tot[k][5], v[5])
    rows = []
    for tname in test:
        base = tot["identity", tname]
        for pname in all_plans:
            pk, mt, hp, worse, frames, worst, excess = tot[pname, tname]
            rows.append(
                dict(
                    test=tname,
                    plan=pname,
                    peak_change_pct=round(100 * (pk / base[0] - 1), 3),
                    utilization_identity=round(base[1] / base[0], 4),
                    utilization_plan=round(mt / pk, 4),
                    hops_change_pct=round(100 * (hp / base[2] - 1), 3),
                    layer_chunks_worse_pct=round(100 * worse / frames, 2),
                    worst_layer_chunk_ratio=round(worst, 3),
                    regression_tiles_pct_of_identity=round(100 * excess / base[0], 3),
                )
            )
    suffix = "_eval" if eval_plans else ""
    _write_csv(od / f"results{suffix}.csv", rows)
    crow = []
    for tname in test:
        base = chunk_pk["identity", tname]
        for pname in all_plans:
            ch = 100 * (chunk_pk[pname, tname] / base - 1)
            crow.append(
                dict(
                    test=tname,
                    plan=pname,
                    chunks=len(ch),
                    chunks_worse_pct=round(100 * float((ch > 0).mean()), 2),
                    worst_chunk_change_pct=round(float(ch.max()), 3),
                    p90_chunk_change_pct=round(float(np.percentile(ch, 90)), 3),
                    median_chunk_change_pct=round(float(np.median(ch)), 3),
                    best_chunk_change_pct=round(float(ch.min()), 3),
                )
            )
    _write_csv(od / f"chunks{suffix}.csv", crow)
    summary = dict(
        layers=corpus.layers,
        steps=steps,
        constraint=constraint,
        skip_prefix=skip_prefix,
        seconds=round(time.time() - t0, 1),
        plans=list(all_plans),
        tests=list(test),
        peak_change_pct={t: {r["plan"]: r["peak_change_pct"] for r in rows if r["test"] == t} for t in test},
        utilization_identity={t: next(r["utilization_identity"] for r in rows if r["test"] == t) for t in test},
    )
    (od / f"summary{suffix}.json").write_text(json.dumps(summary, indent=1))
    log(pivot(rows, list(all_plans), list(test)))
    _CTX.clear()
    return summary


def _write_csv(path: Path, rows: list) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def pivot(rows, plans, tests) -> str:
    lines = ["peak_change_pct".ljust(28) + "".join(f"{p[:14]:>15s}" for p in plans)]
    for t in tests:
        r = {x["plan"]: x["peak_change_pct"] for x in rows if x["test"] == t}
        lines.append(f"{t:28s}" + "".join(f"{r[p]:15.2f}" for p in plans))
    return "\n".join(lines)


def load_plan(workdir: Path, out: str, name: str) -> tuple[list, np.ndarray]:
    z = np.load(workdir / out / "plans_owners.npz")
    if name not in z.files:
        raise KeyError(f"plan {name!r} not in {out}/plans_owners.npz (have {[k for k in z.files if k != 'layers']})")
    return z["layers"].tolist(), z[name]
