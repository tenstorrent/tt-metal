#!/usr/bin/env python3
"""Theory-guided configuration candidates for ``regime_a_matmul`` sweeps.

The tuple order used here is the sweep convention::

    (Ns, Pk, Sm, kb, nsb)

This module deliberately does *not* import or consult the production picker.  It
uses the planner constraints, the dataflow geometry, and several competing cost
models.  The competing models are important: historical sweeps contain pure-K,
KxN, and KxM winners, so pruning with one fitted scalar cost simply bakes in the
current picker's blind spots.

The generator reduces the search in four steps:

1. Enumerate all feasible work factorizations (Pk, Ns, Sm).
2. Keep kb/nsb values at padding, W, N_bpc, transaction, and L1 boundaries.
3. Rank the resulting configurations with a small ensemble of physical models
   (DRAM, compute, latency, reduction, and M-forward sensitive).
4. Preserve factorization/core-count diversity and add deterministic samples
   from the configurations that were pruned, so the pruning can be audited.

Typical use:

    python tools/mm_sweep/regime_a_candidate_generator.py 256 2048 1024
    python tools/mm_sweep/regime_a_candidate_generator.py 256 2048 1024 \
        --budget 96 --audit 8 --json candidates.json
    python tools/mm_sweep/regime_a_candidate_generator.py --backtest \
        'tools/mm_sweep/regime_a_sweep_*.json*'

The JSON output contains both selected candidates and their geometry/features,
so a sweep harness can consume ``selected[*].cfg`` without duplicating logic.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import gzip
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


TILE = 32
TILE_BYTES_BF16 = 2048
TILE_BYTES_FP32 = 4096
L1_BUDGET_BYTES = 1440 * 1024
MAX_WORKERS = 104
DRAM_SATURATION_READERS = 24

Config = tuple[int, int, int, int, int]  # Ns, Pk, Sm, kb, nsb


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def rup(a: int, b: int) -> int:
    return cdiv(a, b) * b


@dataclass(frozen=True)
class Geometry:
    cfg: Config
    cores: int
    readers: int
    kt_slice: int
    m_block: int
    n_owned: int
    n_bpc: int
    n_slice: int
    w: int
    l1_bytes: int
    k_padding: float
    m_padding: float
    n_padding: float
    scheduled_over_logical: float
    read_work: float
    compute_work: float
    compute_work_adjusted: float
    first_block_work: float
    startups: float
    reduction_work: float
    m_forward_work: float
    in0_ring_work: float

    @property
    def factorization(self) -> tuple[int, int, int]:
        ns, pk, sm, _kb, _nsb = self.cfg
        return pk, ns, sm

    @property
    def factor_class(self) -> str:
        ns, _pk, sm, _kb, _nsb = self.cfg
        if ns == 1 and sm == 1:
            return "pure_k"
        if ns == 1:
            return "kxm"
        if sm == 1:
            return "kxn"
        return "mixed"


def geometry(M: int, K: int, N: int, cfg: Config) -> Geometry | None:
    """Mirror the production planner's feasibility and derive scheduling work."""
    Ns, Pk, Sm, kb, nsb = cfg
    Mt, Kt, Nt = cdiv(M, TILE), cdiv(K, TILE), cdiv(N, TILE)
    if min(Ns, Pk, Sm, kb, nsb) < 1 or Sm > Mt or Pk > Kt:
        return None

    n_band = cdiv(Nt, 8)
    smallest_bank = Nt - 7 * n_band
    if smallest_bank < 1 or smallest_bank < Ns:
        return None
    cores = 8 * Pk * Ns * Sm
    if cores > MAX_WORKERS:
        return None

    kt_slice = rup(cdiv(Kt, Pk), kb * 8)
    m_block = cdiv(Mt, Sm)
    n_owned = cdiv(n_band, Ns)
    if nsb > n_owned:
        return None
    n_bpc = cdiv(n_owned, nsb)
    n_slice = n_bpc * nsb

    cb0 = kt_slice * m_block
    cb1 = 4 * kb * nsb
    cb2 = 2 * m_block * nsb
    cb3 = m_block * nsb
    cb7 = 2 * m_block * nsb if Pk > 1 else 0
    l1_bytes = (cb0 + cb1 + cb2 + cb7) * TILE_BYTES_BF16 + cb3 * TILE_BYTES_FP32
    if l1_bytes > L1_BUDGET_BYTES:
        return None

    k_sched = kt_slice * Pk
    m_sched = m_block * Sm
    n_sched = n_slice * Ns * 8
    k_padding = k_sched / Kt
    m_padding = m_sched / Mt
    n_padding = n_sched / Nt

    # Reader count excludes Sm slaves: each mm=0 reader reads once and forwards
    # to the other M slices.  DRAM stops scaling at roughly 24 readers on BH.
    readers = 8 * Pk * Ns
    read_work = (k_sched * n_sched) / min(readers, DRAM_SATURATION_READERS)

    compute_work = kt_slice * m_block * n_slice
    # Historical sweeps show kb saturating near 2 and useful matmul sub-block
    # area saturating near six tiles.  This is used only for broad ranking.
    area = min(m_block * nsb, 6)
    compute_eff = (min(kb, 2) / (min(kb, 2) + 0.5)) * (area / (area + 2.0))
    compute_work_adjusted = compute_work / max(compute_eff, 1e-6)

    # Time until the first useful matmul block is available.  Small nsb is
    # disproportionately valuable on the progressive pipeline: prior deep-K
    # sweeps repeatedly chose nsb=1 even when a throughput-only model preferred
    # a wider block.
    first_block_work = kb * nsb * (m_block + 1)
    startups = (kt_slice / kb) * n_bpc
    reduction_work = max(Pk - 1, 0) * m_block * n_slice
    m_forward_work = max(Sm - 1, 0) * kt_slice * n_slice
    in0_ring_work = (7.0 / 8.0) * m_block * kt_slice

    return Geometry(
        cfg=cfg,
        cores=cores,
        readers=readers,
        kt_slice=kt_slice,
        m_block=m_block,
        n_owned=n_owned,
        n_bpc=n_bpc,
        n_slice=n_slice,
        w=kt_slice // (8 * kb),
        l1_bytes=l1_bytes,
        k_padding=k_padding,
        m_padding=m_padding,
        n_padding=n_padding,
        scheduled_over_logical=k_padding * m_padding * n_padding,
        read_work=read_work,
        compute_work=compute_work,
        compute_work_adjusted=compute_work_adjusted,
        first_block_work=first_block_work,
        startups=startups,
        reduction_work=reduction_work,
        m_forward_work=m_forward_work,
        in0_ring_work=in0_ring_work,
    )


def _kb_values(Kt: int, Pk: int) -> list[int]:
    """kb values at useful compute/W/padding boundaries.

    kb>8 has never been a winner in the retained Regime-A sweeps.  It is still
    considered when a deep K slice can use it without padding; otherwise those
    values are available to the audit sampler rather than the primary search.
    """
    logical_slice = cdiv(Kt, Pk)
    vals = {1, 2, 3, 4, 6, 8}
    # Values that put W close to a small integer are latency/overlap boundaries.
    for target_w in (1, 2, 3, 4, 6, 8):
        x = logical_slice / (8 * target_w)
        for kb in (math.floor(x), math.ceil(x)):
            if 1 <= kb <= 16:
                vals.add(kb)
    if logical_slice >= 16 * 8:
        vals.add(16)
    return sorted(v for v in vals if v >= 1)


def _nsb_values(n_owned: int, m_block: int) -> list[int]:
    """nsb values where N_bpc, padding, CB size, or compute area matters."""
    vals = {1, 2, 3, 4, 6, 8, 12, 16, n_owned}

    # Exact/near divisors and transitions for the useful N_bpc range.
    for target_bpc in (1, 2, 3, 4, 5, 6, 8, 12, 16):
        x = n_owned / target_bpc
        for v in (math.floor(x), math.ceil(x)):
            vals.update((v - 1, v, v + 1))

    # Compute sub-block area boundaries.  These matter most at Mt<=8.
    for target_area in (2, 4, 6, 8, 12, 16, 24, 32):
        x = target_area / max(m_block, 1)
        vals.update((math.floor(x), math.ceil(x)))

    # All small nsb values are cheap to retain and historically productive.
    vals.update(range(1, min(n_owned, 8) + 1))
    return sorted(v for v in vals if 1 <= v <= n_owned)


def _all_feasible(
    M: int, K: int, N: int, *, kb_max: int = 32
) -> Iterable[Geometry]:
    """Full planner-feasible domain, used only for deterministic audit samples."""
    Mt, Kt, Nt = cdiv(M, TILE), cdiv(K, TILE), cdiv(N, TILE)
    n_band = cdiv(Nt, 8)
    for Pk in range(1, min(Kt, MAX_WORKERS // 8) + 1):
        for Ns in range(1, min(n_band, MAX_WORKERS // (8 * Pk)) + 1):
            n_owned = cdiv(n_band, Ns)
            for Sm in range(1, min(Mt, MAX_WORKERS // (8 * Pk * Ns)) + 1):
                for kb in range(1, kb_max + 1):
                    for nsb in range(1, n_owned + 1):
                        g = geometry(M, K, N, (Ns, Pk, Sm, kb, nsb))
                        if g is not None:
                            yield g


def structured_domain(M: int, K: int, N: int) -> list[Geometry]:
    """Feasible candidates at physically meaningful kb/nsb boundaries."""
    Mt, Kt, Nt = cdiv(M, TILE), cdiv(K, TILE), cdiv(N, TILE)
    n_band = cdiv(Nt, 8)
    out: dict[Config, Geometry] = {}
    for Pk in range(1, min(Kt, MAX_WORKERS // 8) + 1):
        for Ns in range(1, min(n_band, MAX_WORKERS // (8 * Pk)) + 1):
            # The smallest physical bank must provide one tile to every Ns.
            if Nt - 7 * n_band < Ns:
                continue
            n_owned = cdiv(n_band, Ns)
            for Sm in range(1, min(Mt, MAX_WORKERS // (8 * Pk * Ns)) + 1):
                m_block = cdiv(Mt, Sm)
                for kb in _kb_values(Kt, Pk):
                    for nsb in _nsb_values(n_owned, m_block):
                        g = geometry(M, K, N, (Ns, Pk, Sm, kb, nsb))
                        if g is None:
                            continue
                        # Very padded configs are almost always dominated.  Keep
                        # them out of the primary domain; the audit sample still
                        # draws from the complete feasible space.
                        if max(g.k_padding, g.n_padding) > 1.35:
                            continue
                        out[g.cfg] = g
    return list(out.values())


# Competing physical hypotheses.  These are deliberately broad and are not
# trained coefficients.  Each model gets its own top candidates.
MODELS = {
    "balanced": dict(block=0.05, start=0.05, reduce=0.35, mfwd=0.02, ring=0.05, pad=0.40),
    "dram": dict(block=0.02, start=0.01, reduce=0.05, mfwd=0.01, ring=0.01, pad=0.80),
    "compute": dict(block=0.02, start=0.02, reduce=0.10, mfwd=0.01, ring=0.02, pad=0.20),
    "latency": dict(block=0.50, start=0.30, reduce=0.25, mfwd=0.03, ring=0.15, pad=0.25),
    "streaming": dict(block=5.00, start=0.00, reduce=0.20, mfwd=0.02, ring=0.05, pad=0.30),
    "reduction": dict(block=0.05, start=0.05, reduce=1.00, mfwd=0.02, ring=0.05, pad=0.30),
    "m_forward": dict(block=0.05, start=0.05, reduce=0.20, mfwd=0.12, ring=0.05, pad=0.30),
}


def model_cost(g: Geometry, model: str) -> float:
    p = MODELS[model]
    # Read and compute can overlap; the remaining terms approximate exposed
    # fill/drain and synchronization costs.
    base = max(g.read_work, g.compute_work_adjusted)
    extra = (
        p["block"] * g.first_block_work
        + p["start"] * g.startups
        + p["reduce"] * g.reduction_work
        + p["mfwd"] * g.m_forward_work
        + p["ring"] * g.in0_ring_work
    )
    return (base + extra) * (1.0 + p["pad"] * (g.scheduled_over_logical - 1.0))


def _stable_seed(M: int, K: int, N: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{M},{K},{N}".encode()).digest()[:8], "little")


def select_candidates(
    M: int,
    K: int,
    N: int,
    *,
    budget: int = 96,
    audit: int = 8,
    include: Sequence[Config] = (),
) -> tuple[list[Geometry], dict[Config, list[str]], dict[str, int]]:
    """Return a bounded, diverse candidate set and selection provenance."""
    if budget < 1:
        raise ValueError("budget must be positive")
    if audit < 0 or audit > budget:
        raise ValueError("audit must be in [0, budget]")
    domain = structured_domain(M, K, N)
    if not domain:
        return [], {}, {"structured": 0, "selected": 0, "audit": 0}

    by_cfg = {g.cfg: g for g in domain}
    reasons: dict[Config, list[str]] = {}
    selected: dict[Config, Geometry] = {}

    def add(g: Geometry | None, why: str) -> None:
        if g is None:
            return
        selected[g.cfg] = g
        reasons.setdefault(g.cfg, []).append(why)

    for c in include:
        add(geometry(M, K, N, tuple(c)), "explicit")

    # Model ensemble.  More than one scalarization is necessary to retain
    # candidates from physically different performance regimes.
    per_model = max(8, min(24, budget // max(len(MODELS), 1)))
    for name in MODELS:
        for g in sorted(domain, key=lambda x: model_cost(x, name))[:per_model]:
            add(g, f"model:{name}")

    # Best representative of each factorization.  If there are more than fit,
    # balanced cost decides which factors enter first.
    fact_best: dict[tuple[int, int, int], Geometry] = {}
    by_fact: dict[tuple[int, int, int], list[Geometry]] = {}
    for g in sorted(domain, key=lambda x: model_cost(x, "balanced")):
        fact_best.setdefault(g.factorization, g)
        by_fact.setdefault(g.factorization, []).append(g)
    for g in sorted(fact_best.values(), key=lambda x: model_cost(x, "balanced")):
        add(g, "factorization")

    # Do not let an imperfect analytical model choose kb/nsb too narrowly
    # inside an otherwise promising factorization.  The retained sweeps contain
    # winners at small kb/nsb.  Reserve five high-value corners for the sixteen
    # best factorizations; broader nsb values still enter through the ensemble.
    # This is at most 80 configs, versus hundreds of arbitrary nsb values per
    # factorization.
    fact_order = sorted(
        by_fact,
        key=lambda f: min(min(model_cost(g, m) for m in MODELS) for g in by_fact[f]),
    )
    for f in fact_order[:16]:
        for g in by_fact[f]:
            _ns, _pk, _sm, kb, nsb = g.cfg
            if (kb, nsb) in {(1, 1), (2, 1), (2, 2), (4, 1), (4, 2)}:
                add(g, "factorization-local-grid")

    # Preserve factor-class and core-count alternatives; these exposed several
    # past picker misses (especially KxM versus deep split-K).
    buckets: dict[tuple[str, int], Geometry] = {}
    for g in sorted(domain, key=lambda x: model_cost(x, "balanced")):
        buckets.setdefault((g.factor_class, g.cores), g)
    for g in sorted(buckets.values(), key=lambda x: model_cost(x, "balanced")):
        add(g, "class+cores")

    # Enforce the requested budget on the structured candidates while keeping
    # explicit configs first and rewarding candidates supported by many models.
    def priority(g: Geometry) -> tuple:
        rs = reasons.get(g.cfg, [])
        return (
            0 if "explicit" in rs else 1,
            0 if "factorization-local-grid" in rs else 1,
            -len(rs),
            min(model_cost(g, m) for m in MODELS),
            model_cost(g, "balanced"),
            g.cfg,
        )

    keep_n = max(0, budget - audit)
    kept = sorted(selected.values(), key=priority)[:keep_n]
    selected = {g.cfg: g for g in kept}
    reasons = {c: reasons[c] for c in selected}

    # The ensemble/factorization union can be smaller than the requested
    # budget.  Fill the remainder by best rank across *any* model, rather than
    # wasting available measurements or adding arbitrary raw configs.
    model_rank: dict[Config, int] = {}
    for name in MODELS:
        for rank, g in enumerate(sorted(domain, key=lambda x: model_cost(x, name))):
            model_rank[g.cfg] = min(model_rank.get(g.cfg, rank), rank)
    fill = sorted(
        (g for g in domain if g.cfg not in selected),
        key=lambda g: (model_rank[g.cfg], model_cost(g, "balanced"), g.cfg),
    )
    for g in fill:
        if len(selected) >= keep_n:
            break
        selected[g.cfg] = g
        reasons[g.cfg] = ["ranked-fill"]

    # Deterministic audit of the *full* feasible space.  Prefer configs absent
    # from the structured domain so it tests the pruning assumptions rather
    # than merely duplicating a selected point.
    if audit:
        pool_all = [
            g
            for g in _all_feasible(M, K, N)
            if g.cfg not in selected and g.cfg not in by_cfg
        ]
        rng = random.Random(_stable_seed(M, K, N))
        # Audit mostly plausible omissions (moderate padding), with at most one
        # deliberately wild point.  Randomly timing eight 4-6x padded configs
        # teaches us nothing about the pruning boundary.
        plausible = [g for g in pool_all if g.scheduled_over_logical <= 1.6]
        wild = [g for g in pool_all if g.scheduled_over_logical > 1.6]
        picked_audit = rng.sample(plausible, min(audit, len(plausible)))
        if len(picked_audit) < audit and wild:
            picked_audit += rng.sample(wild, min(audit - len(picked_audit), len(wild)))
        for g in picked_audit:
            selected[g.cfg] = g
            reasons[g.cfg] = ["audit:pruned"]

    final = sorted(selected.values(), key=priority)
    stats = {
        "structured": len(domain),
        "factorizations": len(fact_best),
        "selected": len(final),
        "audit": sum("audit:pruned" in reasons[g.cfg] for g in final),
    }
    return final, reasons, stats


def result_document(
    M: int, K: int, N: int, budget: int, audit: int, include: Sequence[Config] = ()
) -> dict:
    selected, reasons, stats = select_candidates(M, K, N, budget=budget, audit=audit, include=include)
    rows = []
    for g in selected:
        d = asdict(g)
        d["cfg"] = list(g.cfg)
        d["factorization"] = list(g.factorization)
        d["factor_class"] = g.factor_class
        d["l1_pct"] = 100.0 * g.l1_bytes / L1_BUDGET_BYTES
        d["model_costs"] = {m: model_cost(g, m) for m in MODELS}
        d["reasons"] = reasons[g.cfg]
        rows.append(d)
    return {
        "shape": {"M": M, "K": K, "N": N, "Mt": cdiv(M, TILE), "Kt": cdiv(K, TILE), "Nt": cdiv(N, TILE)},
        "tuple_order": ["Ns", "Pk", "Sm", "kb", "nsb"],
        "budget": budget,
        "audit_requested": audit,
        "stats": stats,
        "selected": rows,
    }


def _open_json(path: str) -> dict:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def backtest(paths: Sequence[str], budget: int, audit: int) -> int:
    """Report regret against historical exhaustive/large sweeps."""
    rows = []
    for path in paths:
        try:
            d = _open_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        results = [
            r
            for r in d.get("results", [])
            if r.get("us_med") is not None and r.get("cls", "ok") == "ok" and len(r.get("cfg", [])) == 5
        ]
        if not results:
            continue
        M, K, N = int(d["M"]), int(d["K"]), int(d["N"])
        chosen, _why, stats = select_candidates(M, K, N, budget=budget, audit=audit)
        cset = {g.cfg for g in chosen}
        measured = {tuple(r["cfg"]): float(r["us_med"]) for r in results}
        overlap = cset & measured.keys()
        if not overlap:
            continue
        full_cfg = min(measured, key=measured.get)
        kept_cfg = min(overlap, key=measured.get)
        regret = measured[kept_cfg] / measured[full_cfg] - 1.0
        rows.append((regret, path, len(results), len(overlap), full_cfg, kept_cfg, stats))

    if not rows:
        print("No usable historical sweep results found.")
        return 1
    print("shape/file | measured overlap | full best -> kept best | regret")
    for regret, path, n, ov, full_cfg, kept_cfg, _stats in sorted(rows, reverse=True):
        print(
            f"{Path(path).name}: {ov}/{n} | {full_cfg} -> {kept_cfg} | {regret * 100:+.2f}%"
        )
    regs = [r[0] for r in rows]
    print(
        f"\n{len(rows)} shapes: median regret={statistics.median(regs)*100:.2f}% "
        f"worst={max(regs)*100:.2f}%  within1%={sum(r <= .01 for r in regs)}/{len(regs)} "
        f"within2%={sum(r <= .02 for r in regs)}/{len(regs)}"
    )
    return 0


def _parse_cfg(text: str) -> Config:
    xs = tuple(int(x) for x in text.split(","))
    if len(xs) != 5:
        raise argparse.ArgumentTypeError("config must be Ns,Pk,Sm,kb,nsb")
    return xs  # type: ignore[return-value]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("M", type=int, nargs="?")
    ap.add_argument("K", type=int, nargs="?")
    ap.add_argument("N", type=int, nargs="?")
    ap.add_argument("--budget", type=int, default=96)
    ap.add_argument("--audit", type=int, default=8)
    ap.add_argument("--include", type=_parse_cfg, action="append", default=[])
    ap.add_argument("--json", type=str, help="write the complete candidate document")
    ap.add_argument("--backtest", type=str, help="glob of historical sweep JSON/JSON.GZ files")
    args = ap.parse_args()

    if args.backtest:
        return backtest(sorted(glob.glob(args.backtest)), args.budget, args.audit)
    if None in (args.M, args.K, args.N):
        ap.error("M K N are required unless --backtest is used")

    doc = result_document(args.M, args.K, args.N, args.budget, args.audit, args.include)
    if args.json:
        Path(args.json).write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"{args.M}x{args.K}x{args.N}: structured={doc['stats']['structured']} "
        f"factorizations={doc['stats']['factorizations']} selected={doc['stats']['selected']} "
        f"audit={doc['stats']['audit']}"
    )
    for row in doc["selected"]:
        print(
            f"{tuple(row['cfg'])!s:20} cores={row['cores']:3d} W={row['w']:2d} "
            f"Nbpc={row['n_bpc']:2d} pad={row['scheduled_over_logical']:.2f} "
            f"L1={row['l1_pct']:.0f}% {','.join(row['reasons'])}"
        )
    if args.json:
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
