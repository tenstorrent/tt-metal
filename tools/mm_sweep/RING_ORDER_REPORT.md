# Physical-topology-aware in0 ring ordering — report

The in0 all-gather ring no longer visits the 8 bank-cores in bank-index order `0→1→…→7`. The factory picks,
**per ring group**, a cyclic visiting order that minimizes the physical NoC route cost of the forward edges,
using the group's **writer NoC** and the authoritative directed-torus hop distance
(`tt::tt_metal::experimental::Device::get_worker_noc_hop_distance` — logical→physical + NOC0 `+x→+y` / NOC1
`−x→−y` routing with wraparound, harvesting-aware). Placement, work partitioning, compute, and reduction are
unchanged; only `ring_pos`/`ring_next_idx`/`ring_prev_idx` change (which core seeds which in0 shard, the
forward route, the in1 rotated read) — the output is **bit-identical for any permutation**, so ring order is
purely a performance lever. No kernel change (host-side factory override).

**Production default = PARETO objective** (see §C). Selected by a two-run objective A/B against the
acceptance gate — **not** by user approval. It is a **single global objective** (same for all `Sm`), not an
`Sm`-conditional policy.

## Correction to prior documentation
- Production originally hardcoded `in.nn_chain = false` → **bank-index order** (not nearest-neighbour as
  earlier notes claimed); the planner's `nn_chain` used *logical* Manhattan distance (a proxy), not physical
  routing. This work replaced both with physical-NoC-aware ordering.
- An earlier draft said the `Sm>1` aggregate objective was "user-approved" and shipped. That is superseded:
  that objective (now called **maxedge**) is **rejected** here (it stably regressed the synthetic Sm=4 case),
  and **pareto** is shipped instead, chosen by the gate below.

## Cost model and its limits
Per forward edge `posₚ→posₚ₊₁` the cost is the directed writer-NoC hop distance `d[a][b]`. For one 8-core
ring the cost is `(max_edge, total_hops)`. **Aggregation levels, kept distinct:**
- **per-ring** — one mm-ring's `(max_edge, total_hops)`.
- **group-aggregate** (Sm>1, one `(kk,nn)` group's Sm mm-rings): `aggmax` = max over rings of each ring's
  max_edge; `aggtot` = sum over rings of total_hops; `maxringtot` = max over rings of total_hops.
- **op-aggregate** — across all `(kk,nn)` groups: max of group `aggmax`, sum of group `aggtot`.

**Caveat (no congestion model):** `get_worker_noc_hop_distance` returns a scalar *distance*, not the link
path an edge traverses, so **shared-link congestion is not modeled**. `total_hops` is only a proxy for
aggregate link pressure; `max_edge` is one objective term. A permutation with lower `max_edge` but higher
`total_hops` can be slower in practice — exactly what sank the **maxedge** objective on Sm=4 (§C).

## M-split correctness constraint (Sm>1)
The in1 *slaves* consume in1 in the `mm==0` *reader's* shard order while their in0 rings are separate
physical cores, so all `Sm` slices of a `(kk,nn)` group MUST use the SAME permutation (reader/slave
`ring_pos` must agree per bank) or the in0/in1 pairing corrupts (caught as PCC 0.65 during development). The
chosen order is applied to every slice in the group; the objective therefore scores across all `Sm` rings.

## A. Sm=1: optimized vs bank order (the original ring-order win)
Median device-profiler kernel µs, 3 interleaved relaunches, Δ vs bank. For `Sm=1` all exhaustive objectives
that are max-first (mm0/maxedge/pareto) pick the **same** order, so this is the shared Sm=1 result. Raw:
`regime_a_ringorder_bench_bankopt.json` — **one** preserved raw run (recovered from the prior ring-order
commit). A second confirming run was measured but **not preserved** (prose only in the prior report); the
numbers below are the one committed raw run.
| shape | cfg (Ns,Pk,Sm,kb,nsb) | bank µs | opt Δ |
|---|---|---|---|
| 256×6144×768 | 1,12,1,2,1 | 51.6 | **−8.4%** |
| 256×6144×2304 | 1,12,1,2,1 | 90.5 | −4.8% |
| 256×6144×4608 | 1,12,1,2,1 | 151.2 | −3.4% |
| 128×6144×4608 (Mt4) | 1,12,1,2,1 | 128.5 | −2.5% |
| 64×6144×4608 (Mt2) | 1,6,1,4,2 | 118.6 | −0.1% |
| 32×6144×4608 (Mt1) | 1,12,1,2,1 | 116.8 | −0.0% |

Op-aggregate route cost: opt roughly **halves** max_edge (e.g. 768: 25→14) and total_hops (1159→683). Win
scales with M (largest on deep-Pk Mt=8; neutral at Mt≤2 where the ring is hidden behind compute). Per-RISC:
all three RISC spans drop ~4–5µs in lockstep on the winners (shorter forward routes cut the ring-gather
latency gating reader/writer/compute alike).

## B. Sm>1 shared-permutation objective — the problem
The permutation is shared across a group's `Sm` rings. Scoring only the `mm==0` reader ring (**mm0**) ignores
the slaves' routes; the first aggregate fix (**maxedge** = min aggmax then aggtot) reduced the worst edge but
could *raise* total_hops, which regressed the synthetic Sm=4 case. §C evaluates better objectives.

## C. Objective A/B (this follow-up) — pareto selected
Six shared-permutation objectives computed in one exhaustive 7! pass (all internal cache-hashed diagnostics,
none in the public API); lexicographic minimize:
`mm0`=(ring0.max, ring0.total); `maxedge`=(aggmax, aggtot); `maxring`=(maxringtot, aggmax, aggtot);
`total`=(aggtot, aggmax); **`pareto`**=min aggmax s.t. `aggtot ≤ mm0's aggtot`, then aggtot. (`greedy` =
non-exhaustive nearest-neighbour, reference only.)

Two independent runs, 3 interleaved relaunches each. Raw: `regime_a_ringobj_run1.json`,
`regime_a_ringobj_run2.json` (every relaunch, per-RISC, selected perms, per-ring/group/op route costs). Δ vs
**mm0** (the shared baseline), both runs shown:

| shape | Sm | maxedge | total | **pareto** |
|---|---|---|---|---|
| 256×2048×1024 (production) | 2 | −4.2 / −3.9 | −4.2 / −3.7 | **−5.2 / −4.6** |
| 128×6144×4608 | 2 | −0.6 / −0.2 | −0.6 / −0.5 | −1.0 / −0.5 |
| 256×2048×1024 (synthetic) | 4 | **+2.7 / +2.9** ✗ | −1.8 / −0.3 | +1.0 / −0.2 |
| 256×6144×4608 | 4 | +0.7 / +0.8 | −0.4 / −0.1 | −0.3 / −0.5 |
| 256×2048×1024 | 3 | −3.2 / −4.8 | −3.5 / −4.4 | −3.9 / −4.0 |
| 256×6144×768 | 1 | +1.2 / +0.1 | +1.6 / +0.4 | +0.5 / +0.7 |
| 256×6144×4608 | 1 | +0.3 / −0.0 | +1.0 / +0.7 | +0.1 / −0.2 |
| 256×6144×2304 | 1 | −0.3 / +0.1 | +1.1 / +1.1 | −0.0 / +0.1 |

Reading (noise floor ≈ 1.4%: for `Sm=1` mm0/maxedge/pareto pick byte-identical orders, so their `Sm=1`
deltas are pure inter-subprocess variance):
- **maxedge** (the prior aggregate default): stably **regresses the synthetic Sm=4** case (+2.7/+2.9%) — it
  chose lower `aggmax` but higher `aggtot`; the missing congestion term. **Rejected.**
- **total** (total-first): fixes Sm=4 but **regresses the common Sm=1 case** (+0.4…+1.6%, consistent across
  runs) — it picks higher-max-edge orders for Sm=1. **Rejected.**
- **pareto** (min max-edge subject to `aggtot ≤ mm0`): keeps the best Sm=2 win (**−4.6…−5.2%**) and Sm=3
  win (−3.9/−4.0%); on Sm=4 its route **strictly dominates mm0** (18:253:70 vs 22:273:109 on the synthetic
  case — lower on all of aggmax/aggtot/maxringtot), so it **cannot stably regress vs mm0** (its ±1% wall is
  noise); and it stays within noise of mm0 on Sm=1 (max-first family). **Selected — single global objective.**

**Gate check (pareto):** preserves the Sm=2 improvement; no stable regression on any tested Sm=1/2/3/4 case
(Sm=4 route-dominated ⇒ noise; Sm=1 byte-identical-family ⇒ noise); correctness + cache-replay preserved.

## Correctness (gtest `RegimeADiagFixture.RingOrderCorrectness`)
bank / greedy / mm0 / maxedge / maxring / total / pareto — random BF16 vs a CPU f32 golden, PCC ≥ 0.999,
fresh AND cached-program, **all BIT-IDENTICAL** across objectives, covering Pk=1 + split-K (both reader NoC
orientations), Ns>1, **Sm=1/2/3/4**, W=1/W>1, balanced K/N tails. Public 20/20 + cache-replay pass on the
pareto default. Watcher clean on the pareto default (Pk=12). (A pre-existing, unrelated `in1_reader` Sm>1
atomic-flush watcher warning fires identically at bank baseline — orthogonal to ring ordering.)

## D. Picker-sensitive corpus sweep (under the pareto default)
The full planner-feasible space is **~13,694 configs** across the requested shapes — intractable to
device-run exhaustively. Used a **principled bounded search** (`ring_corpus_sweep.py`): per shape the
candidate set = {prior broad-sweep configs within 1.5× of that shape's best old time, capped 25} ∪ {current
picker config} ∪ {±1-step neighbours of the picker config, planner-feasible}. This re-evaluates the prior
sweep's broad space (ranks 11–25, beyond the top-10) **and** configs the prior sweep never measured (the
neighbourhood). Each candidate run once under the pareto default; any that beat the picker were re-run 3×.
Raw: `regime_a_ring_corpus_sweep.json`. Shapes: both Mt=8 primaries + all Mt≥4 FLUX/LTX + the Pk>1/Sm>1
six-shape-parity cases (12 shapes).

Result: **11 of 12 unchanged.** One stable, confirmed change:
- **128×15360×768** (Mt=4): picker `(Ns1,Pk12,Sm1,kb1,nsb3)` 71.0µs → **`(Ns1,Pk6,Sm1,kb2,nsb3)` 66.4µs,
  +6.5%** (3-relaunch confirmed, PCC 0.99999 fresh+cached). **Picker updated** (C++ `auto_select_config`
  table + `picker_table.py`). This shape is `Sm=1`, so the gain is orthogonal to the objective work — a
  config the picker had suboptimal, surfaced by the mandated sweep under the current pipelined-drain +
  pareto-ring stack. 128×6144×2304 showed a single-pass candidate but within noise (unconfirmed) → no change.

**Caveat:** an unchanged picker output does **not** prove the selected config is optimal — this is a bounded
search, not the full feasible space. The 128×15360×768 change triggered re-validation: public 20/20 pass,
six-shape parity unchanged (that shape is not a parity case; op stays faster than the frozen oracle on all
six), and the shape's own correctness+perf validated directly.

## Evidence & retention
- **Objective A/B:** both raw runs committed (`regime_a_ringobj_run1.json`, `regime_a_ringobj_run2.json`) —
  every relaunch, per-RISC, per-group RINGCOST.
- **Sm=1 bank/opt:** one raw run committed (`regime_a_ringorder_bench_bankopt.json`); the second confirming
  run is measured-but-unpreserved (prose only) — stated as such, not claimed as committed raw evidence.
- **Corpus sweep:** `regime_a_ring_corpus_sweep.json`.
- The RINGCOST diagnostic is gated behind `TT_MM_RINGCOST`; production compiles are silent. The exhaustive
  7!-per-group search is a one-time host cost at program compile (<10 ms).

## Objective code retained vs removed
`pareto` (default), `mm0`, `maxedge`, and `total` are retained as internal cache-hashed A/B diagnostics —
each reproduces a decision-relevant point (mm0 = pareto's reference / prior objective; maxedge = the
Sm4-regressing objective; total = the Sm1-regressing alternative), plus `bank` (pre-ring baseline). The
dominated `greedy` (non-exhaustive heuristic) and `maxring` (dominated by pareto/total, no distinct evidence)
are **removed**; recover from the implementation commit for this task.
