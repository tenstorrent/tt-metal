# Physical-topology-aware in0 ring ordering — report

Change: the in0 all-gather ring no longer visits the 8 bank-cores in bank-index order `0→1→…→7`. The factory
picks, **per ring group**, a cyclic visiting order that minimizes the physical NoC route cost of the forward
edges, using the group's **writer NoC** and the authoritative directed-torus hop distance
(`tt::tt_metal::experimental::Device::get_worker_noc_hop_distance` — logical→physical + NOC0 `+x→+y` / NOC1
`−x→−y` routing with wraparound, harvesting-aware). Placement, work partitioning, compute, and reduction are
unchanged; only `ring_pos`/`ring_next_idx`/`ring_prev_idx` (which core seeds which in0 shard, the forward
route, and the in1 rotated read) change — the output is bit-identical for any permutation.

**Verdict:** exhaustive optimized ordering is the production default. For `Sm=1` it beats bank order by
−4 to −10% on Mt=8 shapes (neutral on small-M). Under M-split (`Sm>1`) the default objective scores each
permutation **across all `Sm` mm-rings** ("agg"), which beats the earlier `mm=0`-only scoring by a further
−3.6 to −5.3% (two runs) on the real 256×2048×1024 Sm=2 primary. Diagnostics: `DIAG_RING_BANK` (`1<<12`, old bank order),
`DIAG_RING_GREEDY` (`1<<13`), `DIAG_RING_OPT_MM0` (`1<<14`, mm=0-only scoring).

## Correction to prior documentation
Earlier notes described the production ring as nearest-neighbour. That was inaccurate: production hardcoded
`in.nn_chain = false` → **bank-index order**, and the planner's `nn_chain` path used *logical* Manhattan
distance (a proxy), not physical routing. This change replaces both with physical-NoC-aware ordering.

## Cost model and its limits
Per forward edge `posₚ→posₚ₊₁` the cost is the directed writer-NoC hop distance `d[a][b]`. For one 8-core
ring the cost is `(max_edge, total_hops)` = (worst single edge, sum of the 8 edges). **Two levels of
aggregation, kept distinct:**
- **per-ring** — one mm-ring's `(max_edge, total_hops)`.
- **group-aggregate** (Sm>1) — across the `Sm` mm-rings of one `(kk,nn)` group: `aggmax` = max over rings of
  each ring's max_edge; `aggtot` = sum over rings of each ring's total_hops.
- **op-aggregate** — across all `(kk,nn)` groups of the op: max of the group `aggmax`, sum of the group
  `aggtot`. This is what the "route cost" tables below report.

**Caveat (no congestion model):** `get_worker_noc_hop_distance` returns a scalar *distance*, not the link
path an edge traverses, so **shared-link congestion is not modeled**. `total_hops` is only a proxy for
aggregate link pressure; `max_edge` is the primary objective (the longest single forward). A permutation with
lower `max_edge` but higher `total_hops` can therefore be slower in practice — observed on the synthetic
Sm=4 case below.

## Ordering modes (constructed per ring group; a ring is homogeneous in NoC)
Each ring = the 8 bank-cores of one `(kk, n-slice, m-block)` slice; all 8 share the slice's NoC, so the group
has a single writer NoC = opposite the reader's (`noc==0`→writer NOC1, `noc==1`→writer NOC0).
- **bank** (`DIAG_RING_BANK`): `[0..7]` — previous production baseline.
- **greedy** (`DIAG_RING_GREEDY`): greedy nearest-neighbour over the writer-NoC hop distance from bank 0.
- **opt-mm0** (`DIAG_RING_OPT_MM0`): exhaustive, scoring only the `mm==0` ring, applied to the whole group.
- **opt-agg** (DEFAULT): exhaustive, scoring the **group-aggregate** `(aggmax, aggtot)` across all `Sm`
  mm-rings — accounts for the slaves' routes, not just the reader's. Identical to opt-mm0 for `Sm=1`.

Exhaustive = the 7! cycles through bank 0 (rotation-invariant); directed edge costs ⇒ a permutation and its
reverse are both evaluated (both orientations). ~5040 cycles/group scored off a precomputed per-mm-ring 8×8
hop matrix — one-time host cost at program compile (<10 ms), negligible vs kernel build.

**M-split correctness constraint:** the in1 *slaves* consume in1 in the `mm==0` *reader's* shard order while
their in0 rings are separate physical cores, so all `Sm` slices of a group MUST use the SAME permutation
(reader/slave `ring_pos` must agree per bank) or the in0/in1 pairing corrupts. The chosen order is applied to
every slice in the group. (Getting this wrong was caught as PCC 0.65 on Sm=2 during development.)

## A. Sm=1: optimized vs bank (the original ring-order win)
Median device-profiler kernel µs, 3 interleaved relaunches, Δ vs bank. Raw:
`regime_a_ringorder_bench_bankopt.json` (two independent runs; per-relaunch + per-RISC + per-group RINGCOST).
For `Sm=1` opt-agg ≡ opt-mm0 (single ring, identical objective and permutation).
| shape | cfg (Ns,Pk,Sm,kb,nsb) | bank µs | greedy Δ | **opt Δ** |
|---|---|---|---|---|
| 256×2048×1024 | 1,4,2,2,2 (Sm2 — see B) | 28.3 | −0.5% | −4.4% |
| 256×6144×768 | 1,12,1,2,1 | 51.6 | −2.3% | **−8.4%** |
| 256×6144×2304 | 1,12,1,2,1 | 90.5 | −2.9% | −4.8% |
| 256×6144×4608 | 1,12,1,2,1 | 151.2 | −1.8% | −3.4% |
| 32×6144×4608 (Mt1) | 1,12,1,2,1 | 116.8 | +0.1% | −0.0% |
| 64×6144×4608 (Mt2) | 1,6,1,4,2 | 118.6 | −0.0% | −0.1% |
| 128×6144×4608 (Mt4) | 1,12,1,2,1 | 128.5 | −1.5% | −2.5% |

Op-aggregate route cost (across ring groups): opt roughly **halves** both metrics vs bank on deep-Pk shapes,
e.g. 256×6144×768 max_edge 25→14, total_hops 1159→683. Opt beats greedy on every shape (greedy is a
heuristic — it can lose to bank on individual groups). Largest win on the shallow-N deep-K primary
256×6144×768 (−8.4%); neutral on small-M (Mt≤2, ring hidden behind compute). Per-RISC: on the winners all
three RISC spans drop ~4–5µs **in lockstep** (768: 43.4→38.7) — shorter forward routes cut the in0
ring-gather latency that gates the reader/writer/compute alike.

## B. Sm>1: aggregate-Sm vs mm=0-only scoring (this follow-up)
Median µs, 3 interleaved relaunches. `agg vs mm0` is the decision (both improve on bank). Raw:
`regime_a_ringorder_bench.json` + `regime_a_ringorder_bench_run1.json` (two independent runs).
| shape | Sm | cfg | bank µs | mm0 µs (Δbank) | agg µs (Δbank) | **agg vs mm0** | route aggmax bank/mm0/agg |
|---|---|---|---|---|---|---|---|
| 256×2048×1024 (primary) | 2 | 1,4,2,2,2 | 28.4 | 27.4 (−3.7%) | **25.9 (−8.8%)** | **−5.3%** | 23 / 26 / 16 |
| 128×6144×4608 | 2 | 1,6,2,2,1 | 217.1 | 214.8 (−1.0%) | 213.5 (−1.6%) | −0.6% | 25 / 22 / 16 |
| 256×2048×1024 (synthetic) | 4 | 1,1,4,2,2 | 73.0 | 71.1 (−2.5%) | 73.3 (+0.3%) | **+3.0%** | 22 / 22 / 16 |
| 256×6144×768 | 1 | 1,12,1,2,1 | 51.2 | 46.9 | 47.5 | +1.4% (noise) | 14 / 14 |
| 256×6144×4608 | 1 | 1,12,1,2,1 | 150.9 | 146.1 | 146.2 | +0.1% (noise) | 14 / 14 |
| 256×6144×2304 | 1 | 1,12,1,2,1 | 91.0 | 85.5 | 85.5 | 0.0% (noise) | 14 / 14 |

- **Noise floor:** for `Sm=1` opt-agg and opt-mm0 produce **byte-identical orders** (route identical), so the
  `Sm=1` `agg vs mm0` deltas (0 to +1.4%) are pure inter-subprocess measurement variance — establishing a
  ~1.4% noise floor for this harness.
- **Win:** agg beats mm0 by a clear, stable **−3.6 to −5.3%** (two runs) on the real 256×2048×1024 Sm=2
  primary (run-1 bands separated: agg [25.7,25.9,26.0] vs mm0 [27.3,27.4,27.4]); mm0 had left the *slave*
  ring's worst edge at 26 hops (agg cuts the group `aggmax` to 16). Marginal (−0.6%, both runs) on wide Sm=2.
- **Regression (accepted):** agg is **+2.7 to +3.0% slower** than mm0 on the synthetic Sm=4 config — a stable
  result (both runs) where agg's lower `aggmax` (16 vs 22) came with a *higher* `aggtot` (280 vs 273), total-hops mattered
  more there (the no-congestion-model caveat in action). This config is never selected by the picker (Sm=2 is
  2.8× faster on that shape: 26µs vs 73µs), so it is out of production scope.

**Decision (user-approved):** ship opt-agg as the production default — it wins −5.3% on the real primary and
is neutral on Sm=1 — and accept the +3% on the never-selected synthetic Sm=4 config as a documented caveat.
opt-mm0 is retained as `DIAG_RING_OPT_MM0` for A/B. (A total-hops-first objective might avoid the Sm=4
regression while keeping the Sm=2 win; left as a possible future refinement.)

## Correctness (gtest `RegimeADiagFixture.RingOrderCorrectness`)
bank / greedy / opt-mm0 / opt-agg, random BF16 vs a CPU f32 golden, PCC ≥ 0.999, fresh AND cached-program,
**all BIT-IDENTICAL** across orders, covering: Pk=1 + split-K (Pk=2/4/12 give both reader-NoC orientations),
Ns>1, **Sm=1/2/4**, W=1/W>1, balanced K/N tails. Public 20/20 suite passes on the agg default. Watcher clean
on the agg default (Pk=12). (A pre-existing, unrelated `in1_reader` Sm>1 atomic-flush watcher warning fires
identically at bank baseline — orthogonal to ring ordering.)

## Picker re-sweep (agg default)
Ring ordering is independent of picker config *selection*, and for `Sm=1` the op is byte-identical to the
prior ring commit, so only `Sm>1`-selecting shapes could shift. Re-ranked the top-10 candidates of the two
Mt=8 primaries under the agg default:
- **256×2048×1024:** winner `(1,4,2,2,2)` 26.21µs; best candidate `(1,4,2,2,4)` 26.11µs — **+0.4%, inside the
  ~1.4% noise floor** → no change. (Note: the progressive-era re-sweep had shown `nsb=4` +3.8%; that edge has
  since evaporated under the full pipelined-drain + ring-order stack — evidence that config optima drift and
  that leaving the picker unchanged then was correct.)
- **256×6144×768:** winner `(1,12,1,2,1)` unchanged.

No stable, noise-clearing improvement → **no picker entries changed**, so no six-shape-parity / FLUX/LTX
re-validation was triggered.

**Caveat:** an unchanged picker output does **not** prove the selected `(Ns,Pk,Sm,kb,nsb)` is still optimal —
this re-sweep only re-ranked the pre-change top-10 candidates per primary, not the full feasible space, and
did not re-sweep the whole Mt≥4 FLUX/LTX corpus. A full corpus re-sweep is deferred; the risk is bounded
because ring ordering does not alter the compute/reduction tradeoffs that dominate config selection.

## Why the effect appears (qualified)
The optimized ring order helps most on deep-Pk Mt=8 shapes and is neutral on small-M. On these shapes bank
order scattered the ring cores up to 25 directed NoC hops apart on a single forward, which opt cuts to ~14,
and all three RISC spans drop in lockstep — consistent with the in0 forward **route** being on the critical
path. This route-cost lever is distinct from forwarding *depth/bytes* (which earlier scatter/exchange/
replication experiments showed were already hidden behind compute). It is *plausible* that the recently
landed pipelined phase-2 drain, by tightening the reduction chain, increased the fraction of runtime where
the in0 route is exposed — but this ordering experiment did not run an A/B against the pre-drain baseline, so
that causal attribution is **not established**; the measured, attributable fact is simply that shorter
physical forward routes reduce wall time on these shapes.

## Notes
- The exhaustive 7!-per-group search runs host-side once per program compile; the RINGCOST diagnostic line is
  gated behind `TT_MM_RINGCOST` so the production path is silent on compile.
- Raw data: `regime_a_ringorder_bench.json` (agg run 2) + `regime_a_ringorder_bench_run1.json` (agg run 1) —
  every relaunch, per-RISC spans, and per-group RINGCOST. The Sm=1 bank/greedy/opt data is
  `regime_a_ringorder_bench_bankopt.json` (recovered from the prior ring-order commit).
