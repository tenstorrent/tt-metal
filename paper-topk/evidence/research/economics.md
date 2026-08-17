# Incumbent Economics — the Headroom Table a Threshold-Select Selector Must Clear

**Scope:** committed/archived data only (no device runs). Repo `/home/nachiket/tt-metal`, branch `nkapre/sorting`.
**Sources:** `TOPK_LEDGER.html` (committed, rendered from the deterministic `--competition` sweep by
`tests/ttnn/unit_tests/operations/reduction/_topk_ledger_render.py`; result JSONs use `ns_median`, script "never invents a number" — render script docstring lines 4–27), `RADIX_BUCKET_GPU.md` §4–§6, `SORTING.md`, `tt_metal/tt-llk/tests/docs/THRESHOLD_SELECT_DESIGN.md` §5, campaign memory.
**Clock convention:** 1.35 GHz (THRESHOLD_SELECT_DESIGN.md:494 uses it explicitly) → 1 kcyc ≈ 0.74 µs.

---

## 1. Incumbent numbers, per cell (the committed ledger, competition2 run — 24/24 MEASURED)

All µs, single row, BH silicon, Tracy device-kernel duration, per-cell subprocess isolation
(TOPK_LEDGER.html:257–261 methodology block). "op" = `ttnn.experimental.topk_large_indices`
with the multi-core log-tree factory (post tree-merge commit 8794fbb); "routed" =
`ttnn.topk` via PR2 routing (values-native). Row line numbers are in TOPK_LEDGER.html.

| cell | stock ttnn.topk | stock topk_large_indices (rows=2 proxy) | routed ttnn.topk | **op (incumbent)** | cores | blaze | roofline | ledger line |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| k512 @ 32,768 | 80.2 ms | 140.6 | 87.0 | **11.9** | 52 | — | 0.9 | :113 |
| k512 @ 65,536 | 161.6 ms | 279.2 | 93.4 | **15.0** | 52 | — | 1.1 | :114 |
| k512 @ 262,144 | 648.4 ms | 1,111.0 | 145.4 | **32.0** | 52 (P=64 in psweep, :228) | — | 2.4 | :116 |
| k2048 @ 65,536 | 631.5 ms | 356.6 | 70.8 | **41.9** | 26 | 24.5 (fused SDPA+topk, not an op) | 1.8 | :130 |
| k2048 @ 262,144 | 2,557.6 ms | 1,416.6 | 122.3 | **58.6** | 52 | — | 3.9 | :132 |

**K ≤ 64 cells** (no competition-table coverage; from the committed A/B table and archived Tracy baseline):

| cell | path today | time now | evidence |
|---|---|---:|---|
| k32 @ 65,536 | stock ttnn.topk, **1 core** (W<65535 multi-core gate is strict; 65536 falls off) | baseline 10,956 µs; ÷1.154 replay-STORE → **≈ 9,494 µs** | TOPK_LEDGER.html:182; gate fact in campaign memory |
| k32 @ 8,192 | stock ttnn.topk multi-core (65 cores) | **≈ 107 µs** (replay-insensitive, 0.999×) | TOPK_LEDGER.html:180 |
| k32 @ 32,768 | stock multi-core | **≈ 171 µs** | archived `sweep/tracy_baseline.csv` (campaign memory); RADIX_BUCKET_GPU.md:622 cites the same point |
| k32 @ 65,536 — *honest* bar | route through the k=512 op and truncate (trivial, no selector needed) | **≈ 15 µs** | derived from ledger :114 |

The K≤64 regime's headline gap (9.5 ms) is a **routing hole, not an algorithm gap**: it is
closable today by widening the multi-core W gate or truncating the large-k op. Any selector
pitched at K≤64 must be priced against ~15 µs, not 9.5 ms (this is exactly §5.3's
"best current per-cell baseline" language, RADIX_BUCKET_GPU.md:469–472, and the audit's
STRA-3/CRIT-4 complaint about strawman baselines).

---

## 2. The routed-composite envelope, NOW

Gather-era decomposition (committed, TOPK_LEDGER.html:197): k512@65536 routed = 134.0 µs =
**xl 34.1 + tilize 59.9 + untilize 18.2 + gather 11.8 + mask 10.0** — i.e. envelope ≈ 99.9 µs
of 134 (the "~100 of 134" in campaign memory).

Values-native + tree (competition2, what is committed now): envelope = routed − op:

| cell | routed | op | **envelope now** |
|---|---:|---:|---:|
| k512 @ 65,536 | 93.4 | 15.0 | **78.4 µs** (identical to the values-native-era 112.4 − 34.0 = 78.4 — the envelope did not move; only the op did) |
| k2048 @ 65,536 | 70.8 | 41.9 | **28.9 µs** |
| k512 @ 262,144 | 145.4 | 32.0 | **113.4 µs** |
| k2048 @ 262,144 | 122.3 | 58.6 | **63.7 µs** |

Caveat: routing may pick a different P than the op column, so "routed − op" is an upper-ish
bound on pure layout cost; the k512 envelope (tilize-dominated after gather+mask were
dropped) is nonetheless stable at 78.4 µs across two independent runs.

**Consequence for a selector:** the ttnn.topk envelope is *common to both sides* — a
selector routed the same way inherits it. The honest comparison is **op-level vs
op-level**: a threshold-select path must undercut **11.9 / 15.0 / 32.0 / 41.9 / 58.6 µs**,
not the routed or stock columns. Corollary: at k512@65536 the single biggest remaining
prize is the **78.4 µs tilize envelope** (native row-major ingestion), worth 5× more than
zeroing the op's entire 15 µs — and it requires no new selection algorithm at all.

---

## 3. Honest selector cost model (per RADIX_BUCKET_GPU.md §6: CRIT-1, STRA-2, IMPL-2)

### 3.1 Measured constants (all SORTING.md, silicon-validated)

| constant | value | line |
|---|---|---|
| SFPU mask/filter map (Load+SFPGT+SFPSTORE) | 1.003 cyc/vec | SORTING.md:1043 |
| exact single-threshold count (CountD1) — **architectural floor** | 1.997–2.0 cyc/vec | SORTING.md:1045, 1217, 1256 |
| 3-bit histogram (HistMacro+HistSum) | 3.0 cyc/vec (1.00 cyc/bit) | SORTING.md:1342 |
| 8-bucket nibble histogram — measured, **loses** once clamped | 5.0 cyc/vec | SORTING.md:1221, 1288–1292 |
| data-dependent rendezvous (PassSync) | **≥ 25.1 cyc per decision** | SORTING.md:1220 |
| unpack_to_dest stream floor (fp32) | 3.855–3.938 cyc/vec; **SFPU is additive on top** (same-Dest serialization) | SORTING.md:543, 1638–1640 |
| bf16 end-to-end bisection, data resident | ~25.0 cyc/vec (≈3.0× vs local sort only) | SORTING.md:1344–1352 |
| topk_local_sort end_phase 5 (final sort of K) | 76.195 cyc/vec → K=512: ~1.2 kcyc ≈ 0.9 µs; K=2048: ~4.9 kcyc ≈ 3.6 µs | SORTING.md:1051 |
| threshold search, prior + explicit verify (1 pass), N=32k | 2,073 cyc | SORTING.md:1283 |
| packer-histogram composed model, N=32k/K=32 | 1,267 cyc (search 128 + filter 1,027 + finish 112) | SORTING.md:1515 |
| standalone pack when no producer fusion (fusion mechanism does not exist in TTNN) | 806 cyc | SORTING.md:1523 |
| shipping-filter economics vs best merge | relucomp 4.034 / negfilter 6.415 vs xl_merge 6.879 → the 6.7% signed edge **inverts** vs the 1.438 fused macro merge | SORTING.md:278–281, 305–308, 77 |
| where threshold-select loses outright | N ≲ 2,048–8,192/core; bitonic is oblivious (MOP/replay, zero readback) | SORTING.md:1366–1371 |

Audit-mandated structure of the model (RADIX_BUCKET_GPU.md): the exact arm is **threshold
bisection** — 1 bit @ 2.0 cyc/vec or 3 bits @ 3.0 cyc/vec, ≥25.1-cyc rendezvous per
data-dependent decision, count additive to the unpack floor on the stock path (§6.1 items
2–3, :589–607); pre-Gate-4 (no materialization) all passes are strictly additive to the full
bitonic emit — no win region exists (IMPL-2, :706–712); 2×/12.5× and 41.4%/6.7% figures are
strawman-denominator internal ratios (STRA-2 :798–804, CRIT-1 :721–727, CRIT-4 :827–833).

### 3.2 Composition (assumes Gate 2/4 materialization SUCCEEDS — the load-bearing gate)

Per row, column-parallel across C cores, per-core elements e = N/C, using
THRESHOLD_SELECT_DESIGN.md §5 (:470–530), which is the design's own paper model:

- **streaming compute:** fast path 0.42 cyc/elt (pass 1 hist ~4.5–5.4 cyc/vec + pass 2
  filter/count ~8.2–10.2 cyc/vec + cascade), worst-case bounded fallback 4.0 cyc/elt
  (:483–489)
- **rendezvous / fixed:** 2 (fast) to 19 (worst) global O(µs) rendezvous + root refinement
  ≤ 8,192 cyc ≈ 6 µs + host dispatch ~5–10 µs → **F ≈ 15–25 µs, ASSUMED** (:490–496).
  Optimistic floor cross-checked against the incumbent op's own smallest cell
  (k512@2048 = 5.9 µs total, TOPK_LEDGER.html:109): dispatch ~5 µs is real, so
  F_optimistic ≈ **10 µs** is the most charitable defensible value.
- **final sort of K:** topk_local_sort once at the root: 0.9 µs (K=512) / 3.6 µs (K=2048)
- **materialization:** priced at ~0 in the optimistic column (Gate-4 miracle);
  in the realistic column it is inside pass-2 + retry margin. Pre-Gate-4 the model is
  strictly worse than the incumbent everywhere (IMPL-2) and is not tabulated.

### 3.3 The headroom table

Optimistic = F 10 µs + 0.42 cyc/elt + final K sort (everything unproven goes right:
Gate-4 compaction, split-Dest unpack overlap, no retries, no fallback).
Realistic = F 20–25 µs + measured pass laws (~0.5 cyc/elt) + final K sort + one retry margin.

| cell | incumbent-now (op) | selector optimistic | selector realistic | ratio (opt / real) | verdict |
|---|---:|---:|---:|---|---|
| k512 @ 32,768 (C=52, 630 elt/core) | **11.9 µs** | ~11.1 | ~21–26 | 1.07× / 0.5× | **dead** — incumbent sits below the selector's own optimistic fixed cost |
| k512 @ 65,536 (C=52, 1,260 elt/core) | **15.0 µs** | ~11.3 | ~21–26 | 1.3× / 0.6–0.7× | **dead-to-marginal** — only wins if F lands at the very bottom of its band *and* nothing else slips |
| k512 @ 262,144 (C=64, 4,096 elt/core) | **32.0 µs** | ~12.2 | ~22–27 | 2.6× / 1.2–1.5× | **marginal** — real but thin; note incumbent was still descending at the P=64 cap (ledger :228), so it moves too |
| k2048 @ 65,536 (C=26→64, ≤2,520 elt/core) | **41.9 µs** | ~14.4 | ~25–30 | 2.9× / 1.4–1.7× | **winnable** — the only ≤65k cell with post-fixed-cost headroom, driven by the incumbent's K-heavy merge tree |
| k2048 @ 262,144 (C=64, 4,096 elt/core) | **58.6 µs** | ~14.9 | ~25–30 | 3.9× / ~2× | **winnable** — grows with N (design doc's own k2048@1M model: ~25 vs ~50 µs, THRESHOLD_SELECT_DESIGN.md:517) |
| k≤64 @ 65,536 (§5.1's own scope) | 9,494 µs stock / **≈15 µs honest** (truncate op) | ~10.5 | ~21 | 1.4× / 0.7× vs honest bar | **dead as a selector cell** — the 600× is a routing fix, not an algorithm; vs the honest bar the selector loses realistically |

Dual-RISC BF16 two-byte-digit histogram (the audit's strongest unmeasured alternative,
RADIX_BUCKET_GPU.md:385): replaces the SFPU count passes with exact 256-bin scalar
histograms on BRISC/NCRISC — it changes the *streaming* term (unknown scalar cyc/elt;
2 exact passes, no rendezvous-per-bit), **not F, not the final sort, not the envelope**.
Since every dead/marginal cell above is fixed-cost-bound, the RISC engine cannot rescue
k512@≤65k; its leverage is the same two winnable cells plus N≥1M, where it would need to
beat ~0.5 cyc/elt of SFPU streaming with ~2 scalar passes at ≥2 RISCs — plausible,
unmeasured, and exactly what Gate 3's shootout (RADIX_BUCKET_GPU.md:431–438) is for.

---

## 4. CRIT-4 stop rule, evaluated with today's numbers

CRIT-4 (RADIX_BUCKET_GPU.md:827–833): *"measure the stock multicore topk whole-op time for
the exact §5.1 cell before Gate 2 work begins — if that number is already <2× the
threshold-select paper model, stop."* Read with §5.3's "best current per-cell baseline"
(:469–472), the incumbent is the op, not stock. Paper model = design doc §5 with its own
F_low = 15 µs (:494).

| cell | paper model (F=15) | 2× model | incumbent-now | inequality | ruling |
|---|---:|---:|---:|---|---|
| k512 @ 32,768 | ~16.1 | 32.2 | 11.9 | 11.9 < 32.2 | **STOP** |
| k512 @ 65,536 | ~16.3 | 32.6 | 15.0 | 15.0 < 32.6 (incumbent is below even **1×** the model) | **STOP** |
| k512 @ 262,144 | ~17.2 | 34.4 | 32.0 | 32.0 < 34.4 | **STOP** (right at the line; flips only if F < ~13.7 µs) |
| k2048 @ 65,536 | ~19.4 | 38.8 | 41.9 | 41.9 > 38.8 | **proceed** (margin 8%) |
| k2048 @ 262,144 | ~19.9 | 39.8 | 58.6 | 58.6 > 39.8 | **proceed** (margin 47%) |
| k≤64 @ 65,536 (naive) | ~15.5 | 31.0 | 9,494 (stock) | passes trivially | strawman — the gap is routing |
| k≤64 @ 65,536 (honest) | ~15.5 | 31.0 | ~15 (truncated op) | 15 < 31.0 | **STOP** |

Every k=512 cell and the §5.1 K≤64 scope fail the audit's own stop rule **today**. Only
the k=2048 column (and by extrapolation N ≥ 1M at large K) clears it, and k2048@65536
clears it by 8% — one more incumbent improvement (the ledger's open "P-cap raise" item,
TOPK_LEDGER.html:251–254, targets the ~20 µs class) erases it. Also note both surviving
cells sit inside the u16-fused-key N≤65,536 caveat only for the 65k cell; the 262k cell
requires the u32-index/full-key path the audit demanded be kept (§5.1, :406–414).

---

## 5. Conclusions

1. **The number to beat is op-level: 11.9–58.6 µs**, not the ms-scale stock column and not
   the routed column (its 28.9–113.4 µs layout envelope is common to any selector routed
   through ttnn.topk).
2. **Threshold-select is fixed-cost-bound in exactly the cells §5.1 scoped it to.** With
   F ≈ 15–25 µs (design doc's own assumption) vs incumbents of 11.9/15.0 µs, the selector
   cannot win at k512@≤65k or K≤64 even if every unmeasured gate succeeds. CRIT-4's stop
   rule fires on all of them.
3. **The only live economic case is large-K / very-long-row:** k2048@65k (41.9 µs, thin),
   k2048@262k (58.6 µs, ~2× realistic), and unmeasured N≥1M — precisely STRA-3's point
   that the radix-family literature advantage lives at large K, the regime §5.1 excluded.
4. **The capability case is separate from the speed case** and unaffected by this table:
   arbitrary k (k=100/17/5000, no pow-2 W, no k%16) is unsupported by all incumbents
   (SORTING.md:1358–1364; THRESHOLD_SELECT_DESIGN.md:518) — but it justifies a fallback-
   quality implementation, not a perf campaign.
5. **Bigger, cheaper prizes exist before any selector:** (a) the 78.4 µs k512 routing
   envelope (native-RM ingestion — 5× the op's whole budget); (b) the K≤64@65536 routing
   hole (9.5 ms → ~15 µs by truncation/gate-widening); (c) the incumbent's own open P-cap
   raise. Each moves the headroom table against the selector further.
6. **Gate order stands as audited:** Gate 2 materialization-given-known-threshold is the
   go/no-go experiment, and per this table it should be run — if at all — at k=2048,
   N ≥ 262,144, with the harness pinned (canonical sweep, .so-mtime stamping, replay-STORE
   arm), never at the K≤64 scope §5.1 originally named.
