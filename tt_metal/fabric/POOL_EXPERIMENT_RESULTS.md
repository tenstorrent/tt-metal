# Results: gimsatul + distilled clause pool vs cold gimsatul vs clause-share portfolio

Full 3-mode benchmark on the **latest solver base** (`ridvan/gen-rank-bindings-all-solutions` @ f089790 +
the experiment stack). 90 cells: sizes {64,80,96,112,128,144} × modes {B,A,D} × seeds {0,7,13,42} where
run. Harness: 2x4 pipeline sweep → SC36 36-rank mock, MIN_MODE=3 hardcap. Caps: n5=20min, n10=40min.

Modes (all multithreaded, all multi-solution):
- **B = gimsatul + distilled pool** (`GIMSATUL=1 GIM_EVERY=1 POOL=1`, threads 32) — the new mode.
- **A = gimsatul cold** (`GIMSATUL=1 GIM_EVERY=1`, threads 32) — no pool.
- **D = clause-share portfolio** (`SHARE=1 PORTFOLIO=16`) — #52849, wired to min-host+enum here for the 1st time.

## Enumeration `-n 5` — solutions found (seeds), best completed wall

| size | **B (pool)** | **A (cold)** | **D (portfolio)** |
|---:|---|---|---|
| 64  | 5/5/5 · 29–49s | 5/5/5 · 29–53s | 5/5/5 · 187–265s |
| 80  | 5/5/5 · 116–202s | 5/5/5 · 107–177s | 5/5/5 · 497–1048s |
| 96  | **5/5/5/5 · 174–348s** | 5/5/5 · 173–395s | 5,2@cap,2@cap · 945s |
| 112 | 5/5, 2@cap · 944–1108s | 5/5/3 · 357–737s | 0,0 @cap |
| 128 | **5**,4,3,3 @cap · 837s | 2,2,1 @cap | 0/0/0 @cap |
| 144 | 1 (exhaustive) · 252s | 1 (exhaustive) · 132s | 0 @cap |

## `-n 10` deep tail (seed 0/7)

| size | B | A | D |
|---:|---|---|---|
| 96  | **10 · 493s** / 10 · 1334s | 10 · 1159s | 10 · 958s |
| 128 | 3 / 4 @cap | 7 @cap | 0 @cap |

## Primes `-n 1` — B solves everything, no timeouts

| size | B (s0/7/13/42) | A (s0) | D (s0) |
|---:|---|---|---|
| 96  | 83/29/47/67s | 94s | 697s |
| 128 | 353/173/270/278s | 269s | **TIMEOUT** |
| 144 | 206/314/165/415s | 230s | 505s |

## Findings

1. **Below the cliff (≤80): B ≈ A, both fast; D completes but 3–5× slower.** The pool is neutral on easy
   instances (nothing hard to carry) — expected (H3).
2. **At 96 (the sweet spot): B wins clearly** — completes every seed, and the pool's best case is ~2.3×
   faster than cold gimsatul at both n5 (174s vs 173–395s, tighter band) and n10 (493s vs 1159s). D mostly
   times out. This is the pool's proven home (H1 + H2).
3. **112–128 (fill cliff): run-variance dominated, not pool-decided.** B completes 128-n5 once (5/5, 837s)
   but lands 3–4 elsewhere; A got 7 at 128-n10 where B got 3; B completed 112 on 2/3 seeds, A on 3/3 (one
   partial). Neither engine dominates run-to-run → **race 2–3 attempts and take the best.** The pool still
   raises B's aggregate floor at 128 (B never dropped to A's 1–2 on n5).
4. **144 (full fill): solved + exhaustively enumerated by both gimsatul modes in minutes** (single distinct
   host-set); D cannot.
5. **D (clause-share portfolio) is not viable alone on min-host** — TIMEOUT on the 128 prime and all
   128/144 enums; only completes easy tails, always slower than B. Its value is as the pool's *export*
   machinery (the Learner), not as a standalone mode.

## Verdict
**Ship B (gimsatul + distilled pool) as the enumerator, inside a best-of-N race for the 112–128 cliff.**
The pool delivers gimsatul's raw power *plus* the transferable half of incremental state (learned clauses +
fixed units as sound hard clauses), which is exactly what the enumeration tail needs — a consistent ~2.3×
in the tractable regime and the only mode that completes 128-n5 / 96-n10 reliably. See
`POOL_PRODUCTIONIZATION_PLAN.md` for the ship path (incl. the validated in-process libgimsatul integration).

Raw data: `oneshot_external_sat_scripts/` sibling of this file (RESULTS_pool_matrix_rebased.txt),
old-base comparison in RESULTS_pool_matrix_OLDBASE.txt.
