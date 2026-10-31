# Experiment plan: distilled clause pool — CaDiCaL incremental knowledge baked into gimsatul

Branch: `rsong/exp-gimsatul-pool` (base: `ridvan/clause-sharing-portfolio` = solver-min + clause-sharing
portfolio; gimsatul hybrid re-applied from `oneshot_external_sat_scripts/hybrid_code/gimsatul_hybrid.patch`).
Companion docs: `SOLVER_ACCELERATION_MASTER_REPORT.md`, `SOLVER_ACCELERATION_EPIC.md`.

## Hypothesis
gimsatul cannot be incremental (immutable shared clause DB), but the *transferable half* of incremental
state — learned clauses and root-level fixed units, which are formula-entailed — can be baked into each
gimsatul run as ordinary hard clauses. This should:
- H1: speed up repeated gimsatul solves (enumeration: each restart inherits everything already proven);
- H2: flip the 96-stage case (where warm CaDiCaL beat cold gimsatul — the warmth was the edge);
- H3: change 128/144 primes only modestly (cold gimsatul already dominates there).
The non-transferable half (saved phases, VSIDS) was shown harmful/neutral at the fill cliff, so losing it
is acceptable by design.

## Implementation (env-gated, no default behavior change)
1. **Re-apply the gimsatul hybrid** onto this branch (from `hybrid_code/`): DIMACS tee + one-shot gimsatul
   subprocess for the heavy solve (`TT_TOPO_SAT_GIMSATUL=1`, `_BIN`, `_THREADS`), assumptions baked as units.
2. **Distilled pool** (`TT_TOPO_SAT_POOL=1`):
   - Export: reuse the clause-sharing `Learner` hook — every CaDiCaL worker publishes learned clauses of
     size ≤ `TT_TOPO_SAT_POOL_MAX_LITS` (default 8) into a session-persistent pool (dedup by sorted lits);
     root-level fixed units harvested after each solve.
   - Soundness: pool entries tagged with the host-cap level they were learned under; a gimsatul dump only
     includes entries whose cap tag is ≥ (looser than or equal to) the run's cap. Hardcap mode = one tag.
   - Inject: gimsatul dump path appends pool clauses (capped at `TT_TOPO_SAT_POOL_MAX_CLAUSES`,
     default 20000, shortest-first) after base CNF + blocking clauses.
3. gimsatul binary via `oneshot_external_sat_scripts/build_gimsatul.sh`.

## Benchmark matrix (same harness/mock as the master report: 2x4 pipeline → SC36)
Modes (all multithreaded; plain single-thread CaDiCaL is NOT a baseline — the warm/incremental side is
represented by the clause-share portfolio, which is the actual production candidate):
- **A. gimsatul cold** (`GIMSATUL=1`, threads 32, pool off) — known from #51533; rerun as same-harness anchor.
- **B. gimsatul + pool** (`GIMSATUL=1 POOL=1`) — the NEW mode; full grid.
- **D. clause-share portfolio** (`SHARE=1 PORTFOLIO=16`, incremental workers) — known from #52849; rerun as
  same-harness anchor.
A-vs-D comparisons already exist in the two PRs; the experiment's job is placing B against both on the
same harness.

**All modes are multithreaded AND multi-solution.** Per-mode enumeration semantics:
- A/B: gimsatul-every-step (`TT_TOPO_SAT_GIM_EVERY=1`) — every enumeration step is a fresh gimsatul run on
  base CNF + accumulated blocking clauses (+ pool for B). Not just a gim-prime.
- D: the 16 sharing workers drive every enumeration step; blocking clauses are delivered to all workers;
  the shared pool persists across steps (sound: the formula is add-only); distinct-host-set dedup is global.

Tasks × sizes × seeds:
- **B (new)**: prime at {96, 128, 144} × seeds {0, 7, 42}; enumeration `-n 5` at {96, 128} × seeds {0, 7, 42};
  144 enum once (single distinct host-set, should finish outright).
- **A and D (anchors)**: same cells at seed 0 (plus seeds 7/42 only where the B result is close enough that
  variance matters).
Budget guard: per-run cap 15 min (prime) / 20 min (enum); sequential sweep.

## Metrics & decision
- time-to-first-solution; time-to-#k (k=2..5); #distinct solutions within cap; variance across seeds.
- **Pool wins** if B beats A on enumeration time-to-#k (H1) and/or flips 96 (H2) without regressing 128/144
  primes materially. **Portfolio wins** if D matches/beats B across the board (then no gimsatul dependency
  is justified). Ties → prefer D (no external dep, natively incremental).
- Raw outputs: `oneshot_external_sat_scripts/RESULTS_pool_*.txt`; consolidated table appended to this file.
