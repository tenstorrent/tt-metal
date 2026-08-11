# Adaptive-portfolio enumeration — design plan (fastest `search_n`)

**Goal.** Make "find N solutions" (`topology_sat_search_n`) as fast as possible by **racing diverse strategies
concurrently on a shared frontier and keeping whatever produces solutions**, instead of committing to a single
strategy (today: one single-threaded incremental solver). All native CaDiCaL first; gimsatul added as one more producer.

## 0. Why this shape (from the measurements)

- Cost of an enumeration = **cost of solution #1** (a single hard solve; ~51 s of the 85 s on 128, ~60%) **+ the
  super-linear tail** (solutions 2..N). Near-tail solutions are cheap *when warm* (~3.5 s on 128); the far tail is the
  real blocker for large N.
- **No universal winner for the tail** (Story 3): warm-incremental wins when consecutive solutions are structurally
  *close*; cold from-scratch wins when they're *far* (warm phases mislead). Per-instance, not seed. → Don't pick; race.
- **gimsatul** cracks #1 (and rare hard solutions) fastest but is one-shot; naive "gimsatul prime → CaDiCaL" leaves the
  enumerator **cold** (tail ~40 s/sol). → Keep CaDiCaL warm and *concurrent*, not sequential after gimsatul.

## 1. Architecture — one shared frontier, many diverse producers

```
                 ┌───────────────── shared state (lock-guarded) ─────────────────┐
                 │  seen-set (dedup, source of truth) · results ·                 │
                 │  blocking-clause pool · learned-clause pool · latest models    │
                 └───────────────────────────────────────────────────────────────┘
   warm-incr CaDiCaL ×a   cold-restart CaDiCaL ×b   phase-hint exploit ×c   explore ×d   gimsatul ×e
        (near tail)          (far tail, Story 3)     (near a fresh sol)    (far tail)   (#1 + hard tail)
```

Every worker reads/writes the same shared state. The **frontier advances via whoever is fastest** — adaptivity is
*emergent*: a productive strategy publishes blocks that all others build on; a stalled strategy simply contributes
little. No commitment to "incremental."

## 2. Worker strategies (the racers)

1. **Warm-incremental CaDiCaL** (seed-diverse, clause-sharing) — one solver, blocking clauses added, learned clauses
   shared. Cheap for near solutions.
2. **Cold-restart CaDiCaL** (fresh solver + re-encode + replay all blocks each solution) — wins when solutions are far
   apart (Story 3). Reuses `TT_TOPO_SAT_ENUM_FROMSCRATCH` machinery.
3. **Phase-hint exploit CaDiCaL** — warm-started (`phase()` / `phase_hint_from_last_gimsatul_model`) from the latest
   found model → lands on nearby-distinct solutions fast (the cheap near tail).
4. **Diverse-explore CaDiCaL** (diverse seeds, NO hint) — hunts the rare far-tail solutions.
5. **gimsatul cold** (Phase 2) — 32-thread, DIMACS round-trip with the current blocks appended; cracks #1 and rare
   hard solutions its cold power finds fast.

## 3. Shared channels (how info moves, and how gimsatul helps CaDiCaL)

- **Blocking clauses — everyone.** Correctness + distinctness. Portable across engines/workers (deterministic encoding
  ⇒ identical var ids). This is the lingua franca between gimsatul and CaDiCaL.
- **Learned clauses — CaDiCaL workers (always); gimsatul (optional, Channel 3).** The clause-sharing pool. gimsatul
  learned clauses would enter via its DRAT proof filtered to short/low-LBD — high effort, add only if the tail is still
  cold.
- **Phase hints — latest model → exploit workers (per-worker, not pooled).** *The key warm-start.* gimsatul's (or any)
  fresh solution biases an exploit worker's branching to that region → the blocking clause forces one difference → a
  nearby distinct solution falls out fast. **This dissolves the "cold enumerator" problem** — hinted CaDiCaL isn't cold.
  DON'T hint all workers: hints help the near tail but hurt the far tail (bias at the just-blocked region) → keep
  explore workers un-hinted for diversity.

## 4. The "cold + warm race" (the ask), concretely

Run warm-incremental AND cold-restart workers **at the same time** on the same frontier. Whoever finds the next distinct
solution publishes its block; all advance. This resolves Story 3's no-universal-winner **automatically per instance** —
no regime detection needed. Same principle extends to gimsatul-vs-CaDiCaL for #1 and the hard tail.

## 5. Correctness invariants

- Every emitted solution is distinct — the shared `seen`-set (shape-key when unique_shapes) is the single source of
  truth; races (two workers find the same before blocks propagate) are caught there.
- Blocking clauses remove exactly one solution/shape → sound for enumeration (no valid solution lost beyond the intended
  exclusions). Learned clauses are entailed → model-preserving.
- Termination: a worker that hits UNSAT under all current blocks ⇒ exhausted (adding blocks only tightens) ⇒ set `done`;
  reaching N ⇒ set `done`. `done` cancels all workers (atomic flag + terminator poll).

## 6. Phasing (easy → full; each independently shippable, env-gated)

- **Phase 0 — injection point (BLOCKER).** Route the *hard* multi-mesh inter-mesh solve (where the ~50 s + the
  `intermesh-solve`/`Solver running Ns` heartbeats live) through the portfolio. Today the collaborative enum only
  catches `search_n` leaf sub-solves; nothing hits the real bottleneck until this is done.
- **Phase 1 — CaDiCaL adaptive portfolio (80/20, all native, low risk).** Heterogeneous CaDiCaL workers: warm-incr +
  cold-restart + phase-hint-exploit + diverse-explore, sharing blocking + learned clauses on the shared frontier.
  Reuses `clause-sharing-enum` (pool + collaborative loop) + `TT_TOPO_SAT_ENUM_FROMSCRATCH` + `phase()`. Captures the
  fast #1 (~9×) + warm/cold-raced tail. **Build this first.**
- **Phase 2 — gimsatul producer.** Add gimsatul cold worker(s) to the same pool (reuse `gimsatul_solve` DIMACS
  round-trip from `exp-clause-sharing-portfolio`, extend dump to append current blocks). Models → phase hints
  (Channel 2). gimsatul takes #1 + hard rare solutions; CaDiCaL sweeps the cheap ones.
- **Phase 3 — adaptive core reallocation (optimization).** Monitor per-strategy solution yield; shift cores from
  stalled strategies to productive ones (a simple bandit). Nice-to-have; the shared pool already self-balances.

## 7. Estimated payoff (128 → 5 solutions, vs 85 s single-threaded today)

- Phase 1: #1 ~9 s (portfolio) + warm/cold-raced tail ~15–20 s + ~14 s overhead ≈ **~40 s (~2×)**, and far more on
  first-solution-dominated / production-dead cases.
- Phase 2: #1 as low as gimsatul's ~0.2 s (base) with the warm tail preserved via phase hints → tail no longer cold.
- Phase 3: better core utilization on the far tail (large N).

## 8. Metrics to capture

Time-to-#1, time-to-N, per-solution curve (is the tail flattened?), per-strategy yield (who's actually producing),
distinct-solution correctness. Baselines: 128 `-n5` single-threaded = 85 s; 144 = the exact-fill (≈ one distinct
solution) case.

## 9. Easiest highest-impact next step

**Phase 0 + Phase 1a:** find the multi-mesh hard-solve injection point and route it through the existing collaborative
clause-sharing portfolio, then add the warm/cold worker split. Mostly wiring + reuse — no new solver engine — and it's
the single change that turns "nothing hits the bottleneck" into "fast #1 + raced tail."
