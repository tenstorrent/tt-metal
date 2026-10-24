# Host-minimization solve — investigation & experiment catalog

Working doc for two goals:
1. **Make the large solves (esp. 2x4-144) cheaper** — optimize the loop/solve so it's faster.
2. **Cheapest reliable host minimization** — guarantee the minimum host count is used (today: the hard cap).

**Concrete target:** solve **2x4-128 *with* host minimization in under 30 min.**

---

## 1. The problem, and the two independent difficulties

The inter-mesh solve places an N-stage pipeline (a ring of logical meshes) onto physical hosts. Two things
have to happen, and the experiments show they are **separate difficulties**:

- **(A) Base embedding** — find *any* valid placement (consecutive ring stages on physically-connected
  hardware). Hardness grows only as the cluster gets *completely full*.
- **(B) Host minimization** — among valid placements, use the *fewest* hosts (`k_min = total_ASICs/32`). This
  is the expensive, combinatorial part for mid/large pipelines.

The minimum host *count* is free (arithmetic). The cost is finding a *witness* placement that achieves it.

Key evidence (2x4 on SC36, single solution):

| stage | k_min | base embedding only (no min) | with minimization (hardcap, 150s) |
|---|---|---|---|
| 64 | 16 | 32 ms | 11.3 s |
| 80 | 20 | 43 ms | timeout@150s → **6.4 min** with 30-min cap |
| 96 | 24 | 91 ms | 36.8 s |
| 112 | 28 | 63 ms | timeout@150s (retry running) |
| 128 | 32 | 146 s | timeout@150s (retry queued) ← **our target** |
| 144 | 36 | ~910 s (~15 min) | timeout@150s (retry queued) |

Reading this: for 64–112 the embedding is trivial (ms) and **all** the cost is minimization (B). At 128–144
even the plain embedding (A) becomes expensive as the cluster fills. So Goal 1 is mostly about (A) at full
fill; Goal 2 is about (B) for everything from ~64 up.

---

## 2. Experiment catalog (what each was, what it found)

All modes selected by `TT_TOPO_SAT_MIN_MODE` in `topology_sat_solve_minimize_groups`
(`tt_metal/fabric/topology_solver_sat.cpp`). Original behavior preserved at mode 0.

### Mode 0 — baseline (warm solve + soft descent + hard-cap lock)
Find any placement, then shave one host at a time (`<= k-1`, re-solve), each step under a conflict budget.
**Finding:** the soft descent is the villain — each host shaved off is exponentially harder; it times out on
most hard packings. Slowest, fewest optimal. **The soft descent should be removed.**

### Mode 1 — skip-descent (warm solve + hard-cap lock)
Skip the one-at-a-time descent; warm-solve for a feasible model, then jump straight to "fit in k_min".
**Finding:** as good as or better than baseline for far less work (2x4-64: 18.5 s vs baseline timeout).

### Mode 3 — hardcap-only (cold all-or-nothing, one solve)  ← current best
No warm solve. Directly assert "each used host is completely full, ≤ k_min hosts" and solve once. The
all-or-nothing tightening gives strong unit propagation.
**Finding:** fastest correct option (2x4-64: 11.3 s, beat warm-started). Reaches k_min on 22/30 at 150s;
**complementary** to skip-descent (each cracks cases the other times out on). With a 30-min cap it also solves
2x4-80 (6.4 min, optimal). This is the "hard cap" the target refers to.

### Mode 4 — atmostk (cold ≤k counter, one solve)
Same "one solve" idea but with the weak "≤ k" counter constraint instead of all-or-nothing.
**Finding:** weak propagation → times out on every hard 2x4. Proves the *all-or-nothing structure* is what
makes the hard cap fast, not merely "one solve".

### Mode 2 — greedy (constructive host-fill + verify, no SAT search)
Walk the ring, fill each host's slots, verify by unit propagation. `TT_TOPO_SAT_NO_MINHOST=1` also exists to
disable minimization entirely (plain embedding baseline).
**Finding:** reaches k_min on only 2/30. A **traversal-order bug** (BFS visits the ring from both ends, so the
first stages placed aren't contiguous and can't fill a host) — switched to DFS/linear but still fails larger
rings for a second reason: the host-fill heuristic + k_min pruning can't reconstruct the non-obvious packings
SAT finds. Instant where it does construct. **Most promising lever if made robust (see Goal 2).**

### Longer-timeout retries (in progress)
150s "NO SOLUTION" was a *timeout, not a proof*. With longer caps:
- No minimization: **all** 2x4 solve, including full-fill 144 (~15 min). Embedding is always possible.
- Hardcap-only 30-min: 2x4-80 solved (6.4 min, optimal). 112/128/144 running/queued — results in
  `warm_solve_experiment_results.md`.

### Profiling note (72-stage)
`[topo-sat-profile]` timers (build `-DTT_METAL_ENABLE_LOGGING=ON`, run `TT_LOGGER_LEVEL=debug`) show, for the
full-fill 72-stage, ~88% of the inter-mesh solve is a single CaDiCaL solve; encode ~23 ms. So the lever is the
*solve*, not encoding.

---

## 3. Goal 1 — make the big solves (144) cheaper / faster loop solves

144 no-min is ~15 min today; the cost is a single hard CaDiCaL search on a nearly-full cluster. Ideas to
investigate (roughly increasing effort):

1. **Warm-start / phase hints.** The `TT_TOPO_SAT_HARDCAP_WARMSTART` path already shows phase-hinting a cheap
   solve accelerates a harder one. Feed a greedy or ring-rotation candidate as `solver.phase()` hints before
   the big solve → CaDiCaL branches toward it, fewer conflicts.
2. **Incremental / reuse across the loop.** If the "loop solves" are the enumeration or descent, encode once
   and add only blocking/assumption literals between solves (the enumerate path already does some of this) —
   never rebuild the CNF.
3. **Structural ring solver.** A ring→ring/torus embedding is a rotation+reflection; solvable by construction
   (AC-3 + a single walk) with no CDCL. Biggest win on the full-fill case; narrow (pure rings).
4. **Symmetry breaking / better CaDiCaL config** for the first-feasible solve (target SAT, phase-saving,
   restart policy) — cheap to A/B.
5. **Decomposition.** Solve per-host-band and stitch; or contract fully-forced regions before the search.

## 4. Goal 2 — cheapest reliable host minimization (guarantee k_min)

Today = hard cap (mode 3): correct and fastest of the SAT options, but a full SAT search that scales badly.
Ideas:

1. **Greedy construct → k_min by design (mode 2, made robust).** If greedy builds a valid k_min packing, there
   is *no search* — it's ms. Fix the two failure modes: (a) DFS/linear order (done), (b) host-fill heuristic
   that matches how the real packings distribute stages (the observed packings spread consecutive stages across
   adjacent hosts rather than filling one host with 4 contiguous stages — the heuristic must reflect that).
2. **Greedy → hardcap fallback (hybrid).** Try greedy (ms); if it can't construct, fall back to hardcap. Best
   of both: ms on regular cases, correct everywhere. Add as a mode.
3. **Greedy as warm-start for hardcap.** Even a *partial* greedy packing phase-hinted into the hardcap solve
   should cut its search drastically (ties Goals 1 & 2 together).
4. **Tighter mock = easier.** Observed: 2x4-80 solved on the 20-host mock but timed out on 36-host (more hosts
   = bigger search). If the caller can restrict the candidate host set to ~k_min up front, the hardcap solve
   gets much easier. Worth exposing a "candidate host window" constraint.

## 5. Path to the target: 2x4-128 with host minimization in < 30 min

- **Data so far:** 128 embedding alone = 146 s; 128 hardcap at 150s timed out (30-min retry queued — that result
  decides whether plain hardcap already meets the target).
- **If hardcap-30min solves 128** → target met by "drop soft descent, use hardcap-only"; document the time.
- **If not** → combine Goal 1 + Goal 2: **greedy/ring-rotation candidate → phase-hint → hardcap** (idea 3.1 +
  4.3), and/or **restrict the candidate host set to ~k_min** (4.4). The greedy-warm-started hardcap is the most
  likely single change to bring 128 (and 112) under 30 min while guaranteeing k_min.

## Reproduce
- Modes: `TT_TOPO_SAT_MIN_MODE=0|1|2|3|4`; no minimization: `TT_TOPO_SAT_NO_MINHOST=1`.
- Direct producer harness + gen_table.py in the session scratchpad; results in `warm_solve_experiment_results.md`.
- Profiling: `-DTT_METAL_ENABLE_LOGGING=ON`, `TT_LOGGER_LEVEL=debug`, grep `[topo-sat-profile]` / `[intermesh-solve]`.

---

## GOAL-2 PARKED THOUGHT (revisit later)
The all-or-nothing "every used host is completely full" tightening is what makes hardcap fast on PARTIAL
fills (mode 4 without it times out), BUT it is pure overhead when the cap is NOT binding — at FULL fill
(k_min == number of available host groups) there is nothing to minimize, yet it made 2x4-144 go from
15.2 min (plain embedding) to TIMEOUT@30min. Proposed fix: guard to SKIP minimization / the all-or-nothing
when k_min >= available host groups (or more generally when the cap can't reduce host count). Being confirmed
via mode 4 on 144. Pick this up when we return to Goal 2.
