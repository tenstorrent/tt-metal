# Epic: Accelerate the inter-mesh minimal-host topology solve (auto-mapper)

**Owner:** Ridvan · **Area:** fabric / scaleout auto-mapper · **Tracking PR:** #51533 (experiments + reports)

## Problem
The inter-mesh minimal-host solve (place an N-stage pipeline ring onto physical hosts using the fewest hosts) is
intractable at scale on the current single-threaded incremental CaDiCaL path. Production symptoms (2x4 pipeline sweep,
SC36):
- **Single min-host solve:** 2x4-128 ~22 min; 2x4-144 does not finish (>25 min).
- **Multi-solution enumeration (`-n 20`):** 2x4 96/112/128/144 return **0 solutions** in the 15-min search budget
  (cannot even find solution #1); 2x4-64 takes ~47 min for 20; 4x4-72 fails (controller timeout -> MPI deadlock).

## Goal
Make the hard cases solvable and the enumeration workload tractable, **without regressing any solver feature**
(host cap/minimization, preferred at-least-k, pinnings, symmetry, unique-shape / distinct-host-set dedup, incremental
multi-solution).

## What we found (see SOLVER_ACCELERATION_MASTER_REPORT.md for full data)
- **gimsatul (external, one-shot, clause-sharing) crushes the single hard solve:** 128 with host cap 1346s -> 73s
  (18.5x); 144 solvable (~105s) where CaDiCaL times out. Instance-dependent; best thread count non-monotonic.
- **gimsatul unblocks the production-DEAD enumeration cases:** as a prime it finds solution #1 on 96/112/128/144
  (6s..101s) where production gets 0 -> converts "0 solutions" into "a few". 144 fully solved (1 distinct host-set).
- **gimsatul does NOT help enumeration past the prime** (no incremental API -> leaves the CaDiCaL enumerator cold;
  CaDiCaL-only iterative beats the hybrid on easy/mid cases). Enumeration cost is **super-linear** in the tail.
- **fastsat** helps incremental ~5%. **Warm descent** is the worst min strategy (times out at 128). **Partition by
  pinning one stage** does NOT speed a single hard solve.

## Proposed work (stories)

### Story 1 — gimsatul hybrid for the single hard min-host solve  [highest ROI, prototyped]
Delegate the heavy hardcap solve to gimsatul (DIMACS round-trip; faithful clause tee; env-gated); keep CaDiCaL driving
encode/cap/decode/preferred/symmetry. Race a couple thread counts + pure-CaDiCaL cold-cap, first-to-finish (covers
instance-dependence + non-monotonic threads). **Unlocks 128 (73s) and 144 (~105s).** Effort: M (productionize the
prototype: build dep, subprocess/lib, fallback).

### Story 2 — gimsatul-prime to unblock the DEAD enumeration cases  [prototyped]
Use gimsatul only to find solution #1 on the cases production returns 0 for, then enumerate `.next` on CaDiCaL.
Turns 0 -> a-few solutions on 96/112/128; fully solves 144. Effort: S (compose Story 1 into the enumeration prime).

### Story 3 — incremental vs from-scratch enumeration benchmark  [in progress]
Quantify the value of incremental state reuse: enumerate 20 distinct solutions incrementally (reuse solver) vs from
scratch (rebuild + replay blocks each solution), CaDiCaL-only hardcap. Informs whether to invest in "keep warmth" or
"raw re-solve". Effort: S. (Experiment + hypothesis below.)

### Story 4 — attack the super-linear enumeration tail  [research/spike]
The tail (finding the Nth distinct solution) is the real blocker for `-n 20` on hard cases. Candidates:
- **Repeated-cold-gimsatul enumeration** (each solution = fresh gimsatul + blocking clause) — promising where
  incremental stalls; gimsatul's cold power may beat the incremental stall.
- **Cube-and-conquer** (march_cu picks good split vars -> thousands of cubes -> parallel CDCL; enumerate per cube).
  The proper parallelization (naive pin-one-stage failed).
- **Partitioned parallel enumeration** (N incremental CaDiCaL over disjoint sub-spaces) — risks: skew, and
  partition-axis vs distinct-host-set dedup mismatch (cross-thread duplicates). Needs a de-risking measurement first.
- **Mallob** (only true parallel-incremental; heavy MPI). **Dedicated AllSAT** (bc_minisat_all; loses features).
Effort: L. Spike each; pick the winner.

### Story 5 — fastsat + freeze-then-eliminate on the incremental path  [small]
Turn on fastsat for enumeration (~5%); freeze assign vars so BVE runs on auxiliaries during incremental enumeration.
Effort: S.

## Acceptance / success metrics
- 2x4-128 single min-host solve < 5 min (target: ~1-2 min). ✅ met by Story 1 (73s).
- 2x4-144 produces a valid solution (currently: never). ✅ met by Story 1/2 (~105s / 64s).
- 2x4 96/112/128 enumeration returns >0 solutions (currently 0). ✅ met by Story 2 (1-4).
- 2x4-128 enumeration to 20 distinct solutions tractable (< ~30 min). ❌ open — Story 4.
- No regression on preferred / pinnings / cap / dedup. (All hard-clause encoded; preserved by design.)

## Risks / notes
- gimsatul is an external one-shot binary (build dep; no incremental; CLI subprocess round-trip). Fallback to CaDiCaL
  required. Best thread count is instance-specific.
- The super-linear tail may be intrinsic near the fill cliff (tiny, rigid solution space) — Story 4 may cap what's
  achievable for large `-n`.

---
## Story 3 experiment (running) — incremental vs from-scratch enumeration
**Question:** how much does incremental state reuse (learned clauses + saved phases + VSIDS) actually save when
enumerating 20 distinct (unique-shape / distinct-host-set) solutions?
**Setup:** CaDiCaL-only, MIN_MODE=3 hardcap (no warm descent), 2x4-64 (or -80), `-n 20`.
- **A. Incremental:** one solver; solve -> add blocking clause -> solve (current behavior).
- **B. From-scratch:** each solution = fresh solver + re-encode + replay all prior blocking clauses + hardcap solve.
**Hypothesis:** Incremental wins on easy/mid cases — it reuses the elite learned clauses + warm phases so each next
solution is cheaper, and avoids re-encoding. Expected incremental << from-scratch, with the gap WIDENING as solutions
accumulate (from-scratch re-pays the full solve each time; incremental amortizes). Counter-possibility: near the fill
cliff the warm phases point at the just-blocked region and help little, so the gap could be small — which would itself
be evidence that warmth is low-value there (and that a cold/parallel re-solve strategy is viable).
