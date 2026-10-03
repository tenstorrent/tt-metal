# Plan: swap the deterministic SAT engine CaDiCaL → kissat_extras (incremental Kissat)

**Area:** fabric / scaleout inter-mesh min-host SAT solver · **Status:** ACCEPTED — implementation to follow
(this PR is the design/plan of record; no code changes here) · **Author:** (rsong) · **Companion
experiments:** `MERGESAT_EXPERIMENT.md`, `HARVEST_FINDINGS.md`.

## Decision (accepted)
**We will use kissat_extras as the deterministic SAT engine, replacing CaDiCaL 2.2.1.** Rationale: 2–150×
faster on the real min-host CNFs, deterministic (verified), genuinely incremental (where CaDiCaL's incremental
stalls), and algorithmically identical to CaDiCaL so placement *quality* is unchanged. Preferred constraints
stay exactly as-is (hard cardinality) — no re-encoding. The pool / learner / phase-warm machinery is deleted
(net-negative, and depended on CaDiCaL-only hooks). CaDiCaL is kept as a one-env-var fallback during rollout.
Considered and **declined**: phase-based soft-preference biasing (would keep us on CaDiCaL for a heuristic that
kissat's raw speed almost certainly outweighs; not worth blocking the swap). Implementation follows the phased
plan in §7.

This document lays out exactly what we implement, what we delete, what improves, what we cannot yet do, and the
kissat features we can adopt. **This PR does not perform the swap; it is the agreed plan.**

---

## 1. Why consider this

The min-host inter-mesh solve needs a *deterministic* engine for reproducible rank-binding output and as the
incremental driver / fallback. Today that role is CaDiCaL. Benchmarks on the real dumped min-host CNFs show
kissat_extras is a **faster deterministic core** on the same formulas, is genuinely **incremental**, and —
critically — is **algorithmically identical** to CaDiCaL (Kissat is a C reimplementation of CaDiCaL's
algorithms), so it won't change *which* solutions we find, only how fast.

### Single-solve speed (same CNF per size; deterministic engines in bold)
| stage | vars | **CaDiCaL 2.2.1 (1t)** | **kissat_extras (1t)** | gimsatul (32t) | Mallob (2×16) |
|---:|---:|---|---|---|---|
| 32  | 14.5k | 1.6s   | **0.17s** | 0.18s | 0.05s |
| 64  | 28.3k | 63s    | **20.5s** | 8.0s  | 1.6s  |
| 80  | 35.2k | 777s   | **5.3s**  | 23s   | 10.1s |
| 96  | 42.1k | 1085s  | **84s**   | 32s   | 9.8s  |
| 112 | 49.0k | (DNF-ish) | **168s** | 43s | 101s |
kissat_extras beats CaDiCaL at every size (2–150×), sometimes beating even 32-thread gimsatul (80-stage).
Both are single-thread deterministic; the parallel nondeterministic solvers (gimsatul/ParKissat/Mallob) are
faster still but out of scope here (they serve the *speed* path, not the deterministic path).

### Incremental enumeration (block previous model, re-solve), all deterministic (verified over repeat runs)
- kissat_extras: solve-1 fast, then warm re-solves. At SMALL sizes (32/64) later solves are ~1ms *trivial
  variants* (flip one free var); at LARGER sizes (96) solve-2 is a genuine 51s search, and at 80/112 it can
  stall — i.e. the dramatic small-instance speed is partly a blocking artifact, not a universal win.
- CaDiCaL: incremental solve-2 stalls even at 64-stage (>337s) — effectively not viable for enumeration.
- MergeSat (det. parallel): every solve genuinely distinct but slow MiniSat core (~5× kissat).

**Takeaway:** for the deterministic engine role, kissat_extras is a strict upgrade over CaDiCaL on raw speed,
and a real (not artifact-inflated at scale) incremental capability where CaDiCaL's incremental stalls.

### Determinism (verified, not assumed)
kissat_extras is single-threaded with a fixed seed, so both the search path and the returned models are
reproducible. VERIFIED by running the incremental enumeration twice per instance: identical solution sequence
each run (only ~100ms wall-clock jitter). cnf64 incremental, run1 vs run2: solve-1 20584/20484ms, solve-2
6268/6334ms, solves 3-20 0-3ms both — same models, same order. So the deterministic-mapping guarantee holds
END-TO-END through enumeration: each blocking clause is derived from the previous (deterministic) model, so
the whole sequence of rank bindings is bit-reproducible. Two conditions (both met by the plan): stay
single-threaded (kissat has no threads anyway), and fix the seed. Introducing a parallel portfolio would break
determinism — which is why the *deterministic* role stays single-thread kissat and the *speed* role stays
gimsatul.

## 2. Feature compatibility — what the current solver uses vs what kissat_extras provides
CaDiCaL calls used by `TopologySatSolver::Impl` and their kissat_extras equivalents:
| CaDiCaL feature | kissat_extras | notes |
|---|---|---|
| `add`, `assume`, `solve`, `val` | `kissat_add/assume/solve/value` | ✅ core — full parity |
| `limit("conflicts", n)` | `kissat_set_conflict_limit` | ✅ |
| `reserve(maxvar)` | `kissat_reserve` | ✅ |
| `set(opt,val)` | `kissat_set_option` | ✅ (option NAMES differ — see §5) |
| terminator (`connect_terminator`) | `kissat_set_terminate` (callback) | ✅ (re-wire heartbeat as a callback) |
| assumption-failed | `kissat_failed` | ✅ bonus |
| `fixed(v)` | — | ❌ MISSING — used only by pool `harvest_fixed_units` (dead end) |
| `phase`/`unphase` | — | ❌ MISSING — used only by optional warm-start hints (dead end, see §4) |
| `connect_learner`/`disconnect_learner` | — | ❌ MISSING — used only by pool `ClauseExportLearner` (dead end) |
| `write_dimacs` | — | ❌ MISSING — solver already emits its own tape (`dump_tape_`) |
**Every missing function is either dead-end machinery or already self-provided.** No productive capability is
lost for the min-host solve as it exists today. See §6 for the one *future* limitation.

## 3. What we would implement, and how
Mirror the in-tree gimsatul integration (vendored static lib + thin C-API use, gated by an option/env):
1. **Vendor** `jix/kissat_extras` under `tt_metal/third_party/kissat` (pinned commit), with a CMake target
   building `libkissat.a` (`./configure -O3` flags baked in; `option(TT_METAL_FABRIC_KISSAT ... ON)`), same
   pattern as the gimsatul target. No shim needed — `kissat.h` is a clean C API.
2. **Engine abstraction.** Introduce a minimal internal interface behind `TopologySatSolver::Impl` with two
   backends: `CadicalEngine` (today) and `KissatEngine`. Select via `TT_TOPO_SAT_ENGINE=kissat|cadical`
   (default cadical initially → flip after validation). Only the ~8 core methods (add/assume/solve/
   solve_limited/val/reserve/terminate/set_option) are on the interface.
3. **Option mapping.** Map CaDiCaL option usage to kissat: `set("quiet",1)`→ kissat verbosity/quiet;
   `set("ilb",2)` (incremental lazy backtracking — CaDiCaL-specific) has **no kissat analog** → drop (kissat
   manages incremental trail reuse internally). Enumeration must therefore rely on kissat's own incremental
   trail handling, not ILB.
4. **Terminator as callback.** `kissat_set_terminate(solver, data, cb)` where `cb` polls the existing
   `HeartbeatTerminator` cancel flag. Replaces `connect_terminator`.
5. **Incremental protection (kissat-only improvement).** Before enumeration, `kissat_protect()` the
   mapping/assignment variables so bounded-variable-elimination doesn't remove them between solves — keeping
   incremental blocking sound and efficient. CaDiCaL does this implicitly (heavier restore logic); kissat
   gives explicit control.

## 4. What kissat lets us DELETE (code removal) and IMPROVE
The whole clause-pool / warm-start apparatus was shown net-negative (see HARVEST_FINDINGS.md). Swapping to
kissat (which lacks the hooks anyway) is the natural moment to **remove** it:
- **Delete** `DistilledClausePool`, `ClauseExportLearner`, `harvest_fixed_units`, `note_permanent_cap` pool
  plumbing, `pool_cap_tag`, and the `TT_TOPO_SAT_POOL*` env surface. (Depends on `fixed`/`connect_learner`.)
- **Delete** the phase warm-start paths: `phase()/unphase()`, `phase_hint_from_last_gimsatul_model`,
  `apply_base_warmhint`, `TT_TOPO_SAT_BASE_WARMHINT`, `TT_TOPO_SAT_PHASE_WARM`. (Depends on `phase`.) These
  were best-effort branching hints and were shown *harmful at the cliff* — dropping them is a feature.
- **Keep** the gimsatul in-process path unchanged for the *speed* (nondeterministic parallel) role.
IMPROVE: faster deterministic solves (§1), genuine incremental (where CaDiCaL stalls), `kissat_protect` for
incremental soundness, and a smaller, simpler codebase once the pool/warm-start code is gone.
On PLACEMENT specifically: no change to solution *quality* (same algorithms, same hard/cardinality encoding of
`preferred`/host-cap) — only faster to reach the same placements, and reproducibly.

## 5. What we CANNOT do from CaDiCaL yet (limitations / risks)
1. **Soft-preference via `phase()`** — kissat has no public phase API, so we cannot *softly* bias branching
   toward preferred assignments. TODAY this is a non-issue (preferred = HARD at-least-k cardinality clauses,
   pure CNF — §preferred in topology_solver_sat.cpp). It only forecloses a *future* soft-preferred design;
   for that we'd use weighted cardinality CNF or a MaxSAT layer (neither engine has native MaxSAT).
2. **`ilb` (incremental lazy backtracking)** — CaDiCaL-specific trail-reuse *optimization* (not a capability)
   used by `configure_for_blocking_clause_enumeration`. IMPORTANT: `ilb` is NOT why kissat's incremental wins.
   CaDiCaL *has* `ilb` (we enable it) and STILL stalls on the blocked formula (solve-2 >337s @64); kissat
   *lacks* `ilb` and is faster anyway (solve-2 6.3s @64). So losing `ilb` does not cost incremental
   performance on our workload. The residual risk is narrow: some *other* workload (many small assumption-only
   re-solves) could benefit from trail reuse, so validate broadly (§6). Note kissat has `restartreusetrail`
   (trail reuse across restarts) which covers part of the same ground.
3. **Learner-based clause export** — gone (pool dead-end). If we ever want cross-solve clause sharing in the
   deterministic path, kissat would need its clause-export API exposed (Mallob's kissat has `clauseexport.c`;
   `jix/kissat_extras` may not expose it) — not planned.
4. **`write_dimacs`** — replaced by our own `dump_tape_` (already the source of truth for the gimsatul feed).
5. **Vendoring a third-party fork** — `jix/kissat_extras` is a community fork, not upstream Kissat (upstream
   Kissat lacks incremental). Pin a commit; document provenance; keep CaDiCaL as the always-available fallback
   engine so any regression is a one-env-var revert.
6. **Determinism scope** — kissat_extras is deterministic because it is single-threaded. It does NOT give
   deterministic *parallel* solving; the parallel/speed role stays with gimsatul (nondeterministic). If
   deterministic-parallel is ever required, that's DPS-Kissat / MergeSat territory, out of scope here.

## 6. Validation plan (before flipping the default)
1. **Solution-equivalence:** for a suite of sizes/seeds, assert kissat and CaDiCaL both return SAT and that
   the decoded mappings satisfy all constraints (they need not be the *same* mapping, but must be valid and
   must respect host-cap/preferred). Confirms no soundness regression from the engine swap or the pool/warm
   deletion.
2. **Determinism:** run kissat engine twice per instance; assert identical decoded mapping (single-thread →
   must be bit-identical).
3. **Incremental enumeration parity:** enumerate K mappings with kissat vs the current path; assert the same
   *set* of distinct mappings is reachable (modulo order) and measure wall-time.
4. **Perf gate:** kissat must be ≥ CaDiCaL on the SC24 sweep sizes (expected from §1) before default flip.

## 7. Phased rollout
- **P1** Vendor kissat_extras + CMake target (`TT_METAL_FABRIC_KISSAT`). No behavior change.
- **P2** Engine interface + `KissatEngine` behind `TT_TOPO_SAT_ENGINE=kissat` (CaDiCaL default). Option/
  terminator mapping; `kissat_protect` for enumeration vars.
- **P3** Validation suite (§6). Fix incremental-trail gaps if any.
- **P4** Delete pool + warm-start code (§4) once kissat path is the deterministic default.
- **P5** Flip default to kissat for the deterministic role; keep `TT_TOPO_SAT_ENGINE=cadical` escape hatch.

## 8. Kissat features/improvements we can adopt NOW (beyond just "faster")
Swapping in kissat unlocks capabilities CaDiCaL doesn't expose. These are opt-in tuning, not required for the
swap, but are the reason kissat is more than a drop-in speedup:

- **A. `sat` configuration preset** — `kissat_set_configuration(solver, "sat")` applies a SAT-tuned profile
  (`--target=2 --restartint=...`). Our min-host solve is SATISFIABLE-dominated (we're finding *valid*
  placements, not proving UNSAT until enumeration exhausts), so the SAT profile should directly speed up the
  placement solves. No CaDiCaL equivalent preset.
- **B. Integrated local search / "walk" (SLS)** — kissat has a built-in stochastic local search
  (`walkinitially`, `walkeffort`, `walkreuse`, `walkweighted`). Running an initial walk can find a satisfying
  assignment (i.e. a valid placement) fast on SAT instances — useful for the FIRST solve / base embedding.
- **C. Explicit seed control (`seed`)** — one clean knob for BOTH requirements: fix the seed → deterministic
  reproducible mapping; vary the seed → diverse solutions. This unifies the deterministic path and the
  parallel-diverse enumeration idea (different seeds → different placements) under one mechanism.
- **D. Target phases (`target=2`)** — aggressive phase-saving toward the best model found so far; accelerates
  convergence on SAT instances (part of the `sat` preset, also settable directly).
- **E. Chronological backtracking (`chrono`/`chronolevels`)** — kissat supports chronological backtracking,
  the enabling primitive for blocking-clause-FREE AllSAT enumeration (Biere et al., TabularAllSAT / disjoint
  projected enumeration). This is the principled attack on our enumeration "cliff": no blocking clauses ⇒ no
  solve-2 escalation and no unit-propagation degradation. Not a quick win, but kissat gives us the primitive
  CaDiCaL doesn't; a future enumeration engine could exploit it.
- **F. Fine-grained inprocessing control** — per-technique knobs (`eliminate*`, `sweep*`, `substitute*`).
  Two uses: (1) tune inprocessing for the min-host instance family; (2) for incremental enumeration, control
  variable elimination — we learned elimination interacts with cross-solve soundness (the preserve-vars
  finding in HARVEST_FINDINGS.md), and with `kissat_protect` + these knobs we get finer control over what
  survives between solves than CaDiCaL offers.
- **G. `kissat_protect`/`unprotect`** — protect the mapping/blocking variables from elimination across
  incremental solves (also listed in §3.5) — keeps incremental blocking efficient and sound.

Recommended first adoptions (cheap, likely wins): **A (sat config)** + **C (seed control)** — one speeds the
SAT-dominated solves, the other cleanly serves determinism *and* the diversity/portfolio path. B/D follow if
first-solve latency matters. E is a longer-horizon research lever for the enumeration cliff.

## 9. Explicitly NOT in this proposal
- Replacing gimsatul (the parallel/speed engine) — unchanged.
- Deterministic *parallel* solving (DPS-Kissat/MergeSat) — separate investigation.
- MaxSAT / soft-preferred — not supported by either engine natively.
- Mallob adoption — fast+parallel+incremental but nondeterministic; fails the deterministic requirement.
