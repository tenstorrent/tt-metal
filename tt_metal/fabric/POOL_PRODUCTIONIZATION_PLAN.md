# Productionization plan: gimsatul + distilled clause pool for the inter-mesh min-host solve

Companion to `POOL_EXPERIMENT_PLAN.md` (the benchmark that validated this). This plan turns the
env-gated experiment (`TT_TOPO_SAT_GIMSATUL` / `GIM_EVERY` / `POOL`) into a shippable auto-mapper path.

## What we're shipping (and why)
The benchmark (6 sizes x 3 modes x 3 seeds, latest base) showed:
- **B = gimsatul + distilled pool** is never worse than cold gimsatul and clearly better in the
  tractable tail (96: ~2.3x; only mode to complete 128 `-n 5`). It is the enumerator to ship.
- **112-128 (the fill cliff) are run-variance dominated** for every mode -> ship B *inside a
  best-of-N race*, not as a single attempt.
- The clause-share portfolio (D) is not viable alone on min-host; but its `Learner`/`ClauseSharingPool`
  export machinery IS the pool's producer -> keep it as the export half, drop it as a standalone mode.

## The architecture, in one picture
```
CaDiCaL driver (owns the session: encode, cap, preferred, pinnings, symmetry, blocking, decode)
  │  every solve() attaches a Learner  ──► DistilledClausePool  (short learned clauses + fixed units,
  │                                          deduped, cap-tagged, size-bounded; per-process, in-memory)
  │
  ├─ prime / each enumeration step needs a heavy solve:
  │     race{ gimsatul(threads=T1) , gimsatul(threads=T2) , CaDiCaL-cold-cap }   ◄── best-of-N
  │        each gimsatul snapshot = base CNF + permanent cap units + blocking clauses + POOL
  │     first SAT wins; parse v-lines back to driver vars; kill the losers
  │
  └─ add blocking clause -> next step (pool now richer) -> repeat
```

## Work breakdown

### Story P1 — Solidify the pool + gimsatul path (the experiment code, hardened)  [M]
The prototype already implements: DIMACS tee, `gimsatul_solve()`, GIM_EVERY per-step,
`DistilledClausePool` (Learner export, fixed-unit harvest, cap-tag filter, shortest-first inject,
`POOL_MAX_LITS`/`POOL_MAX_CLAUSES`), and cap-permanence in MIN_MODE=3. Productionize:
- **gimsatul as a real dependency**: vendor/build via CPM (or a checked-in build script + CI cache),
  not a hand-run `build_gimsatul.sh`. Detect at runtime; if absent, fall back to CaDiCaL cleanly.
- **Subprocess hygiene**: temp-file lifecycle (unique per rank/step, cleaned on exit/timeout), timeout
  + SIGKILL, stderr capture, non-zero-exit -> fallback (never a hard failure).
- **Config surface**: collapse the ~8 env knobs into one policy enum + a small struct
  (`FabricSolverConfig{ engine=auto|cadical|gimsatul_race, pool=on/off, race_threads=[8,32],
  pool_max_lits, pool_max_clauses }`), env-overridable but with a sane default. `auto` = the race below.
- **Determinism note**: gimsatul is nondeterministic across runs; document that solution *set* is stable
  (dedup by distinct host-set) but *order/time* is not. Keep a `SEED` for the CaDiCaL side.

### Story P2 — Best-of-N race wrapper  [M, highest cliff-value]
Replace "call gimsatul once" with "launch {gimsatul t8, gimsatul t32, CaDiCaL cold-cap} on the same
snapshot, take first-to-finish, cancel the rest." Covers instance-dependence (gimsatul lost at 96 pre-pool;
CaDiCaL wins some 128 runs) and the 112/128 variance (B and A each won different seeds).
- Race harness: fork the K attempts, poll for the first `s SATISFIABLE`, SIGKILL siblings, parse winner.
- The pool feeds every gimsatul attempt; the CaDiCaL attempt stays warm/incremental across steps.
- Thread counts configurable; default `[8, 32]` (the report's non-monotonic sweet spots) + 1 CaDiCaL.

### Story P3 — Pool enrichment for the pure-gimsatul path  [S, unlocks more pool value]
Observed limitation: in *pure* GIM_EVERY the driver CaDiCaL never searches, so the pool only gains
root-fixed units between steps (1/step). Two cheap enrichers:
- **Always run the CaDiCaL cold-cap attempt in the race (P2) with its Learner on** -> its search fills the
  pool every step even when a gimsatul attempt wins the *solve*. (This is why P2+P3 compose: the race's
  CaDiCaL arm is also the pool's producer.)
- **Optional DRAT mining** of gimsatul's own short learned clauses (filter proof-addition lines, units +
  binaries): sound to re-inject (entailed by formula+baked-cap), but heavy (GB traces) -> gate off by
  default, enable only for a "hard tail" escalation.

### Story P4 — Correctness + regression guards  [S, required]
- **Cap soundness test**: assert a clause tagged at cap K is only injected when run-cap <= K
  (unit test on `DistilledClausePool`), so descent-mode reuse can never unsound a looser solve.
- **Family gate**: keep the >5000-var gate (intra-mesh formulas must never share a pool with inter-mesh).
- **Model-equivalence test**: for a fixed small MGD, assert the *set* of distinct host-sets is identical
  across {cadical, gimsatul, gimsatul+pool} — proves the pool changes speed, not answers.
- **Fallback test**: run with gimsatul binary hidden -> must silently fall back to CaDiCaL cold-cap.

### Story P5 — Wire into the production enumerator + gate rollout  [S]
- Route `generate_rank_bindings` / the sweep driver through `FabricSolverConfig{engine=auto}` so the
  race+pool is the default for min-host enumeration; keep a `engine=cadical` escape hatch.
- **This is the lever to close #51629**: the 96/112/128/144-stage rings disabled in the SC24 sweep
  (PR #54682) can be re-enabled once the auto path lands, since it makes them tractable.

## Acceptance metrics (from the benchmark)
- 96 `-n 5` completes < ~6 min every seed. ✅ (B: 174-348s)
- 128 `-n 5` returns >= 3 distinct solutions, race completes 5 on a good attempt. ✅ (B: 3-5; race raises floor)
- 144 solved + exhaustively enumerated < 5 min. ✅ (B/A: 132-252s)
- No regression on preferred / pinnings / cap / dedup (all hard-clause encoded). ✅ by construction.
- gimsatul absent -> no failure, CaDiCaL fallback. (P1/P4)

## Sequencing
P1 -> P2 (these two are the ship) -> P4 (guards, land with P1/P2) -> P5 (default + #51629) -> P3 (enrichment,
follow-up) . Estimated: P1+P2+P4 ~ one focused PR; P5 a small follow-up; P3 optional research-y.

### Story P6 — In-process gimsatul as a linked library (replace subprocess + DIMACS tee)  [M, preferred integration]
Goal per direction: drive gimsatul *directly from our C++*, not via a DIMACS file + CLI subprocess.
gimsatul's core is already library-shaped (from `gimsatul.c`):
```
new_ruler(num_vars, &options)           // ruler = owner of the shared clause DB + N worker rings
  → add clauses directly                // unit: assign_ruler_unit(ruler,u)
                                        // binary: new_ruler_binary_clause(ruler,l0,l1)
                                        // n-ary: PUSH(ruler->clauses, new_large_clause(size,lits,false,0))
  → solve_rings(ruler)  → winner ring   // winner->status: 10=SAT, 20=UNSAT
  → extend_witness(winner) → signed char* witness[var]   // read the model in-process
```
Literal encoding is gimsatul's `unsigned = 2*var_index + sign_bit`; we convert our DIMACS ±lits once.
Design:
- **Vendor gimsatul source, build it as a static lib** (CPM or in-tree third_party), and add ONE small
  in-tree shim (`libgimsatul_shim.c`, compiled into that lib) exposing a stable C API
  (`gim_new(nvars,threads) / gim_add_clause(h, int* lits, n) / gim_solve(h) / gim_val(h,var) / gim_free(h)`).
  A shim is needed because the internal add primitives / `verbosity` / options defaults may be `static`
  or global — the spike (below) determines the exact footprint.
- **Pool + blocking clauses feed through `gim_add_clause` directly** — no DIMACS serialization at all;
  the `DistilledClausePool` becomes an in-memory producer straight into the ruler.
- **Per-solve isolation**: each heavy solve builds a fresh ruler (gimsatul is one-shot by design — a
  ruler solves once); repeated enumeration = new ruler per step, but now via API calls, not fork+file.
  Confirm there is no process-global state that forbids sequential `new_ruler/solve_rings/free` in one
  process (the spike checks this; if globals exist — verbosity, signal handlers — the shim resets them).
- **Fallback unchanged**: if the lib is unavailable/misbuilt, fall back to CaDiCaL cold-cap.
Payoff vs subprocess: removes DIMACS write/parse (nontrivial at 100k+ clauses), removes fork/exec and
temp-file lifecycle, removes stdout parsing, and lets the pool stream in as clauses. Cost: build-system
integration + the shim + a link footprint.
**Spike result: GREEN — validated end-to-end.** A standalone probe built the ruler directly via the C API
(no DIMACS), solved SAT + UNSAT + a repeated 3rd solve in one process, model verified correct. Findings:
- **No fork of gimsatul needed.** Every required symbol is non-static/linkable: `initialize_options`,
  `new_ruler`, `assign_ruler_unit`, `new_ruler_binary_clause`, `new_large_clause`, `simplify_ruler`,
  `clone_rings`, `solve_rings`, `extend_witness`, `detach_and_delete_rings`, `delete_ruler`.
- **Exact API sequence** (mirrors `gimsatul.c:main`): `initialize_options(&opts)` → `opts.threads=T` →
  `new_ruler(nvars,&opts)` → add clauses (dispatch by size as in parse.c) → `simplify_ruler` →
  `clone_rings` → `solve_rings` → `winner->status` (10 SAT / 20 UNSAT) → `extend_witness(winner)` →
  read `witness[2*var_idx]` (+1 true / −1 false) → `detach_and_delete_rings` + `delete_ruler`.
- **Packaging**: `libgimsatul.a` = all non-`main` objects (53/58 pulled; linker drops parse/catch/logging/
  statistics/types). A ~150-line shim (`gim_new/gim_add_clause/gim_solve/gim_val/gim_free` + a 10-line
  header declaring the 11 symbols and `extern int verbosity`). **No gimsatul source changes.**
- **Re-entrancy**: all state lives in the heap `struct ruler`; sequential `new_ruler/solve/free` per
  enumeration step is safe (proven). `solve_rings` asserts single-solve-per-ruler; distinct rulers are
  independent. `threads=1` runs in the main thread (no pthread); `threads>1` is per-ruler internal.
- **THE footgun to bake into the build**: the shim TU **must** compile with `-DNDEBUG` (matching the
  prebuilt objects) — `struct ruler` has an `#ifndef NDEBUG` field, so a mismatch silently corrupts the
  ABI. Also carry `-O3`. Set `verbosity=-1` to silence solver stdout.
Effort: ~half a day. **P6 supersedes P1's subprocess mechanics** — ship the in-process lib as the
integration; keep a DIMACS-dump debug path only for repro.

## Explicitly NOT doing
- Standalone clause-share portfolio as a min-host mode (loses; kept only as the pool's export path).
- Making gimsatul incremental (architecturally impossible — the pool IS the workaround).
- Cube-and-conquer / Mallob (heavier; only if a hard `-n 20`-at-128 requirement survives P2's race).
