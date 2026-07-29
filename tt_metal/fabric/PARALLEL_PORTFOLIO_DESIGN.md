# Parallel seed portfolio — design (Goal-1 base-embedding speedup)  [NOT IMPLEMENTED YET]

## Motivation / data
Base-embedding SAT solve time swings ~6x by CaDiCaL seed (2x4-128: 33s..205s, median 66s). A seed does NOT
reliably transfer across instances, so we can't just hardcode one. The robust win is a **parallel portfolio**:
race k seeds, take the first to SAT → wall ~= fastest seed.

Naive realization (launch k separate 36-rank producers) FAILS: K-sweep wall-to-first on 2x4-128 —
K=1:38s, K=2:66s, K=3:75s, K=4:82s, K=6:141s. It gets *worse* with K because each producer carries 35 idle
ranks that **busy-poll** their barrier, so K*36 ranks oversubscribe 64 cores (K=6 -> load 151).

## Key finding (generate_rank_bindings.cpp)
- **Only rank 0 solves** — `main()` line ~574: `if (current_rank == 0) { run_topology_mapping / enumerate }`.
- Ranks 1..35 do discovery + `gather_mock_cluster_desc_paths` (ends in a barrier), then wait at
  `context->barrier()` (line ~773) while rank 0 solves. Those 35 ranks **busy-poll** (spin) and waste ~35 cores
  per producer. In ONE producer: 36 cores busy (35 spin + 1 solve), ~28 cores idle-free.

=> Exploit the ranks/cores we already allocated instead of spawning more producers.

## Design A — per-rank MPI portfolio ("one solver per rank")  [preferred for full core use]
1. Rank 0 assembles the inter-mesh graph + constraints (as today).
2. **Broadcast** the graph+constraints to all ranks (serialize; each rank re-encodes locally — encode is ~0.1s).
3. Each rank runs the SAT solve with a distinct seed (`seed = base + rank_id`) — optionally seed x fastsat.
4. **First-to-SAT wins:** each solver's CaDiCaL `terminate()` callback polls a non-blocking MPI signal
   (Iprobe/Test on a "someone finished" tag). The finisher broadcasts done + its model index; losers abort.
5. Rank 0 (or the winning rank) decodes + writes the solution; all ranks meet the final barrier.
- Pros: 36-way portfolio, all 36 cores solving, zero oversubscription, no spinning. Matches "one per rank".
- Cons: MPI plumbing (broadcast of problem, non-blocking first-to-finish + cancel). Must ensure determinism of
  the chosen solution (pick lowest rank on ties) so output is reproducible.

## Design B — rank-0 thread portfolio ("rank 0 uses the idle CPUs")  [simplest to prototype]
1. Rank 0 keeps the assembled problem; spawn N std::threads, each its own CaDiCaL instance built from the same
   encoding, distinct seed (+ fastsat).
2. Shared `std::atomic<bool> done`; each solver's `terminate()` callback returns true once `done` is set.
   First thread to reach SAT sets `done` + stores its model, joins the rest.
- Pros: no MPI changes; threads share memory (encode once, or cheap re-encode per thread). Easiest win.
- Cons: limited to FREE cores (~28) unless the 35 idle ranks are quieted (barrier sleep) — otherwise rank-0
  threads contend with the spinning ranks. Note: OpenMPI `--mca mpi_yield_when_idle` was REJECTED here; need the
  correct progress-yield knob for this build, or a custom sleep-wait barrier, to free all 63 non-solving cores.

## fastsat (config) — jotted, NOT implemented in the portfolio yet
`TT_TOPO_SAT_FASTSAT=1` sets CaDiCaL `target=2, phase=1` — a robust ~3x single-solve win (2x4-128: 63s vs 205s),
and it compounds on a good seed (seed99+fastsat = 21s). PLAN: enable fastsat in EVERY portfolio worker so the
race is over (seed x fastsat) candidates, not seed alone. It's a pure config change (no parallelism), safe to
turn on independently.

## Recommendation / order
1. Prototype **Design B** first (fastest to build; validates the portfolio win on free cores).
2. To free ALL cores for B, find the correct MPI progress-yield option for this OpenMPI/prterun build (the
   `mpi_yield_when_idle` name was wrong) OR replace the post-solve barrier with a sleep-poll wait.
3. Then **Design A** for the full 36-way, all-cores portfolio (the clean production form).
4. Turn on **fastsat** in all workers.

Expected: wall ~= fastest of N (seed x fastsat) candidates ~= low tens of seconds on 2x4-128 (vs 205s baseline),
scaling to the larger cases — without depending on a transferable seed.

## Not addressed here
- Whether a single seed transfers across stage counts: assume NO (untested, unreliable) — that's why we need the
  portfolio.
- Goal 2 (host minimization) is parked separately (see HOST_MINIMIZATION_INVESTIGATION.md).

---

## Design B — IMPLEMENTED & VALIDATED (env: TT_TOPO_SAT_PORTFOLIO=N, +TT_TOPO_SAT_FASTSAT=1)
Rank 0 spawns N threads, each its own CaDiCaL solver + independent encoding (encode serialized under a mutex,
~0.1s) + distinct seed; first-to-SAT sets an atomic that all workers' terminator polls (added a cancel-flag
hook to HeartbeatTerminator). One 36-rank producer + N threads = 36+N cores (<= 64 => no oversubscription).

Results on 2x4-128 base embedding (baseline single default seed = 205s):
| config | intermesh solve | speedup |
|---|---|---|
| baseline (1 seed) | 205s | 1x |
| 8-way | 34s | 6x |
| 8-way + fastsat | 22.5s | 9x |
| 16-way | 41s | 5x (thread overhead vs 8-way) |
| 16-way + fastsat | 13.7s | 15x |

Findings:
- WORKS and is ROBUST (no dependence on a transferable seed; the portfolio finds a fast one every time).
- Contrast the multi-PROCESS portfolio (139s, throttled by K*36 idle-rank busy-poll) — Design B avoids that by
  reusing the single producer's free cores instead of launching new producers.
- fastsat compounds strongly (16-way+fastsat=13.7s best). More threads alone is NOT strictly better
  (16-way 41s > 8-way 34s: added thread/memory-bandwidth contention) — sweet spot ~8-16 workers WITH fastsat.
- Correctness: hosts=36 (valid base embedding); winner mapping decoded from the winning thread's model.

Recommended default for the base embedding: ~12-16 worker portfolio + fastsat. Next: (a) verify on 144, (b)
consider Design A (per-rank, all 36 cores) only if we need >~16-way; (c) wire portfolio into the real solve path
(not just base embedding / NO_MINHOST) once we return to Goal 2.
