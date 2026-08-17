# cgtceq bench bring-up debug — narrative + measured constants

Date: 2026-08-16. Hardware: single Blackhole p150a, exclusive access
(`flock /tmp/tt-device.lock`). Branch `nkapre/sorting`. Files touched:
`tests/sources/cgtceq_perf.cpp`, `tests/python_tests/test_cgtceq_perf.py`,
`tests/sources/CGTCEQ_RUNBOOK.md` (all campaign-new; no tracked/shipping file
edited; `cgtceq_analysis.py` needed no change).

**End state: 51/51 GREEN** (full `-m perf` consumer run, 0 failed, 0 errors,
0 TENSIX TIMED OUT), every rendezvous XOR-checksum and every bisect row
field-exact against the sign-magnitude goldens, flags==0 everywhere.
7 consumer iterations used (budget 25).

---

## Starting state

`/tmp/logs/cgtceq_consumer.log`: 12 passed / 39 failed / 1 error.
- PASS: all stream-additivity arms, `rate-f0-s0-i512`, keymap anchors.
- FAIL: every `rendezvous` arm, every `bisect` cell, AND `rate-f0-s0-i2048`
  — all `TENSIX TIMED OUT ... waited 2 seconds for Math, Unpacker`.
- ERROR: session-end `PerfSchemaError` (2 column schemas in one CSV).

## Method / iteration log

**Static phase (no device).** Read the kernel, driver, and the LLK functions
the fill INIT calls (`llk_unpack_A.h`, `cunpack_common.h`, `cmath_common.h`,
`llk_math_common.h`, `llk_math_eltwise_unary_datacopy.h`, `counters.h`,
`profiler.h`, harness `test_config.py` / `perf/core.py`). Key structural
facts established:

- The unpack-to-dest fill handshake is mailbox (math→unpack dest index) +
  `semaphore::UNPACK_TO_DEST` (unpack posts tile-done, math gets).
- The hang-detector lists per-TRISC completion mailboxes: "Math, Unpacker"
  means Pack COMPLETED — so the wedge had to leave pack's path clean.
- `rate` (no fill) at i512 passed while structurally-identical i2048 failed
  → strong contamination signal: no device reset happens between consumer
  tests, so one wedge poisons everything after it. (`sfpu_count_above`
  proved the identical free-wrapping walk at i2048, ruling out a real i2048
  bug.)
- `_llk_math_pack_sync_init_` (BH `llk_math_common.h:140`) begins with
  `tensix_sync(); while (semaphore_read(semaphore::MATH_PACK) > 0) {}` —
  a drain spin that only pack can satisfy.

**Root cause #1 — MATH_PACK semaphore leak (the hang).** Fill-arm math ended
with `_llk_math_dest_section_done_` → `set_math_semaphores()` → posts
`MATH_PACK`. In fill arms the pack thread does nothing and never consumes
it. `PerfConfig.run` re-runs the kernel `run_count`(=5) times with no
semaphore reset, so run 2's `_llk_math_pack_sync_init_` spins forever on the
leaked token: Math wedges in INIT (RISC spin), Unpacker wedges at the fill's
`mailbox_read` (math never sends the dest index), Pack completes → exactly
"waited 2 seconds for Math, Unpacker". The token is device-persistent, so
every subsequent test (any arm calling `_llk_math_pack_sync_init_`, i.e. all
of them) also wedged — explaining `rate-i2048` and all bisect cells.
*Fix:* fill arms no longer call `_llk_math_dest_section_done_`; the next
run's `pack_sync_init` (SEMINIT + `reset_dest_offset_id` + StartZero)
re-establishes all the state the call would have flipped.
(An additional latent collision was documented along the way: under
`--enable-perf-counters` builds, the perf-zone EXIT barrier uses
`PERF_EXIT_SEM == semaphore::UNPACK_TO_DEST` — the same semaphore as the
fill handshake — and under MATH_ISOLATE the idle unpack/pack exit-spinners
can steal fill posts. Not the failure mode of these profiler-build runs,
and moot for the runbook flow, which forbids `--enable-perf-counters`.)

**Iteration 1** (post-fix, single arm `rendezvous-f0-s0-i512`, fresh reset):
hang GONE — all 15 executions (3 run types × 5 runs) complete; now a clean
correctness failure: device count = 2048 every segment vs golden 2047/2046
(`last=2048 != 2046`, checksum self-consistently 0).

**Iterations 2–4 (probes).** 2048 = every datum "above" the (negative)
threshold. SFPGT is strict per the ISA functional model
(`SignMagIsSmaller`, SFPGT.md), so ">= semantics" was ruled out; the two
live hypotheses were "compare broken" vs "walk reads zeros/junk". Added an
out-of-zone diag probe (diag[9..15]): raw MMIO Dst words + twin-style
per-tile SFPU counts (`sfpu_start(t)` + 16-pass walk) against `-0.0` and
`THR_BITS`. Also re-ran the proven correctness twin
(`test_sfpu_count_above`: all_above_four_tiles / positional_ramp) on this
branch — PASSED, proving fill+walk+macro machinery is healthy today.
Probe result: cnt_t0_neg0=1024, cnt_t1_neg0=1024(exp 0), cnt_2t=2048,
cnt_t1_thr=1024(exp 1023) — every value consistent with the walk reading
**zeros/junk everywhere** (zeros > any negative threshold), i.e. the
stimulus never reached L1.

**Root cause #2 — PerfConfig never writes stimuli.** `TestConfig.run()`
writes `variant_stimuli` to L1 (`test_config.py:1661`); `PerfConfig.run()`
(`helpers/perf/core.py`) overrides run() and never does — its loop is only
`write_runtimes_to_L1()` + `run_elf_files()`. Perf tests are timing-only by
design; cgtceq is the first perf bench whose kernel READS its stimulus.
*Fix:* driver-side `_write_stimuli()` calls
`configuration.variant_stimuli.write(TestConfig.TENSIX_LOCATION)` before
every device run (guarded out of `--compile-producer`). L1 buffers persist
across the perf run loop, so one write per test suffices.

**Iteration 5:** single arm PASSES — probes exact (1024 / 0 / 1024 / 1023),
`last=2046`, checksum match, flags=0, and the MMIO window verified exact
end-to-end (raw probe r0w0 = stim[0]; r64w0 = 0xBE2D0000 = two's-complement
view of sign-magnitude 0xC1D30000, i.e. the fmt=INT32 conversion behaving
per Dst.md). R0_WORD=0, the SyncHalf base, and the MMIO write path (S2) —
all three pre-declared levers — were correct as designed; none needed.

**Root cause #3 — CSV schema gates (the 1 error).** (a) rendezvous/bisect
ran `MATH_ISOLATE` only while the stream tests ran the triple → two column
sets in one CSV. Fixed by giving all tests the same
`[L1_TO_L1, UNPACK_ISOLATE, MATH_ISOLATE]` triple (fill arms ignore
PERF_RUN_TYPE, so the extra types are redundant-but-harmless
re-measurements). (b) Iteration 6 (full suite): 51 passed but the gate still
fired — bisect's `run_count=1` emits no `std(...)` columns. Fixed by using
the shared `_RUN_COUNT`(=5).

**Iteration 7 (full suite):** `51 passed in 3.41s`, no error, no timeout.
Second consecutive full-suite green (iteration 6 was already 51/51 on the
tests themselves).

## What was verified (exactness gates all enforced in-test)

- 18 rendezvous arms (3 folds × 3 syncs × iters {512, 2048}): per-segment
  XOR checksum + last-count vs the host-simulated threshold automaton
  (a real data-dependent control loop: next threshold = f(count just read
  back by the RISC through memory-mapped Dst)), flags==0 (no bounded-poll
  timeout, no surviving sentinel — so the S2 MMIO-write path works).
- 57 bisect cells → 60 golden-checked rows (random/clustered/allequal ×
  4 seed-groups, kstraddle K∈{31,32,33}, ties, allneg, ±0/Inf/NaN/denormal
  specials, sync∈{0,1,2}): found threshold, Cgt, Ceq, decision count, exit
  mode all field-exact; invariant Cgt < K <= Cgt+Ceq (CERT) / Cgt==K
  (VALIDSET) holds on every row; decisions ≤ 17 always.

## MEASURED CONSTANTS (p150a, exact-count-certified)

### (i) Stream additivity (cyc per 32-elem vector, slope over tile_cnt {16,64})

| arm | MATH_ISOLATE | L1_TO_L1 | UNPACK_ISOLATE |
|---|---|---|---|
| stream_none (floor) | 0.000 | 4.079 | 3.938 |
| ctrl_load | 1.406 | 5.480 | 3.938 |
| ctrl_swap | 2.375 | 6.453 | 3.938 |
| stream_single | 2.438 | 6.499 | 3.938 |
| stream_dual | 4.531 | 8.594 | 3.938 |

- **Additivity holds:** single L1_TO_L1−floor = 2.420 vs MATH_ISOLATE 2.438;
  dual 4.515 vs 4.531. The count is fully additive on the fp32
  unpack_to_dest floor (the ~3.94 cyc/vec prior reproduces at 3.938).
- Priors confirmed with a caveat: the streamed controls carry a per-tile
  restart (~+0.4 cyc/vec amortized over 32 vectors), so ctrl_load reads
  1.406 rather than the loop-only 1.0; the pure-loop `rate` arm reads
  **2.0000 exactly** (the CountD1 sanity anchor). Deltas are restart-free
  because both sides of each subtraction carry the same restart.

### (ii) Rendezvous cost (cycles per data-dependent decision, slope over ITER {512,2048} × 64)

| fold \ sync | S0 tensix_sync | S1 t6_sem_post<WAIT_SFPU>+pc_buf | S2 Dst sentinel poll |
|---|---|---|---|
| **R0 full fold (RISC reads 1 word)** | **81.0** | 101.0 | 98.0 |
| R1 partial fold (reads 16) | 132.0 | 157.0 | 151.0 |
| R2 no fold (reads 64) | 756.0 | 770.0 | 773.0 |

- `rate` partner slope 2.0000 cyc/vec.
- The ≥25.1-cyc PassSync floor reproduces inside S0/R0's 81 (which adds the
  full fold, the threshold/accumulator restart, and one MMIO read on top of
  the bare store+sync ARM_PASS_SYNC measured).
- MMIO Dst reads cost ~10 cyc/word (R1−R0 ≈ +51 for +15 words; R2−R0 ≈
  +675 for +63) → folding on the SFPU before the RISC read is mandatory;
  "no fold, read raw lanes" is ~9× worse.
- The sentinel poll (S2) does NOT beat tensix_sync (98 vs 81): its polling
  loads are themselves ~10-cyc MMIO reads. tensix_sync (S0) is the best
  primitive measured; the prior 25–100 band is confirmed at the top end
  once a real fold+read+restart is attached.

### (iii) Bisection to the exact K-th threshold (1 row = 1 tile = 1024 words, K=32)

| distribution | rows | decisions p50 / p95 | cycles p50 / p95 |
|---|---|---|---|
| random (s0) | 12 | 14 / 17 | 2313 / 2960 |
| random (s1) | 3 | 16 / 17 | 2932 / 3291 |
| random (s2) | 3 | 16 / 17 | 2916 / 3279 |
| clustered | 12 | 17 / 17 | 2935 / 2953 |
| allequal | 12 | 17 / 17 | 2929 / 2948 |
| kstraddle K=31/32/33 | 3 ea | 17 / 17 | 2950 / 2968 |
| ties | 3 | 17 / 17 | 2920 / 2938 |
| allneg | 3 | 13 / 14 | 2107 / 2241 |
| specials (±0/Inf/NaN/denorm) | 3 | 15 / 15 | 2448 / 2478 |

≈165 cyc/decision ≈ per-tile count pass (~64–70) + rendezvous (~81) + loop
overhead — the (ii) matrix composes into (iii) as the model predicts.
Clustered/allequal pin at the worst case 17 (16 probes + dual cert) exactly
as key-space bisection predicts (1 bit/decision, adversarial-value-immune).

### Honesty guard (unchanged)

These are Gate-2 oracle constants only. At the bench's own numbers, one
certified per-tile threshold costs ~2.3–3.0k cycles/row — nothing here is a
claimed speedup over the incumbent bitonic path; the numbers close dep-map
open dep #1 and give the Gate-3 shootout an honest SFPU-side comparator.

## Residual notes

- The rendezvous XOR checksum is identically 0 for this stimulus (counts
  alternate 2047/2046 and pair-cancel). The `last`-count assert, the flags
  word, and the 60 field-exact bisect rows carry the real verification
  weight; a rolling hash would strengthen the rendezvous gate if anyone
  re-tightens it later.
- Kernel keeps out-of-zone diag probes (diag[9..15], rendezvous INIT):
  MMIO words + twin-style per-tile counts, printed by the driver — free
  insurance against stimuli/placement regressions.
- Generalizable harness lessons (also in the RUNBOOK):
  1. `PerfConfig.run()` never writes `variant_stimuli` — any perf kernel
     that READS its stimulus must write it from the test body.
  2. A fill-arm kernel whose pack thread is idle must not post MATH_PACK
     (`_llk_math_dest_section_done_`) — `run_count` reruns deadlock on the
     leaked token in `_llk_math_pack_sync_init_`, and the token poisons
     every later test on the un-reset device.
  3. One perf module = one CSV schema: identical `run_types` AND identical
     `run_count` (run_count=1 drops the std(...) columns) across all tests.
