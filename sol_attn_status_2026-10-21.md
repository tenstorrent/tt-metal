# Sol-Attn status, 2026-10-21: where the time goes and what gets 5 s to 2x

Reference dump of the numbers and reasoning discussed on 2026-10-21. Markers: **(m)** measured, **(e)** estimated
from measured parts, **(?)** never measured and the estimate is weak. All times op-only, median of 3, wall clock
with host syncs, 130 W Blackhole Galaxy 4x8, TP=4 / SP=8, random weights and inputs (`test_sol_bench_minimax_h3.py`).
Two machine changes happened during the measurements; dense re-measured the same within noise each time.

Decisions taken on 2026-10-21: bf8 K/V is **not** an option; the separate prefix-row recompute (§4) is **deferred**
until the kernel optimizations below are done (bench default `SOL_BENCH_PREFIX=off`; the code stays on the branch).

## 1. Time by component

"Sparse op" is the production chain kernel (`sol_mode="compact"`, 320-row query chunks at 5 s, 256 at 10 s / 15 s),
with the dense-prefix mask rule off (`SOL_BENCH_PREFIX=off`), i.e. without the straggler of §10 in
`sol_attn_kv_once_plan.md`. "Chunk density" is the fraction of (query chunk, 64-key block) pairs the op computes.

| shape, tau | chunk density | dense op | sparse op | prefix recompute (deferred) | sparse op vs dense | sparse + prefix vs dense |
|---|---|---|---|---|---|---|
| 5 s, tau 1  | 0.61 (m) | 8.3 (m)  | 7.5 to 8.5 (e; 8.2 measured with the rule on) | 1.0 (e) | ~1.0 to 1.1x | ~0.9x |
| 5 s, tau 2  | 0.16 (m) | 8.3 (m)  | 5.84 (m) | 1.0 (e) | 1.4x | 1.2x |
| 5 s, tau 3  | 0.06 (m) | 8.3 (m)  | 4.9 (m)  | 1.0 (e) | 1.7x | 1.4x |
| 10 s, tau 1 | 0.51 (m) | 29.5 (m) | 26 to 28 (e; 28.2 with the rule on) | 1.3 (e) | ~1.0x | ~1.0x |
| 10 s, tau 2 | 0.11 (m) | 29.5 (m) | 17.9 (m) | 1.3 (e) | 1.65x | 1.5x |
| 10 s, tau 3 | 0.03 (m) | 29.5 (m) | 10.3 (m) | 1.3 (e) | 2.9x | 2.5x |
| 15 s, tau 1 | 0.50 (m) | 66 (m)   | 52 to 57 (e; 56.6 with the rule on) | 1.8 (e) | ~1.2x | ~1.2x |
| 15 s, tau 2 | 0.10 (m) | 66 (m)   | 35.7 (m) | 1.8 (e) | 1.85x | 1.8x |
| 15 s, tau 3 | 0.02 (m) | 66 (m)   | 17.4 (m) | 1.8 (e) | 3.8x | 3.4x |

Other measured lines:

| item | 5 s | 10 s | 15 s | note |
|---|---|---|---|---|
| K/V all-gather alone (m) | 3.2 | 5.9 | 8.7 | runs inside the op, overlapped with compute; the op cannot end before it does |
| 64-row chunks + packing (`pack`), tau 1, rule on (m) | 19.3 | 61.8 | 149 | block density 0.23 / 0.19 / 0.18; 4 to 6x the visits of 320/256-row chunks |
| chain forwarding floor, drain-all, rule on (m) | 5.7 | 16.4 | 28.7 | 19 to 27 us per visit; includes the straggler, so an upper bound |
| pre-pass + its launches, outside the op (m, untraced wall) | +5.6 | +10.7 | +15.2 | sparse layer minus dense layer, same at every tau; under trace capture most of it should vanish (?) |
| dense attention layer (module) (m) | 12.1 | 36.9 | 77 | projections, norms, attention, output projection |

Block-level density at tau 1 is 0.23 / 0.19 / 0.18 (m, from the 64-row runs). The 320-row chunk keeps the union of
its five blocks' masks, which is how 0.23 becomes 0.61.

## 2. What the 2026-10-21 fine-zone profile says (`profile_sparse_logs/`, 10 s, tau 2, prefix off, zone level 3)

First real fine-zone capture of the sparse kernel (the four earlier "sparse" fine captures held the dense run's
data, because the sparse builds had failed). Complete file: 14,720 computed steps and 12,476 drained visits captured
on 3,520 cores (46 % drained, which is chunk density ~0.09 for 8-block pieces); 35 % of computed steps are narrow
(phase busy under 20 us, median 8.9 us).

| computed steps, by phase-busy time | RING_STEP wall | phase busy | what it is |
|---|---|---|---|
| p10 | 36 us | 8.7 us | narrow step, 1 kept block of 8 (2 QK sub-blocks of ~4.4 us) |
| p25 | 32 us | 13 us | narrow step, 2 to 3 blocks |
| p50 | 55 us | 55 us | full-width step (dense steps measure 65 to 73 us under the same zones) |
| p75 / p90 | 77 / 97 us | 77 / 96 us | wider than dense: the pooled-correction and bias steps |

Three conclusions and two caveats.

1. **Narrowing works.** A 1-block step is ~9 us of compute against ~56 for a full one under level-3 zones. The
   earlier claim "a step costs ~4.5 us per query tile row whatever its width" (plan §9) is wrong; it came from
   lockstep-contaminated visit times.
2. **Level-3 zones roughly double every step** (dense: 35 us at level 1, 65 to 73 at level 3), so the 28 ms op time
   of this run is a profiler artifact, not a regression. Per-visit *pacing* must come from level-1 profiles.
3. **Half of the computed steps on this bench are full-width even at tau 2**, i.e. the random-input mask is bimodal
   (a few query chunks keep everything, most keep nothing). That is a property of random weights; production
   masks are unknowable here. Any design must be judged on the 64-row block density, not on this bench's mask.
4. (caveat) The narrow steps' RING_STEP of 32 to 36 us against 9 to 14 us busy is waiting for the chain: the chain is
   paced by whichever member computes a full-width step, and by the forwarding.
5. (caveat) The run reported `density 0.72`, inconsistent with its own drain counts (40 % of visits drained, which
   is chunk density ~0.1, and with 17.9 ms at tau 2 before). Either the host readback of the mask is wrong on the
   new machine or something regressed. One plain bench run decides it:
   `SOL_BENCH_PREFIX=off SOL_TAU=2.0,3.0 ./run_sol_validation.sh noprefix` (expect 17.9 / 10.3 ms at 10 s).

## 3. Bottleneck at 5 s, in plain words

Required: 4.15 ms (2x of 8.3). Measured sparse op: 5.84 ms at tau 2, ~8 ms at tau 1.

1. **The all-gather is 3.2 of the 4.15 ms.** Everything else has to hide under it. bf8 is off the table, so the
   bytes stay; what can change is overlap and, if it is below link capacity, the gather kernel itself (unknown (?)).
2. **Every K/V piece is streamed through every core once per query chunk the core owns**, 19 to 27 us per visit,
   independent of sparsity. At 320-row chunks: ~208 visits per core, ~2 ms, under the gather. At 64-row chunks:
   ~1040 visits, ~10 ms, which is why `pack` lost 2.3x to dense.
3. **At 320-row chunks tau 1 keeps 61 % of the key blocks** (union over five 64-row blocks), so compute is ~dense.
   At 64-row chunks it would be 23 %, but bottleneck 2 forbids 64-row chunks today.
4. **The chain is paced by its slowest member per piece** (lockstep), so one full-width step on any of the 9 to 12
   members of a head chain costs every member a full-width step's time.

## 4. Fixes

### 4a. Making 64-row chunks pay: the optimizations required, with confidence

The goal is: one pass of each K/V piece per core per shard (removes 2), 64-row resident query chunks (removes 3),
each resident chunk computing only its own kept blocks straight out of the staged piece (keeps the narrow steps of
§2.1). This is the plan's §3 / §9 restructure of the chain kernel; nothing of it exists yet.

| # | change | where | confidence it does what it says | main uncertainty |
|---|---|---|---|---|
| A | Loop order shard -> piece -> resident chunk: stage each piece once, forward once; N accumulator sets (max, sum, output) live in L1 for the whole op; output written once per pass | reader + compute + writer (`ring_joint_reader.cpp`, `compute_streaming.hpp` `sdpa_ring_v2`, writer) | **high** that visits drop by the chunks-per-core factor (it is arithmetic: visits = pieces x shards x passes) | the per-visit forwarding cost stays 19 to 27 us (m); whether compute then overlaps it or adds to it is **not known** (today they add; the lockstep pacing of §2.4 suggests the reader waits on compute) |
| B | Per-resident-chunk narrow step without a per-chunk packed page: the step's K loop walks the chunk's kept blocks at their offsets inside the staged piece (block-run addressing of K^T columns and V rows) instead of a contiguous compacted range | `sdpa_inner_loop_step` tile addressing; the `packed_sources` / `PackedKVChunkPlan` run machinery that exists for the joint path is the model | **medium**: the step code is the hardest part of the kernel (sub-block geometry, DEST limits); the run-addressing precedent exists but for a different source layout | narrow-step cost at 64 rows is ~2 tile rows of the §2 numbers, i.e. ~2 to 2.5 us real for 1 to 2 blocks (e); never measured at 64 rows |
| C | L1 budget for ~10 resident 64-row chunks at 5 s: Q 16 KB + output 32 KB (fp32) + stats 4 KB each = ~0.5 MB; staged piece double-buffered 2 x 192 KB (384 keys, K and V); QK scratch ~48 KB; total ~1.0 MB of 1.5 | factory CB sizing | **medium-high**; the previous L1 estimate in the plan was ~10 % optimistic (plan §7), so keep that margin. At 10 s / 15 s (17 / 27 chunks per core, 512-key pieces) two to three passes are needed; visits drop 6x instead of 17x there | the real CB set of the streaming kernel has more buffers than this list (sub-block scratch, bias, plan CBs) |
| D | Pooled correction once per resident chunk on its first rotation, from the existing per-chunk pooled step | compute | **high** (exists per chunk today) | none |
| E | 64-row mask from the pre-pass and the Q1 chunking | `sol_prepass.py`, factory | **done** (`q_chunk=64` geometry and masks exist; the `pack` mode runs on them) | the pre-pass cost itself (§1, +5.6 ms wall at 5 s) is a separate item |
| F | Writer: no per-chunk accumulator save/restore between rotations; one output write per pass | writer | **high**; the save/restore traffic simply disappears | pass handling at 10 s / 15 s |
| G | Lockstep pacing with resident chunks: a piece is now paced by the slowest member's *sum* over its resident chunks | design consequence | **medium**: on a random-like mask the sum of ~10 narrow steps (~20 us) is about one forwarding (19 to 27 us); on a bimodal mask (§2.3) a member with dense chunks paces the chain | mask structure of the real model (unknowable here) |

Build order with a measurable checkpoint each: **M1** = A + C + D + F with full-width steps (no narrowing): proves the
visit reduction alone (5 s: 104 visits per core; op expected ~4 to 4.5 ms at any tau because steps are still full,
i.e. ~2x at tau 3 already if the three overlap). **M2** = B: the narrow per-chunk steps (the speedup at tau 1).
Effort: the plan's 4 to 6 weeks estimate for A to F stands; M1 is the first two of them.

**The same list, short form (as summarized on 2026-10-21):**

- **A, high confidence:** stream each K/V piece once per core per shard, with one accumulator set per resident
  query chunk. Visits drop by the chunks-per-core factor. Open question: whether compute then overlaps the
  forwarding or adds to it (today they add).
- **B, medium confidence:** each resident chunk computes only its own kept blocks straight out of the staged piece,
  by block-run addressing in the step. This is the hard part of the kernel. The narrow-step cost at 64 rows is ~2 to
  2.5 us, extrapolated from the 2026-10-21 profile, never measured at 64 rows.
- **C, medium-high:** L1 fits ~10 resident 64-row chunks at 5 s (~1.0 MB of 1.5); 10 s and 15 s need two to three
  passes.
- **D, F, high:** pooled step per resident chunk; one output write per pass.
- **E, done:** 64-row masks and chunking already exist.
- **G, medium:** lockstep pacing becomes the slowest member's sum over its resident chunks; fine on random-like
  masks, worse on bimodal ones.

Likely speedup at 5 s, tau 1: forwarding 2 to 2.8 ms and compute ~1.8 ms both fit under the 3.2 ms gather. If the
three overlap, 3.5 to 4 ms = 2.1 to 2.4x; if reader and compute add, ~4.5 ms = 1.8x. tau 2 and tau 3 land in the same
place. 10 s about 3.3 to 4.9x, 15 s about 4x. The gather sets the floor at 5 s; the only remaining levers there are a
faster gather kernel or the SP=4 / TP=8 split (§4c). M1 (A + C + D + F, full-width steps) already shows ~2x at tau 3
at 5 s if the overlap holds; M2 (B) is what tau 1 needs.

### 4b. Expected result of 4a (5 s, tau 1, block density 0.23)

| term | per core | value | basis |
|---|---|---|---|
| forwarding | 104 visits x 19 to 27 us | 2.0 to 2.8 ms | (m) per-visit cost, (e) count |
| compute | 104 visits x 10 resident chunks x P(chunk keeps >= 1 of the piece's 6 blocks) = 0.79 x ~2.2 us | ~1.8 ms | (e) from §2 narrow steps scaled to 2 tile rows (?) |
| all-gather | | 3.2 ms | (m) |
| op if the three overlap | max of the three + startup | **3.5 to 4 ms, 2.1 to 2.4x** | (e) |
| op if reader and compute add | 2.4 + 1.8 under 3.2, plus the last shard's tail | **~4.5 ms, 1.8x** | (e) |

tau 2 / tau 3 at 5 s land in the same place (compute is smaller, the floors are the same): ~3.4 to 4 ms, 2.1 to 2.4x.
10 s: 144 to 288 visits x 23 us = 3.3 to 6.6 ms, compute ~3 to 4 ms, gather 5.9: op ~6 to 9 ms, 3.3 to 4.9x.
15 s: ~4x. The 190 W production part speeds dense (compute-bound) more than this path (gather-bound at 5 s), so the
5 s ratio is an upper bound.

What this does **not** cover: the pre-pass (+5.6 ms wall at 5 s outside the op: run under trace or use the fused
pre-pass; must end near 0.3 ms for the *layer* to show 2x), and the prefix recompute (~1 ms at 5 s (e), deferred).

### 4c. The gather (bottleneck 1), without bf8

- Overlap is the only lever inside the op: with 4a the compute and forwarding fit under 3.2 ms, so the op is the
  gather plus the last shard's work (~0.3 ms).
- If the gather kernel is below link capacity (unknown (?); the broadcast_ring work measured the fabric), a faster
  gather kernel moves the floor. SP=4 / TP=8 would cut gathered bytes per chip 2.3x (102 vs 237 MB) and halve the
  ring, at the cost of larger TP collectives in the projections: a parallelization change, not a kernel one.

### 4c-bis. How the 5 s estimate moves from the 130 W part to the 190 W production part

The 130 W runs are not throttled in these ops (profiler cycle counts at 1350 MHz match the wall-clock op times),
so the 190 W part's gain is its higher sustained clock, call the ratio r (unknown here; two 30-second runs on a
190 W part, `dense` and `agather`, pin it). What scales with r: dense attention (compute-bound), the chain
forwarding (NoC) and the narrow steps (compute). What does not: the K/V all-gather if it is link-bound (3.2 ms at
5 s); if it is bound by the gather kernel's issue rate it scales partly.

| r | dense 5 s | sparse op after §4a (gather flat) | speedup |
|---|---|---|---|
| 1.0 (130 W) | 8.3 ms | 3.5 to 4 ms | 2.1 to 2.4x |
| 1.2 | 6.9 ms | ~3.5 ms (gather 3.2 + tail) | ~2.0x |
| 1.3 | 6.4 ms | ~3.4 ms | ~1.9x |

So at 190 W the gather is the whole floor and the 2x margin is gone unless the gather is shortened or hidden.
Levers that do not need bf8: (1) **gather prefetch**: issue the ring K/V all-gather as its own op on the CCL
sub-device right after the K/V projection, overlapped with QK-norm/RoPE and the pre-pass, and give the ring op a
"K/V already gathered" mode (factory: no AG workers; reader: every shard present). Hides whatever runs in
parallel with it, ~1 to 1.5 ms at 5 s once the pre-pass is traced. (2) **SP=4 / TP=8**: 2.3x fewer gathered bytes
per chip. (3) A faster gather kernel if `agather` on the 190 W part shows it is not link-bound.

### 4d. Prefix recompute (deferred) and pre-pass

`MiniMaxH3Attention._sol_prefix_dense_rows` (plan §10c) is written, host-validated (PCC 0.999995), unrun on device;
`sol_prefix_mode` = separate | fused | off. The bench default is `off` until the kernel work is done. The pre-pass
cost needs one traced run to know what is real.

## 4e. M1 implementation status (2026-10-21, written, not yet compiled or run)

`sol_mode="resident"`, selectable from the module (`sol_mode="resident"` picks the 64-row query chunk and the
chunk-form pre-pass mask) and the bench (`SOL_BENCH_MODE=resident`). What was built:

| piece | where | what it does |
|---|---|---|
| host | `ring_joint_sdpa_device_operation_types.hpp`, `..._device_operation.cpp`, `..._program_factory.cpp` | mode `resident` (a chunk mode; generated bias only; in the program hash); cb_q_in holds every resident chunk; the restore CBs (`cb_prev_out`, `cb_stats_in`, `cb_sum_in`) become an L1 FIFO `max_q_per_core + 1` deep; named args `sol_resident`, `sol_resident_slots` |
| compute | `compute_streaming.hpp` `sdpa_ring_resident`, `ring_joint_sdpa.cpp` | per ring iteration: pop one plan page per resident chunk; for each streamed piece, every chunk that keeps it runs the unchanged inner step with its state read from the FIFO front and written to the FIFO back (`save_out_cb`/`save_max_cb`/`cur.sum` redirects), Q at `q_base_tiles = i * chunk tiles`, K/V held until the last stepping chunk (`hold_kv`); chunks that do not keep it rotate their state (`sol_rotate_fifo_entry`); pooled entries step every chunk of the head with its own bias row; last iteration normalizes each chunk into `cb_out` in order |
| reader | `ring_joint_reader.cpp` | reads all resident Q chunks once; per head: chain-wide union over every chunk of every member; one plan page per chunk; streams each live piece once (heads interleaved per k); pooled K/V once per head, then one bias row per chunk |
| writer | `ring_joint_writer.cpp` | no save/restore, no signals; writes each chunk's output on the last iteration |
| tests | `test_sol_compact_minimax_h3.py::test_resident_matches_chunk`, `./run_sol_validation.sh resident` | resident vs chunk at q_chunk=64 (same mask, PCC > 0.999), then the 5 s bench at tau 1/2/3 |

Invariants the implementation relies on: chunk 0 of every shard is kept for every Q chunk (the reader forces it),
so every state is created on the first piece and the FIFO order never drifts; the FIFO CBs carry the step's
intermediate data format (static_assert); no joint K, no K split, no Q rotation, separate V entries.

Known costs left for M2 and later: a chunk that skips a piece pays a ~12-tile copy to rotate its state (~1 to
2 us; relevant at tau 3), steps are full-width (chunk semantics), and the chain is in lockstep (item G).

**First device run (2026-10-21 evening, `log_val_resident_*.txt`, after one compile fix in the writer):**

- Correctness: `test_resident_matches_chunk` passed (rows 1024, q_chunk 64, ~2 resident chunks per core, tau 2,
  chunk density 0.78): PCC(resident, chunk) 0.99931, PCC(resident, dense) 0.99595. The 0.9993 is lower than the
  ~0.9999 two implementations of the same arithmetic should give; the next run's `test_resident_dense_equivalence_q64`
  (tau -1e4: every piece steps every chunk, nothing rotates) separates the step path from the rotation path.
- 5 s bench, prefix off, three reps per tau, op-only: **11.9 / 4.3 / 11.5 ms** at tau 1 (chunk density 0.52),
  12.2 / 4.3 / 11.5 at tau 2 and 12.3 / 4.3 / 11.5 at tau 3 (both 0.117: at this bench's data the tau 2 and 3
  masks are the forced rules only, sink + band + chunk 0, which is 12.3 of 104 pieces). The fast reps are the
  gather floor plus ~1 ms, i.e. ~1.9x against dense 8.3 ms, at every tau; the slow reps alternate with them by
  call parity and are not explained. Same op, same mask, the state is per launch, so the suspects are something
  that alternates per call (the all-gather ping-pong buffer and semaphore sets) or the new machine. Step 12 of the
  script (`./run_sol_validation.sh resident2`) runs dense, compact and resident with 6 to 8 reps each on the same
  machine plus a level-1 profile of four resident reps (per-run kernel spans, `scratchpad/byrun.py`).

**Third run (later on 2026-10-21, `log_val_resident_check.txt`, resident now in the mode-parametrized tests at
64-row chunks):** 5 passed, 13 failed. What the per-chunk error maps say:
- rows 1024, every piece kept (tau -1e4): resident vs chunk 0.99975 overall, but the difference is localized: shard 2,
  query chunks 9 to 15 (chain members 4 to 7 of every head) at ~0.988, everything else 1.0000; in other cases shard 3
  chunks 0 to 6 (members 0 to 3). Chain-position-dependent and device-dependent.
- chunk mode itself is only 0.997 to 0.999 against dense at 64-row chunks (its 256-row gate is 0.9995), so part of
  the gap is in the shared q_chunk=64 path, not in the resident loop. `test_q64_base_determinism` (next run) checks
  whether dense and chunk at q64 are even deterministic.
- `test_determinism[resident]`: two identical runs differ by up to 6.8e-2. The pre-pass density readback also varies
  run to run (0.77692 / 0.77685: borderline threshold flips on device), which changes the mask, so the keepall cases
  are the clean comparison.
- rows 1184 (partial last K chunk and Q chunk): all three pairwise PCCs are NaN and the per-chunk map is NaN on shard
  0, i.e. some tensor has zero variance there; the next run attributes it (per-shard NaN count and std for dense,
  chunk and resident).
- The 5 s bench alternation (11.9 / 4.3 / 11.6 ms) reproduced a third time, identically.
The watcher run gave nothing: the full instrumentation overflows the kernel config buffer (71,600 of 70,656 B); the
step is now assert-only.

**Fourth run (`log_val_resident_diag.txt`): the base kernel is broken at 64-row query chunks, independent of
resident mode.**
- `test_base_determinism`: the DENSE ring op at q_chunk=64 differs between two identical runs by up to 0.36
  (outputs have std ~0.1); chunk mode by 0.25. Deterministic kernels do not do this: the 64-row path has a race.
- rows 1184 at q_chunk=64: dense, chunk and resident all return values of order 1e33 on seven of eight shards. The
  earlier "NaN on shard 0" was this: the only sane shard made the correlation undefined.
- The error maps confirm the sharing: at rows 512 the worst resident-vs-dense and chunk-vs-dense entries are the
  same cells (shard 2, head 5, PCC −0.2), i.e. the dense reference itself is wrong there.
- Mechanism (suspect): Sq_chunk_t = 2 tiles picks the QK sub-block {2, 4}, so `q_num_subblocks == 1`, which
  routes the step through the pack-to-unpack rendezvous for the in-place softmax row
  (`qktv_first_group_reads_inplace_row`, `compute_streaming.hpp` ~1800, "the matmul reads stale L1"). That
  rendezvous appears not to hold. The comment there already records a narrower earlier version of the bug.
- Consequence: every 64-row-chunk measurement on this branch (Q1, pack, resident) sat on a racy base. The clean
  comparison is at a chunk size where the base is deterministic.
- Resident residue that is not yet separable from the base race: at rows 1184 the third resident chunk on
  three-chunk cores (0.90) and the first two chain positions of every head (0.81 / 0.97) against chunk mode.

The four A/B logs of the same step:
- **Chains off** (`TT_SOL_NO_CHAIN=1`): at rows 512 resident equals chunk mode **bit for bit** (PCC 1.000000, from
  0.9946 with chains); at rows 1184 the two resident-specific patterns (third chunk of a three-chunk core at 0.90,
  first two cores of each head at 0.81 / 0.97) vanish. So the resident loop's arithmetic is right, and whatever is
  left involves the chain, or the base race's timing changing when every core reads DRAM. The rows 1024 dips stay
  with chains off, i.e. they are the base race.
- **Pooled off**: no change anywhere. The pooled entries and bias rows are not involved.
- **Reader trace**: on chip 2, cores (0..4, 0), every core reports the same per-iteration decisions and push counts
  (3 on the first iteration: two pieces plus the pooled chunk; 2 after), so there is no push-count phase fault
  between chain neighbours. It also shows this part has a 110-core worker grid (cores 0 to 3 hold three chunks at
  rows 1024), so the earlier chain-position maps were built on the wrong core count; fixed in the test.
- **Timing, six reps**: 11.9 / 4.3 / 12.3 / 4.3 / 12.2 / 4.3 ms. Strict alternation by call parity, resident only
  so far (dense and compact on this machine are in `resident2`).

**`qchunk_diag` result:** the dense ring op is non-deterministic at every chunk size (run-to-run 0.33 at q64, 0.94
at q128 and q256 on outputs of std ~0.1; 7.8e35 on most shards at rows 1184 at every size), and chains off changes
nothing (0.33 again). So it is not the 64-row sub-block path, and not the chain. My commits were reviewed line by
line against the pre-work commit f5800fbeb26: every host change is gated on `sol_resident` or on the
`TT_SOL_NO_CHAIN` switch, the only kernel-side edit outside `if constexpr (sol_resident)` is the `hold_kv` guard
(default false), so the dense program built here is the one that passed `test_determinism[chunk]` on the first
machine. Unknown whether the trigger is this part (110-core grid, different CCL column and chain lengths, firmware
19.13.2) or a latent base race that only this timing exposes; the pattern (plausible but wrong values in some
cells at rows 1024, unwritten-memory magnitudes at rows 1184) looks like the reader using gathered K/V or writing
output rows out of step with the fused all-gather. `test_base_determinism` now also logs where the two dense runs
differ per (shard, head, chunk) and the PCC of each dense run and of chunk mode against an fp32 host reference per
shard, so the next run says whether dense is correct at all here.

**Control (same test on HEAD, on the pre-work commit f5800fbeb26 and on 6e0329553, 27 commits older): identical.**
Same run-to-run dense difference (0.936), the same worst cells ((0.936, shard 0, head 9, chunk 0), (0.844, 7, 9, 0),
...), and against an fp32 torch reference the dense op scores 0.32 to 0.57 per shard at rows 1024 and 0.00 at rows
1184, on all three commits; chunk mode scores the same as dense. So the ring op's results are wrong on this
machine for every commit, while on the first machine gather and compact matched the torch reference at 0.98+ at
rows 1184. The likeliest mechanism is the fused all-gather's shard-to-slot (ring id) mapping or its arrival sync
behaving differently here (firmware bundle 19.13.2, 110-core grid). `_run` now logs, after every dense launch,
which torch K shard each device's gathered slot holds (`[ag] ...`; expect slot s = shard s, `?` at the local slot),
which is the direct check of that mechanism.

Next: `./run_sol_validation.sh qchunk_diag` (dense and chunk twice each at q 64 / 128 / 256, rows 1024 and 1184,
~5 min). If 128 is clean, resident moves to 128-row chunks with `SOL_RESIDENT_Q_CHUNK=128` (module and tests read
it; the union is then over two blocks, tau-1 chunk density ~0.35 instead of 0.23). Fixing the 64-row rendezvous is
a base-kernel task on its own.

**Diagnostics added for the next run (`./run_sol_validation.sh resident_diag`, ~8 min, five logs):**
- `resident_diag`: the geometry set now includes rows 512 (one resident chunk per core, no FIFO rotation, no second
  slot); each case logs three per-chunk maps (resident-vs-chunk, resident-vs-dense, chunk-vs-dense), per-shard NaN
  counts and standard deviation per tensor, and the mean PCC by chain position and by slot-within-core (the
  factory's flat split reproduced on the host). `test_q64_base_determinism` runs dense and chunk twice at q64.
- `resident_diag_nochain`: `TT_SOL_NO_CHAIN=1` (new factory switch, not hashed) builds no store-and-forward chains,
  every core reads its own K/V from DRAM. If the localized error disappears, it is the chain protocol in resident
  mode; if it stays, it is compute.
- `resident_diag_nopooled`: `TT_SOL_POOLED_OFF=1`. At keepall the pooled step contributes nothing, so a change here
  means the pooled K/V entries or the bias rows are at fault.
- `resident_diag_trace`: `TT_SOL_DEBUG=1` (new, a reader kernel define) prints one line per (core, ring iteration)
  on chips 0, 2, 3, cores (0..7, 0): kept bits, union bits, entries pushed, pooled flag. Chain neighbours that
  disagree on the push count would be a phase fault.
- `resident_diag_timing`: six resident reps at 5 s.

## 5. Runs, in order

1. `SOL_BENCH_PREFIX=off SOL_TAU=2.0,3.0 ./run_sol_validation.sh noprefix` on the new machine: settles the §2.5
   density anomaly and re-baselines (expect 5.84 / 17.9 / 35.7 ms at tau 2).
2. `SOL_BENCH_PREFIX=off SOL_TAU=2.0 ./run_sol_validation.sh profile_sparse` at level 1 (the default): per-visit
   pacing at tau 2 without zone inflation, for the overlap question in 4a-A.
3. M1 of 4a, then its bench.

## 6. 2026-10-21, later: the dense op on machine 2, and what the resident profile says

**Dense ring op is wrong on machine 2, and it is not precision.** The repo's own nightly
`tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_logical_tensor_accuracy[noncausal]`
fails here at PCC 0.60 (threshold 0.994, max abs error 27 on outputs bounded by ~4). Identical on HEAD, f5800fbeb26
and 6e0329553. Root cause (code, not hardware): every log from this machine says, for all 32 chips, "inconsistent
DRAM harvesting ... expects 0 units, but harvesting mask indicates 1 units"; machine 1's logs never do. With one
harvested bank, Blackhole's TRANSLATED DRAM coordinates depend on WHICH bank is harvested
(`blackhole_coordinate_manager.cpp` `fill_dram_noc0_translated_mapping`), tt-metal only checks the harvested count
matches, and every CCL writer puts its own chip's bank->NOC coordinates into the fabric packet
(`addrgen_api.h:103` states the assumption). So on a neighbour the page lands in the wrong bank or nowhere:
deterministic-but-wrong after the first call, never-written (~1e33) pages where the shard is not chunk-aligned,
no hangs, local ops fine. Fix (written by an agent, not yet built): the ring-attention all-gather writer gets the
neighbour's bank table as trailing runtime args and rewrites the (x,y) of each page
(`ring_attention_remote_dram_addrgen.hpp`, `make_remote_dram_bank_table_args` in the all-gather factory,
`get_dram_bank_to_noc_xy_table` in `tt_metal/impl/internal/cluster_noc_helpers.cpp`); flag 0 = identity on
machines where the tables agree. The generic all_gather_async / reduce_scatter (and so the pre-pass's pooled K/V
gathers) have the same bug on such a machine and are NOT fixed.

**Resident 5 s parity, from `profile_resident_logs/` (ops 100/170/240/310/380 = compile, slow, fast, slow, fast):**

| | slow call (170) | fast call (240) |
|---|---|---|
| device span | 9.3-10.7 ms | 3.1-3.6 ms |
| RING_STEP per core | 125 (every core) | 117-125 |
| busy in steps per core | 1.6 ms median | 1.8 ms median |
| last step on a core | 1.6-3.5 ms (one at 6.3) | 2.8-3.2 ms, paced in 7 waves |
| all-gather worker kernels end | 2.95-3.1 ms | 2.99 ms |
| reader/compute/writer kernels end | 5.4-10 ms, together per head-chain | 3.25 ms |

So the two parities do the same compute. In the slow calls compute finishes before the all-gather has delivered
the last slabs (steps are not paced by arrival), then every RISC on every core idles until a head-chain-specific
time between 5.4 and 10 ms. 125 steps per core = 8 shards x 5 pieces x 3 resident chunks + pooled: at k_chunk 256
and tau 2 every 64-row chunk keeps every 256-key piece, i.e. the mask removes nothing in resident mode; the fast
calls' 2x over dense is K/V-once dataflow alone. M2 (block-level narrow steps inside a piece) is where the
sparsity will have to come from.

**Next run, one command: `./run_sol_validation.sh resident3`** (a) the nightly dense test with the all-gather fix,
(b) `test_resident_call_parity`: four resident calls with fresh inputs each, each checked against chunk mode run
the same way, with per-call ms (a call that skipped an arrival wait is wrong on its own inputs), (c) the profiled
5 s resident bench with new wait zones: reader RD_SYNC / RD_RES_PLAN / RD_RES_PIECES / RD_CHAIN_RX / RD_CHAIN_TX /
RD_FETCH_K / RD_RES_POOLED / RD_DUMMY, writer WR_RES_OUT, compute RES_NORM / RES_ROT / RES_DRAIN / RES_DUMMY,
all-gather AG_SLAB_WAIT / AG_SIGNAL (the latter two emit under any profiled run). Logs: log_val_resident3_*.txt,
profile_resident3_logs/.

## 7. Correction (2026-10-21, resident3 profile with wait zones): the slow calls are the honest ones

The device profiler keeps only the first ~125 zones per RISC per core. In the slow resident calls every RISC hit
that cap within the first 1-4 ring iterations, so "compute done by 2 ms, then idle" in section 6 was an artifact:
those cores were busy the whole 9.5 ms. The fast calls fit under the cap. Reading both as truncated streams:

| per core, per ring iteration | slow call | fast call |
|---|---|---|
| pieces received over the chain (RD_CHAIN_RX) | ~11.5 of 13 | ~2.5 of 13 |
| (entry, chunk) steps | ~115 | ~12 |
| arrival wait (RD_SYNC) | 0 (slabs already there) | ~150 us (paced by the all-gather) |
| step cost | 12.1 us | 16.5 us (includes CB waits) |

So the two parities are two different masks: the slow calls run the mask the pre-pass reports (density 0.52-0.55 at
tau 2) and are compute-bound at 9.5 ms device / 12 ms host, slower than dense (8.5); the fast calls run a far
sparser mask (density never logged after the compile call; the pre-pass gathers on this machine go through the
unfixed generic all_gather and can change between calls). The parity test at the real 5 s shape confirms it: every
call 12 ms at density 0.55. Resident at ~50% density gives no speedup because the 64-row step costs ~1.6x per
(row, key) pair of the dense 256-row step. Projection at the required 85% sparsity (density 0.15): compute
~0.15 x 1.6 x 7.7 ms = 1.9 ms per core, under the 3.0 ms all-gather floor, so ~3.3-3.5 ms device / ~4.3 ms host =
2x. Measured next by a tau sweep with density logged per call: `./run_sol_validation.sh resident5`.

Accuracy of the dense op on this machine is with a separate probe (`./run_accuracy_probe.sh`, written by the
subagent): the all-gather itself is now verified page-exact here, the review of the arrival-signal thresholds found
no defect, and the probe separates premature reads / an SDPA-side defect / the K/V forwarding chain.

## 8. Second machine, resident5 result, and M2 (2026-10-21, late)

**resident5 on the second Galaxy (healthy pre-pass gathers):** tau works and the mask is stable call to call.

| tau | density at 64-row x 384-key | op ms (6 calls) | vs dense 8.5 | 64x64 block sparsity (derived) |
|---|---|---|---|---|
| 1 | 0.658 | 10.3 | 0.8x | ~80 % |
| 2 | 0.224 | 6.1 | 1.4x | ~93 % |
| 3 | 0.124 | 4.5 | 1.9x | ~95 % |
| 4 | 0.117 | 4.4 | 1.9x | ~95 % |

The last column is derived, not logged: the forcing rules (band 1, 1245 sink tokens, block 0 of every shard) fix a
floor of 5.0 % of blocks and 11.7 % of 384-key chunks; one random keep probability per block fitted to the logged
chunk density reproduces the Gaussian tail of tau to three digits (0.158 / 0.023 / 0.0013 / 0). Consequence: at 85 %
block sparsity with unstructured placement, 52 % of the chunks still hold a kept block, which is M1's 12 ms case.
M1 (whole-piece steps) cannot give 2x at 85 % on an arbitrary mask; the bench now also logs the 64x64 density
(`block density (64x64)` next to `density`).

**Dense op accuracy: root cause found and fixed (unbuilt).** The arrival probe (`test_dense_arrival_probe.py`, run
on the second machine, no DRAM harvesting there) showed that after the fused dense op every device's persistent
K/V buffer holds ring distance +1, +2, +3 exactly and half of distance 4, while distance -1, -2, -3 and the other
half of 4 are never written; with the exact gathered K/V pre-filled the same op reads PCC 0.9875 on every shard.
Cause (subagent, verified): in the ring-attention all-gather factory the backward writers' op-signaler block was
pushed with worker count 1 while the block holds one coordinate pair per forward worker (one per link). The
kernel's OpSignaler skips `count * 2` words, so with 2 links the backward writer's arg cursor sat 2 words short,
and the neighbour DRAM-bank table (appended after the signaler block by the 2026-10-21 fix) was parsed from inside
the receiver-core coordinates: "enabled" with Tensix coordinates, so every backward payload write was redirected
off DRAM while the arrival signal (explicit coordinates) still landed. Harmless before that fix (nothing was read
after the signaler block), which is why the first machine went 0.60 -> 0.71 rather than to 0.99. Fix: the
backward writers push the same worker count as the forward ones. `./run_sol_validation.sh m2` runs the nightly
test first.

**M2, written (unbuilt): block-level exact sets in resident mode.** `sol_mode="resident"` now takes the block-form
mask (the "compact" pre-pass output, bias generated on device):

- reader: per resident chunk, this shard's kept FULL blocks as bits (`sol_res_block_words` words, block 0 forced,
  the bias row patched to match), the chain-wide union per piece (any block of the piece), plan page =
  [block words][union lo, hi][do_pooled]; bias from the block row (`generate_tiles<false>`);
- compute (`sdpa_ring_resident`): a chunk steps on a piece when it keeps any block of it; the step gets a run table
  (`sdpa_inner_loop_step(..., run_cols)`): K^T sub-block s reads physical column `run_cols[s]`, V rows likewise,
  scores packed at the compacted columns, 2 tiles per sub-block, `active_Sk` = 2 x kept blocks; a fully kept
  piece runs the old full-width step. No in-entry masking (only full blocks are kept);
- host: resident validates a block-width mask; `_sol_prepass_mode` -> "compact"; `emit_bias` off for resident;
- tests: `test_resident_matches_compact` (was ..._chunk), `test_resident_call_parity` against compact, the tau /
  tail tests treat resident as block-form. Run: `./run_sol_validation.sh m2`.

Expected at 5 s, random inputs: compute per core ~ 10 chunks x 104 pieces x P(keeps a block of the piece) x ~3 us
narrow step + rotations; at tau 2 (block density 0.07, chunk-level 0.22) about 1 ms of steps, so the op should sit
on the all-gather floor (~4.3 ms host, ~2x) at every tau >= 1, and the speedup now follows block density.

## 9. M2 measured (2026-10-21 night, 180 W machine) and the head-boundary-core defect

Floors on this machine at 5 s: dense op 7.7 ms (6 reps 7.6-8.1), all-gather alone 3.2 ms. **Target set by the
user: 2x at 5 s at tau 1 = 3.85 ms, all-gather untouched.**

| tau | block density | M1 op ms | M2 op ms |
|---|---|---|---|
| 1 | 0.20 | 10.3 | 7.7 |
| 2 | 0.07 | 6.1 | 5.5 |
| 3 | 0.05 | 4.5 | 4.65 |

Why the 3.8-4.3 ms projection missed: (1) a narrow step is NOT proportional to kept blocks; the M2 sweep implies 5-8
us per narrow step against ~1.5 us per rotation, i.e. the 12 us full step was mostly fixed cost (handshakes, max /
exp passes, correction, and the FIFO traffic: ~12 tiles read from the front and written to the back per step, 12
copied per rotation) -- 1040 (chunk, piece) events per core per op; (2) compute and chain forwarding do not hide
under the all-gather: the chain is paced by the slowest member's per-piece sum, ~0.85 ms per shard at tau 1
against 0.4 ms of gather per shard. Per-shard budget for the target: ~0.4 ms for 13 forwards (~20 us each) plus
130 events, i.e. ~2 us per event -- not reachable with one step per (chunk, piece); needs block-major batched
steps over all resident rows with slot-indexed (in place) state and no rotations (M3, design in progress).

Correctness: exact (PCC 1.000000 vs compact) at 512 / 1024 rows; wrong at 1184 (0.87-0.94) and NaN at 4736, only
on cores whose resident chunks span two heads. Chains off (TT_SOL_NO_CHAIN=1): both pass (0.99998; 0.9999-1.0).
Mechanism: `ChainLink::forward` writes the piece to the SENDER's own K/V CB address on the next core, so every
member's K/V CB write pointer must be in lockstep; a two-head core pushes the second head's pieces (fetched
itself) into the same 2-deep CBs between the chain's forwards, drifts by one entry whenever an odd number of own
pushes fell in between, and the forward lands on the entry its compute is reading. Fix (factory, per-head chain
build): resident mode drops multi-head cores from the chain tail; they read both heads' K/V themselves.
**Verified (`m2b`)**: 1184 rows PCC 0.99998 vs compact (every slot 1.0000), 5 s parity finite at 0.99986-1.0. With
the fix the tau 1 op at 5 s reads **6.7 ms** (6 reps 6.70-6.75; was 7.7): the multi-head cores no longer pace their
chains. Target 3.85.

**M3 (in progress, 2026-10-21 night):** state per resident-chunk slot updated in place (the state CBs are exactly
max_q_per_core entries, filled once, never pushed/popped until the final normalize: unpack reads at an index from
the read pointer, pack writes at an index from the write pointer, which has wrapped to the base), no rotations,
and block-major steps: per piece, one step per kept 64-key block over every resident chunk that keeps it
(`sdpa_resident_block_step`, new; `SOL_RES_CHUNK_MAJOR` define = one step per chunk in chunk order for debugging).
Factory CB sizes are in (`cb_qk_im` max(qk_tiles, slots x Sq x 2), `cb_max_A` / `cb_exp_max_diff` slots x Sq).
Design estimate 22-28 us per piece (today 60+); a fused per-piece step with per-chunk run tables (M3b) is the
follow-up if the per-piece cost stays above ~18 us.

**Profile of the fixed M2 at 5 s, tau 1 (`profile_m2b_logs`, one rep, device span 5.39 ms, host 6.75 ms):**

| what | per event | note |
|---|---|---|
| narrow step, 1 block | 3.3 us (p10-p25) | the fixed cost of a step |
| narrow step, 4-6 blocks / full width | 8-10 us | ~+1 us per block pair |
| step mean incl. CB stalls | 7.2 us | 5% of steps > 20 us: waiting for the piece |
| rotation (skipped piece) | 1.1 us | 10 per piece at tau 1 |
| reader, one ring iteration (13 pieces) | 390-500 us | ~34 us per piece: paced by compute |
| chain receive wait | 10 us mean, p90 24 | members waiting for the injector = for compute |
| chain forward | 4.7 us mean, p90 15 | the write is ~1.5 us; the rest waits for members |
| pooled correction, first iteration | 470 us per core | 40 full-width steps (10 chunks x 4 pooled entries) |
| all-gather workers' kernel end | **2.6-2.75 ms** | the gather is done at half the op |
| host minus device | 1.36 ms | dispatch + the two host syncs of the timing wrapper |

**M3 first device run (2026-10-22 03:07):** 512 rows (keep-all, tau 2), 1024 rows keep-all and the q64 dense
equivalence pass at PCC 0.99993-0.99995 vs compact (M2 was 1.000000: the in-place rescale rounds the sum and
output to bf16 once more per step; above the 0.999 gate). 1024 rows / k 512 / tau 2 (the first geometry with two
pieces per shard and partial kept sets) hung with 13 cores stuck on device 0; every later step failed at device
open until a board reset. Fix applied: the step's end-of-step PACK_DONE token was taken by the NEXT step, and
`init_sdpa_streaming_semaphores()` re-initializes the T6 semaphores on the pack thread at every ring iteration, so a
token left dangling across that reset is lost and the unpack thread waits forever; the step now posts and takes
its token before returning. If the hang persists: `./run_sol_validation.sh m3_hang` (watcher waypoints of the
failing geometry, then the one-step-per-chunk variant).

**M3 second run (2026-10-22, token fix):** all 11 correctness tests pass (PCC 0.99986-0.99995 vs compact), 5 s
parity finite at 0.9993-1.0. Timing at 5 s: tau 1 **7.1 ms** (M2 6.7), tau 2 5.5 (5.5), tau 3 4.7 (4.65), parity
mask (47 %) **17.5 ms** (M2 12.1). Block-major did not help and is worse at high density. Cost fit from the two
densities (557 block-steps of ~2.2 chunks at tau 1, 624 of ~4.7 at 47 %): ~5.5-6 us per (chunk, block), no fixed
part to speak of. The per-(chunk, block) cost is the running-max machinery: the row max reduce, the alpha exp and
the in-place rescale of the sum and the 8 output tiles, which M2 paid once per (chunk, piece) and M3 pays once per
(chunk, block). The batched matmuls gained nothing because the matmuls were never the cost.

**M4 (started 2026-10-22): static per-row max.** Replace the running max by a fixed per-query-row upper bound of
the raw logits, ||q_row|| x max_k ||k|| + ln(64)/scale (Cauchy-Schwarz; the audit's "statically pre-seeded running
max", lever 4.A.4), computed by the pre-pass as a new op input `sol_row_max` (bf16 tile [1, NH, q_rows_pad, 32],
column 0), loaded by the reader into the max slots once. The step keeps only Q@K^T, the exp with the static max
accumulated into the slot sum, and P@V accumulated into the slot out: no reduce, no alpha, no rescale. Exact up to
rounding since the intermediates are bf16 (exponent range of fp32; an overestimate of the max by 10 logit units
costs nothing). Expected per (chunk, block) ~1 us: tau 1 compute ~1.8 ms per core, i.e. the op becomes
gather-bound (~3.1-3.3 ms device). Kernel switch `sol_static_max` keeps the M3 dynamic path when the tensor is
absent.

**M4 measured (2026-10-22):** tau 1 **5.85 ms** (M3 7.1, M2 6.7), tau 2 5.15, tau 3 4.58, 47 % mask 11.0 ms (M3
17.5). Removing the per-step max / alpha / rescale is worth 1.25 ms at tau 1 and 6.5 ms at 47 %. But the a-priori
bound is too loose for bf16: PCC vs compact 0.9996 at 512/1024 rows (M3 0.9999) and **0.9918 at 1184** (gate
0.999). The bound sits 10-20 scaled units above the true max, and (score - max) is a bf16 subtraction of two
numbers of magnitude ~100-180 raw units, whose absolute resolution (0.5-1 raw unit, 4-9 % of the exp argument
per term, random per term) is the error. Scaling all terms equally is harmless; the lost resolution of the
argument is not. Fix (**M5, frozen max**): each chunk's max is the row max of its FIRST step (the forced block 0
of piece 0, which holds the sink keys) and is frozen; later keys above it only make their terms exceed 1, which
bf16 absorbs (overflow needs ~85 units). No pre-pass tensor needed (sol_row_max stays an accepted, unused input).
Remaining cost after M4: ~3 us per (chunk, block) = ~0.75 us per score tile, i.e. the exp on the pack thread and
the two packs per tile (scores back in place, sum L1-accumulate). Next levers in that order: the approximate SFPU
exp (`SOL_EXP_APPROX=1` A/B in `m5`), the row sum as a P x ones matmul instead of per-tile packs, and the pooled
phase (0.47 ms). Run: `./run_sol_validation.sh m5`.

**M6 (written 2026-10-22, unbuilt; run `./run_sol_validation.sh m6` after `m5`):** the step covers one piece for a
sub-group of up to 4 resident chunks, each with its own run table of kept blocks (score scratch = 4 x Sq x
Sk_chunk_t tiles, +112 KB L1), so the output accumulator is packed once per (chunk, piece) instead of once per
(chunk, block), and the row sum is P x e0 (the column-identity tile; the writer now zeroes it before writing
column 0) accumulated into the slot sum instead of one L1-accumulate pack per probability tile. At tau 1 that is
~32 % fewer packs per op on the pack thread; the exp count is unchanged, which is what the `SOL_EXP_APPROX=1`
A/B in `m5`/`m6` measures.

## 10. M5 measured (2026-10-22): the frozen max is wrong, and why

**Result.** Timing as M4: tau 1 **5.78 ms**, tau 2 5.1, tau 3 4.5; `SOL_EXP_APPROX=1` 5.83 (no change). Correctness
fails everywhere: PCC vs compact 0.69-0.72 at 512/1024 rows (keep-all included), 0.81 at 1184; 5 s parity 0.70-0.72.
No NaN; the resident output's std is 0.068 against 0.100 for dense (the weights are flattened). Per head the PCC
falls with the chunk's position along the ramp: chunks 0-3 at 0.94-0.997, chunk 4 at 0.43, 5 at 0.25, 6 at 0.18,
7 at 0.13 (512 rows; same shape at 1024 with the last chunk worst).

**Cause.** The resident step's exp is `exp_packthread_tile<approx=true, InputClamping::None>`
(`ckernel_sfpu_exp.h`, the Schraudolph bit trick with a signed 16-bit round): **"for inputs above 0.72 the output
saturates to exp(0.72)"**. The dense/compact paths never feed it a positive input (score minus the running max
is <= 0), so the limit was invisible. With the frozen max every key that scores more than 0.72 above the first
block's row max gets the weight ~2.06 instead of exp(score - max), and on the ramp data the gap between block 0
and the diagonal grows with the chunk's position (chunk 4: ~10 scaled units, chunk 7: ~75), which is the
observed gradient. The reasoning in M5's design ("later keys above the max only make their terms exceed 1, which
bf16 absorbs") was right about bf16 and wrong about the exp. A second, independent limit stays even if the exp
were fixed: the dest register is 16-bit, so (score - max) is rounded to bf16 before the exp, and its resolution
is 0.25 at magnitude 32-64; that is what capped M4 at 0.992. Conclusion: **the max must be an upper bound of the
row's scores to within ~0.7, and within a few units of the true max** -- i.e. a running max, or something that
tracks it.

**`SOL_EXP_APPROX` A/B is void** for resident mode: the step's exp is the approximate pack-thread one regardless;
the flag only reaches the alpha exp (S3) and the dense/compact paths. Dropped from `m6`.

**M6 measured with the frozen max (2026-10-22, the old `m6` step):** tau 1 **5.62 ms** (M5 5.78), tau 2 4.98
(5.10), tau 3 4.45 (4.50), 47 % parity mask 10.35 (10.95). Correctness fails as in M5 (0.72-0.82; 0.98 at 512
rows because the first step's max now covers the whole 384-key piece 0, i.e. most of a 512-row shard). The fused
step removed about a third of the pack-thread work for a 3 % gain: the pack count is not what the ~5 us per
(chunk, piece) is made of. The matmuls already run at HiFi2. What is left per event is the matmul work itself
(~28 tile-ops, ~1.5 us), the exp and packs (~1 us), and thread rendezvous / MOP reconfiguration (several
`tile_regs` round trips and format reconfigs per chunk, ~1.5-2 us): no single dominant term, hence the small
wins. The `m6` step now also profiles the dyn variant (`m6_profile`, fine zones) to measure that split.

**Frozen path removed (2026-10-22, after the M6-frozen run):** the `sol_frozen_max` named arg, the
`TT_SOL_DYNAMIC_MAX` switch and the `static_max` template path of the step are gone; resident compute always
runs the running max (S2-S4 every step: the step's row max merged with the stored max is what S5 subtracts, so
every exp input is <= 0 by construction). Audit of every exp this work added or touched: S5's pack-thread
approximate exp gets (score - merged max) * scale <= 0, with the packer ReLU set around it for inputs below -88;
S3's alpha exp gets (old max - new max) * scale <= 0 because the new max is the elementwise max of the two; the
final normalize divides by a sum >= 1 (the max element contributes exp(0)); the dense/compact exps are unchanged
and subtract their own running max. The `sol_row_max` tensor (M4) is still accepted and ignored; removing its
plumbing (types, validation, reader arg layout, pre-pass, model, test) is a separate cleanup. The `m6` step now
runs the correct kernel only: correctness set, parity, tau 1/2/3, fine-zone profile. Rebuild required.

**M7 plan (no code until M6 is measured).**
1. *Max that is correct at near-frozen cost (lazy rescale).* Keep S2 (the step's row max) every (chunk, piece)
   event but decide on device whether a rescale is needed: d = step max - stored max (2 tiles), column max-reduce
   to one scalar, pack it, all three threads read it from L1 after the PACK_DONE token. If d <= 0.5 (inside the
   exp's safe range): keep the stored max, skip S3/S4, S5 subtracts the stored max. Otherwise the full S3/S4 with
   the new max. Iteration 0 is the local shard and holds every chunk's diagonal block, so after it the max rarely
   grows. Estimated cost ~0.5 us per event (~300 events per core at tau 1) against ~3 us for the full stats,
   i.e. ~0.15 ms of the 1.25 ms that M3's always-on stats cost. Skip this if M6-dyn lands within ~0.3 ms of
   M6-frozen.
2. *Pooled phase (0.47 ms per core = 10 chunks x 4 pooled entries x 24 score tiles, ~35 % of the exp work at
   tau 1).* The math must stay compact's (ln 64 bias on dropped blocks, -inf on kept), so the levers are (a) the
   bias add: a 0/-inf row mask per chunk applied as one L1-accumulate of a precomputed tile row instead of per-tile
   generation, (b) scheduling: the pooled K/V come from DRAM, not the gather, so the pooled steps can run in the
   gather wait of iterations 1-7 once the per-iteration compute is below the gather's per-shard spacing
   (~0.34 ms); in a compute-bound regime they add in full. (c) Coarser pooling (128 keys) halves it but changes
   the approximation -- a decision for the user, not an optimization.
3. *Re-profile M6-dyn* (step zones, pack-thread occupancy) and re-rank: remaining per-score-tile cost, chain RX,
   K/V CB depth.
Arithmetic of the target: 3.85 ms host = 2x; the gather alone is 3.2 ms host (2.7 device), so the op must be
gather-bound with a <= 0.65 ms tail: per-iteration compute <= ~0.34 ms per core and the pooled work in the slack.
M5-frozen compute is ~4.4 ms device per core; M6 should remove a third of the packs; items 1-2 are what remains.

## 11. Plan to 4.3 ms (2026-10-22, written while the first correct M6 run was in progress)

**Budget.** Host op time = device time + ~1.36 ms of harness overhead (the same for dense). 4.3 ms host =
~2.95 ms device. The gather delivers shard k at ~0.34 k ms and finishes at 2.7 ms, so the device time is
max(compute-bound: 686 events x e + pooled + tail, gather-bound: 2.7 + last shard's compute + tail), where e is
the cost of one (chunk, piece) event (686 per core at tau 1, 1.73 kept blocks each) and the tail is the final
normalize + output write (~0.1 ms). The gather-bound floor is ~2.85-2.9 ms device = **~4.2-4.3 ms host: 4.3 is
the floor itself**, reached only when compute hides under the gather, i.e. e <= ~2.5 us and the pooled work in
the slack. Today: M6-frozen e ~5.4 us (4.26 ms device); M6 with the running max is expected at e ~6.

**Tier 1 -- known levers, sized from the M2/M4/M6 measurements (confidence medium; the M6 profile fixes the
numbers).** Expected landing: **~4.9-5.1 ms host**, correct.
1. Lazy rescale (section 10, item 1): recovers ~80 % of the running-max cost, -0.25 to -0.4 ms.
2. Group-batched S6: P@V and the row sum of all chunks of a group in 2-3 dest batches with one matmul setup
   and one pack configuration, instead of 2-3 batches and 2 setups per chunk. The M6 pack-count experiment
   (a third fewer packs for 3 %) says the per-chunk overhead is in the thread rendezvous and the MOP/format
   reconfiguration, not in the packs: ~9 fewer round trips and ~6 fewer setups per group of 4, ~0.5 us per
   event, -0.3 to -0.4 ms.
3. Pooled phase: run the pooled entries through the same group batching (40 one-chunk steps -> 12) and apply
   the ln(64) / -inf bias as one precomputed row per chunk, -0.1 to -0.15 ms of its 0.47; the rest hides only in
   gather slack once Tier 2 lands.
4. Larger sub-groups (4 -> 6 chunks per step) where L1 allows (score scratch +50 %), -0.05 to -0.1 ms.

**Tier 2 -- what 4.3 needs beyond Tier 1: e from ~4.3 to ~2.5 us.** Both are restructurings of the step's
data flow, to be scoped from the profile's math-vs-pack occupancy.
5. Pipeline the step across groups: QK^T / P@V (math) of group n+1 while the pack thread runs exp + packs of
   group n, the dense kernel's two-phase structure applied to the resident step. Today every stage is a fill-
   then-drain round trip with no overlap (full dest sync, 16 tiles). Potential: e -> max(math, pack) ~1.5-2.5 us.
6. Two pieces per step where the K/V CB can hold them (+192 KB L1 per extra page): halves the per-event fixed
   cost against the same per-block work (343 events instead of 686).
Not on the table: all-gather changes, bf8 K/V, LoFi, coarser pooling (an approximation change: user's call).
The 1.36 ms harness overhead is inside the 4.3 and not reducible by the op (trace removes it end to end).

**M6 with the running max, measured (2026-10-22, the re-run `m5` step after the frozen path was removed):** all 11
correctness tests pass (0.99990-0.99998 vs compact), parity 0.9996-1.0, tau 1 **6.20 ms**, tau 2 5.12, tau 3
4.55, 47 % mask 11.9. The running max costs 0.58 ms at tau 1 over the frozen floor (5.62).

**Cross-check against the VSA ring-op audit (claude.ai artifact, 2026-10-22):** a sibling op (vsa_ring_sdpa,
SP=8 gather fused into the fine stage). Its levers that transfer: (a) "reduce pack->unpack syncs per chunk,
bigger visits" = Tier 1 item 2 and Tier 2 items 5-6; (b) "split dense rows across a head's workers and merge the
partial states once" = load balance across cores, which we have not measured: the op's time is its slowest
core's, and the M6 profile's per-core RING_STEP totals will show the spread (a density-aware chunk-to-core split
is the fix if it is large; the chain structure allows any assignment within a head's cores); (c) gather-side
items (MUX multi-worker gather, gate signalling, sender-core occupancy) are all-gather changes, off the table
here, and our reader never waits for the gather anyway (RD_SYNC 44 us). Nothing else in it applies: its
transposed design (KV-resident cores, multicast Q) is a different kernel family, and its DRAM-partials transplant
it rejects itself. No obvious lever missed.

**Tier 1 implemented (2026-10-22, unbuilt; run `./run_sol_validation.sh m7` after a rebuild):**
1. Lazy rescale in `sdpa_resident_piece_step`: S2 unchanged; new S2b packs d = new max - stored max of every
   initialized member into the alpha scratch, column-max-reduces them (REDUCE_COL) into one tile and the unpack
   thread's RISC reads its [0][0] after polling the PACK_DONE token (a Tensix-side wait would not order a RISC
   load); `read_tile_value` hands the word to the math and pack threads through the mailboxes, so the three
   threads take the same branch. Threshold 0.5 / scale raw units, compared as bf16 bit patterns (truncated, so
   conservative). Rescale: S3 alpha + copies, S4 as before. Lazy: S3 copies only the first-step members' max,
   no alpha, no S4. S5 now always subtracts cb_max_in, after a PACK_DONE token drains the S3 packs. The exp
   input is (score - stored max) * scale <= 0.5 by construction.
2. S6 batched across the group: P@V units (member, V sub-block) fill DEST up to pv_units_per_acq per batch with
   one matmul setup and one pack configuration for the group (L1-accumulate toggled per unit as members differ),
   then all members' row sums in one batch.
3. Pooled entries stepped in sub-groups of kGroupCap chunks (3 steps per entry instead of 10); S1 applies each
   member's bias row in turn (the bias CB holds one row; the reader's push order is the member order).
4. Larger sub-groups dropped: with 6.6 chunks keeping a piece on average, cap 4 and cap 6 both give 2 steps.

## 12. M7 measured and what the M6 profile actually says (2026-10-22, evening)

**M7 (Tier 1) measured:** correct (11/11 at 0.99989-0.99997, parity 0.9996-1.0, 47 % mask 11.24 ms from 11.87),
tau 1 **6.16 ms** (M6 6.21), tau 2 5.26 (5.10, noisier), tau 3 4.53 (4.55). Tier 1 bought ~1 %, against an
estimate of 10-13 %. Together with M6's 3 % for a third fewer packs, the conclusion is firm: the time is not in
the group-level stages (stats, sync, packs per group) that M5-M7 reorganized.

**M6 profile, full (all 32 chips, ~121 captured steps per core = iterations 0-3; `m6_iter2.py`).**
- *Per-tile cost is the problem.* A pooled step (one chunk, the full 12-tile width, contiguous: the best case for
  our step) takes 22.6 us for 24 score tiles = **0.94 us per score tile**. Piece steps (groups of up to 4 chunks,
  ~1.7 blocks each) average 12-14 us for ~11 score tiles = ~1.2 us per tile. The dense kernel does the whole
  problem (22.3k score tiles per core) in ~6.2 ms of device time = **0.28 us per score tile**. Our step is 3.5x
  slower per tile than the dense streaming step on the same hardware, and that ratio has not moved since M2. The
  FPU work per score tile (8 HiFi2 tile-ops) is ~320 cycles; the step spends ~1300-1700. The difference is the
  issue mechanics: per-tile `sdpa_matmul_block_no_mop` calls (the RISC issues every tile-op instead of a MOP
  sequence), per-tile address-programmed packs (`sdpa_pack_tile_ooo`), per-unit sub/exp with a flush each, and
  one fill-then-drain round trip per stage. M4's 1.25 ms gain fits the same model: it removed ~1200 per-block
  stat operations, each paying that per-operation cost.
- *The pooled phase is 0.85 ms per core, not 0.47.* 36-40 pooled steps x 22.6 us; the reader's RD_RES_POOLED
  zone (0.44 ms) only covers its pushes. That is ~25 % of a core's compute at tau 1.
- *Cores wait on the chain.* Per ring iteration a core's steps account for 290-360 us but the iteration takes
  330-575 us: fast chains lose ~45 us per iteration outside the steps, median and slow chains 180-215 us. The
  per-core step time differs by ~10 % between the fastest and slowest cores; the end times differ by 1.5 ms
  (3.4 vs 4.9 ms on the same chip; all chips: p10 3.9, median 4.5, p90 4.9, max 5.05 ms). The chain forwards a
  piece only when every member has freed a slot (K/V CB depth 2), so a head's cores run at the pace of their
  slowest member per piece, and the op ends with the slowest chain. The reader never waits for the gather
  (RD_SYNC ~5 us per core per op).

**Revised plan.**
1. *Measure the lockstep cost:* `./run_sol_validation.sh m7x` = `TT_SOL_NO_CHAIN=1` at tau 1 and 2 (every core
   fetches its own pieces; no rebuild). The gap to 6.16 ms bounds what decoupling the chain (deeper K/V CB,
   +192 KB L1 per page, or a non-lockstep forward) can buy; the profile says up to ~1.4 ms of device time.
2. *Tier 2 redefined: the step's inner loops on the dense kernel's MOP primitives.* QK^T and P@V as block
   matmuls per run (adjacent kept blocks merged into one run; MOP sequences instead of per-tile issue), the exp
   through the dense `sub_exp_block_bcast_cols` path (32-iteration exp, `pack_contiguous_rows` block packs),
   block packs for the outputs. Start with the pooled steps: contiguous, full width, 25 % of the compute, the
   dense primitives apply unchanged (expected 0.85 -> ~0.25 ms per core). Then the piece steps. Target per-tile
   cost 0.3-0.4 us: per-core compute from ~3.3 ms to ~1.5 ms, which makes the op chain/gather-bound at
   ~2.7-2.9 ms device = 4.1-4.3 ms host.
3. Tier 1 stays (correct, marginally faster, and the lazy rescale is the right mechanism once per-tile costs drop
   and the stats become visible again).

**m7x measured (2026-10-22, 20:11):** `TT_SOL_NO_CHAIN=1` (every core fetches its own pieces from the gathered
buffer): tau 1 **9.85 ms**, tau 2 8.30 (exact: 0.99997 / 0.99995). The chain is worth 3.7 ms at tau 1; the
per-core DRAM/NoC fetch of 104 pieces x 192 KB is far slower than the forward. The chain stays. `TT_SOL_KV_PAGES=3`:
6.23 ms (M7 6.16): one more buffer page buys nothing. 4 pages do not fit: the CBs reach 1,696,064 B against the
1,572,864 B L1 (3 pages leave ~73 KB). So the compute idle time inside each iteration (13-40 %) is not lockstep
slack that a deeper buffer absorbs; what paces the chain has to come from the reader zones (per-piece delivery
intervals, RX/TX waits by chain position), extracted next from the M6 profile.

## 13. The chain paces the op (2026-10-22, late: M6 reader zones, chip 10, `m6_reader.py`)

Per core, the reader's piece arrivals (RD_CHAIN_RX end on members, RD_FETCH_K end on injectors) give the
delivery interval; the compute thread's kernel end gives the core's finish time:

| role | cores | finish (median / min / max) | piece interval (median of per-core medians) | RX wait (median) |
|---|---|---|---|---|
| standalone fetcher (reads its own K/V from DRAM; the two-head cores dropped from chain tails) | 2 | 3.46 / 3.46 / 3.48 ms | 12.5-13.6 us | 0 (fetch 3.6 us) |
| chain injector | 24 | 4.37 / 3.46 / 4.86 ms | 23.1 us | 0 |
| chain member (forwards) | 70 | 4.58 / 4.31 / 4.89 ms | 25.8 us | 6.0 us |
| chain tail | 18 | 4.51 / - / 4.90 ms | 26.0 us | 12.5 us |

A piece every 25-28 us per chain (mean 32-35 with the gaps) against ~22-28 us of compute per piece per core:
the chain members' compute waits on the chain, which is the 13-40 % idle time of section 12, and the slow chains
end 1.4 ms after the standalone fetchers that get a piece every 13 us. The per-core step time differs by ~10 %;
the delivery pace is what differs.

Why the chain is slow: `chain_link.hpp` is unicast store-and-forward with a one-piece handshake per hop (the
receiver flags itself invalid, signals the upstream core, waits for the data; the sender waits for that signal,
writes 192 KB, flags valid). Each hop therefore adds its transfer plus two semaphore round trips to the piece's
latency, and a hop cannot take piece p+1 until it has forwarded p, so the chain's cycle grows with its length
(7-8 cores at 5 s). A third CB page cannot help (the handshake, not the buffer, serializes), which `m7x`
confirmed; no chain at all (every core from DRAM) saturates DRAM/NoC at 9.85 ms. The multicast variant of the
chain exists but is disabled for the Sol chunk modes (resident included) and is all-or-nothing with a same-row,
no-gap, uniform-q-count eligibility that 7-8-core chains on 11-core rows rarely meet.

**Revised plan (both parts are needed for 4.3 ms):**
- *Chain:* first `./run_sol_validation.sh m7x2` = `TT_SOL_CHAIN_LEN=4` and `2` (segments with their own DRAM
  injector, 2x / 4x the K/V DRAM reads, no rebuild): if the time falls with the length, the fix is
  row-segmented chains with a resident-aware multicast (one hop for every member; the union plan already makes
  every member stream every union piece, so the multicast injector never waits for a receiver that skipped),
  or, failing that, short unicast segments. Expected: per-piece delivery ~13-18 us, i.e. compute-bound again;
  the standalone fetchers' 3.46 ms says the op would land near 3.5 ms device = ~4.8 ms host at today's compute.
- *Compute (Tier 2 of section 12):* the step's inner loops on the dense kernel's MOP primitives, pooled steps
  first (0.85 -> ~0.25 ms per core), then the piece steps; compute per piece 25 -> ~10 us. With the chain fixed
  as well, the op becomes gather-bound at ~2.8-2.9 ms device = ~4.2-4.3 ms host. Compute alone (chain as is):
  ~3.2 ms device = ~4.5 ms host; chain alone: ~4.8.

**m7x2 measured (2026-10-22, ~21:00):** `TT_SOL_CHAIN_LEN=4`: **6.52 ms** (M7 6.16), `=2`: 8.05, `=1` (no chain):
9.85; exact at L=4 (0.99990). Shorter chains are slower: each extra injector adds a full set of K/V DRAM reads
and the DRAM/NoC path is what saturates. So the chain's 25-28 us per piece is not hop-count latency; it is the
per-hop handshake: `receive()` signals "ready" only when a hop starts the next piece, i.e. after it has forwarded
the previous one, so every hop idles while its neighbours move (a bucket brigade: throughput ~ 1 / (2 x hop
time)), independent of the chain length and of the CB depth. The standalone fetchers' 13 us per piece is the
one-hop time. Correction to section 12: the pooled phase is 0.45 ms per core (20 steps x 22.6 us: 2 pooled
entries per chunk, not 4), ~14 % of a core's compute.

**M8 (written 2026-10-22 night, unbuilt; `./run_sol_validation.sh m8` after a rebuild):** chain overlap. In
`chain_link.hpp`, `receive()` is split into `signal_ready()` (flag invalid, signal upstream) and `wait_data()`.
In the reader's resident piece stream, a member pushes piece p, then, if the CB page for p+1 is already free
(non-blocking check), reserves it and signals ready for p+1 BEFORE forwarding p, so the upstream's write of p+1
overlaps this core's forward of p. The armed page carries to the next piece; nothing is armed after the last
piece of an iteration (pooled fetches and dummy pushes reserve as before). The iteration's piece list is built
first so each piece knows whether another follows. Fetchers only gain the early reserve. Dense paths keep
`receive()` (the `m8` step re-runs the dense equivalence test). Expected: chain pace 25-28 -> ~13-15 us per
piece (the injector's fetch + forward), every core compute-bound: ~3.5 ms device = **~4.8-4.9 ms host (1.6x)**.
Then Tier 2 (the step's per-tile mechanics, pooled first) and an injector prefetch (issue the DRAM read of p+1
before forwarding p) for ~4.2-4.3 ms (1.8x).

**M8 first run hung (2026-10-22, 20:47): root cause and fix.** The 512-row check (one piece per iteration, so
nothing armed) passed; the 1024-row check (two pieces) hung with every reader in `noc_semaphore_wait` and
compute/writer in CB waits (log_triage.txt). Cause: in MHA `k_chain` and `v_chain` are the SAME `head_chain`
object, so a link's K and V handshakes share one ready count and one valid flag. Signalling K-ready(p+1) right
after pushing K(p) handed the upstream, then waiting for V-ready(p) on the same count, a second increment: its
equality wait (`sender_sem.wait(1)`) never returned. Fix: arm only the link's next handshake in sequence --
V(p)'s ready before forwarding K(p), K(p+1)'s ready before forwarding V(p) -- so the overlap is the same (the
upstream's write of the next chunk overlaps this core's forward of the current one) and the link never has more
than one transfer outstanding. Lesson recorded in chain_link.hpp: "one outstanding per link" means per ChainLink
object, not per tensor.

**M8 measured (2026-10-22, 21:19-21:42, fixed protocol):** no hang; correctness as before (512 rows 0.99993,
dense-equivalence q64 0.99997, parity 0.9996-1.0, 47 % mask 11.22 ms); tau 1 **6.12 ms** (M7 6.16), tau 2 5.21,
tau 3 4.55. No gain. The early ready signal is sent only when the next page is already free, and with a 2-deep
K/V CB that page holds piece p-1, which compute is still on when p arrives (compute ~25 us per piece, chain
~25 us per piece), so the arm rarely fires and the protocol degenerates to the old one. So both constraints bind:
depth 3 with the old handshake gave nothing (m7x), the new handshake with depth 2 gives nothing (M8). Next:
`./run_sol_validation.sh m8x` = M8 kernel with `TT_SOL_KV_PAGES=3` (no rebuild), tau 1/2, a 1024-row check and a
profile (RD_CHAIN_RX waits on members should collapse if the overlap happens). The dense equivalence run in `m8`
passed its first parametrization and was interrupted while opening the device for the second: `m8d` re-runs it
alone.

**m8x measured (2026-10-22, ~22:30): M8 handshake + 3 CB pages: tau 1 6.26 ms, tau 2 5.20; exact (1024 rows
0.99997); dense equivalence 4/4 pass with the shared header.** No gain from either the overlapped handshake or
the depth. Per-piece compute across chain members (M6 profile, chip 10, iterations >= 1): mean 18.9 us, median
20, p90 35, 21 % of pieces with no compute at all; for groups of 8 consecutive cores, sum over pieces of the
slowest member's compute / the mean member's compute = **1.51** (1.43-1.56). Reading: the chain runs each piece
at its slowest member's pace (one transfer outstanding per link, so a hop cannot run more than one piece ahead
of its compute), and with a random 20 % mask the slowest of 8 members is 1.5x the mean. That is the 35 % idle
time of the members, and it explains why no chain-side change moved the op: the overlapped handshake still
leaves one transfer outstanding, so three pages cannot absorb variance either, shorter chains add DRAM traffic,
and the compute of the slowest member is what every piece waits for. It also explains why M4 (less compute per
block) moved the op by 1.25 ms while M5-M8 (group-level costs, chain mechanics) moved it by ~1 %: the op scales
with compute per tile, through the lockstep. Conclusion: Tier 2 (the step's per-tile mechanics, 3.5x the dense
kernel's) is the lever, and it scales the chain pace with it; a credit-based multi-transfer chain protocol would
add at most the ~15 % that 3 pages of slack can absorb. M8 stays (correct, harmless, and useful once transfers
are multi-outstanding).

## 14. M9 = Tier 2 written (2026-10-22 night, unbuilt; `./run_sol_validation.sh m9` after a rebuild)

A new step, `sdpa_resident_piece_step_compact` (the M5-M8 step stays in the file, unused), and new grouping in
`sdpa_ring_resident`. What changed in how the work is issued:
1. *Compact score layout.* Member g's scores are Sq_chunk_t rows of exactly W_g tiles (its kept width) at offset
   sum of the earlier members' Sq_chunk_t x W; row stride W_g instead of the piece width. The scratch (96 tiles)
   then holds every chunk of the head that keeps a block of the piece at tau 1 (10 x 2 x ~3.4 = 68 tiles): one
   step per piece instead of two, and each stage runs several DEST batches back to back (the kernel's DEST is
   half-sync, 8 tiles, so math and pack overlap across consecutive batches). Groups split by cumulative width
   when the scratch would overflow (47 % mask: 2 steps; keep-all: 3).
2. *Merged runs.* Adjacent kept blocks form one run (2, 4, 6 ... tiles; `run_ws` per run). A unit of up to
   qk_cw_cap = 4 columns is one matmul call and, at >= 4 tiles (MIN_BLOCKED_PACK_TILES on BH), one pack call per
   row instead of one per tile, in S1 and in the exp stage (S5 used single-tile packs before).
3. *P@V at the widest DEST sub-block* (pv_w = min(vDHt, kDst / Sq_chunk_t) = 4): one matmul call per P column
   tile per member and blocked output packs.
4. Members sorted by width within a group so the width-dependent MOP setups (row-max reduce, bcast subtract, the
   two P@V setups) are re-issued only when the width changes.
Also: S2b's test batches over DEST properly for groups larger than four members. The lazy running max, the row
sum as P x e0, the pooled bias and the token protocol are unchanged.
Expected: fewer steps (~104 instead of ~200 per core), fewer matmul and pack calls per tile, more overlap. With the
per-piece compute at ~19-25 us today and the chain following the slowest member, a 25-35 % cut in compute per
piece would put the op near 4.9-5.2 ms host; the m9 profile then shows the remaining per-tile split.

**m8x profile reviewed (chip 10, `chain_review.py`; M6 values in brackets):** device span 4.84 ms [4.90]; chain
members' piece interval median 22.7 us [25.8], mean 31.0 [32.3]; tails 23.1 / 35.7 [26.0 / 36.0]; member receive
wait median 6.6 us [6.0], tails 8.3 [12.5]. The median member core computes 265-272 us per iteration against a
442-498 us iteration: ~45 % idle, as before. Per-piece compute mean 17.8 us [18.9], slowest-of-8 / mean **1.60**
[1.51]; mean interval / mean compute = 1.74. So the overlapped handshake plus the third page shortened the
median gap by ~3 us and changed nothing in the mean, the idle fraction or the finish times: the chain follows the
slowest member's compute on every piece. This is the evidence for Tier 2 and against further chain work: the op
scales with compute per piece, and nothing on the transport side is left to recover at this compute level.

**M9 first build (2026-10-22, 22:31-22:53):** the correctness set passed, 11/11 at 0.99989-0.99997 (the small
geometries build), but every 5 s run failed at program creation: "Program size (73728) too large for kernel
config buffer (70656) on TENSIX". The compact step's two flush lambdas were inlined at 2 and 3 sites and the
width-sort/grouping lambda at 2 (piece loop and pooled loop), on each of the three TRISCs; the 5 s shape also
carries the largest runtime-arg tables. Fix: `__attribute__((noinline))` on the three lambdas (one copy per
TRISC, a call per DEST batch, which the dense kernel's own measurements put below noise). If the budget is still
short, the next outlining candidates are the S6 pack loops and the S2 reduce block.

**M9 measured (2026-10-23, after the outlining):** tau 1 **6.12 ms** (M8 6.12), tau 2 5.17 (5.21), tau 3 4.52
(4.55), 47 % parity mask 10.84 (11.22, -3 %: merged runs pay where blocks are adjacent). Correct (11/11 on the
first build). Tier 2 as built moved nothing at tau 1.

**Why (what the numbers say now).** Two facts frame it. (1) The op's floor without any compute is the gather plus
the tail plus the harness: ~2.7 + 0.15 + 1.3 = ~4.15 ms host; tau 3 (5 % density) sits at 4.52, i.e. the whole
compute-dependent part at tau 1 is ~2 ms of the 6.1, and everything since M4 has been fighting over that 2 ms.
(2) Counted against the dense kernel's step, the compact step issues a similar amount of work per score tile:
a pooled step (24 tiles) is 36 matmul calls of <= 8 tile-ops, 24 SFPU exps and ~40 packs; the dense step does
~32 calls, 32 exps and ~76 packs per 32 tiles. The dense step takes 0.28 us per tile, ours ~0.95. So the gap is
not the number of calls or packs (M6, M7 and M9 all reduced those and gained 1-3 %); it is that our stages run
fill-then-drain, one after another: QK^T (math) -> pack -> subtract (math) -> exp (pack thread) -> pack -> token
-> P@V (math) -> pack, so math, SFPU and packer time ADD, while the dense kernel overlaps the exp and packs of
one row group with the matmuls of the next (its math util is ~70 %). Our step's math alone, at the dense rate,
would be ~7 us of the pooled step's 22.6.

**Next: measure the split before changing anything else.** `./run_sol_validation.sh m9a` (rebuild first) runs
tau 1 six times with one stage's instructions removed each time (`TT_SOL_ABLATE`: qk, exp, pack5, pv, stats,
pack1; CB/DEST/token operations stay, results wrong, timing valid). The drops against the baseline are the shares
of Q@K^T, the exp, the exp-stage packs, P@V + row sums, the stats and the score packs. If the shares add up to
about the step time, the stages are serial and the fix is to pipeline them across members (sub+exp of member i
on the pack thread while P@V of member i-1 runs on math, as the dense kernel does across row groups); if the
"stats" share is large, the lazy test itself is too dear; if no single share is large, the cost is in the
rendezvous between stages and the step needs fewer of them.

## 15. Ablation (2026-10-23) and the bounded expectation for what is left

`m9a`, tau 1, host op ms (6 reps): baseline **6.12**; Q@K^T matmuls skipped **5.55** (-0.57); exp skipped
~6.02 (-0.10, reps 5.87-6.24); exp-stage packs skipped **6.00** (-0.12). The P@V knob hung the op on its first
run (no diagnostics; the sweep stopped there, so the stats and score-pack knobs did not run). P@V + row sums
issue the same tile-op count as Q@K^T with slightly more unpacks, so ~0.55-0.6 ms; the stats (row max + lazy
test) are the residual, ~0.4; the score packs ~0.12. Sum ~1.9 ms = the whole compute-dependent part of the op
(6.12 - the 4.15 ms no-compute floor): the stages are serial, as section 14 argued.

| stage | share of op (ms) | basis |
|---|---|---|
| Q@K^T matmuls | 0.57 | measured |
| P@V + row-sum matmuls | ~0.55-0.60 | same tile-ops as Q@K^T, more unpacks |
| row max + lazy test | ~0.4 | residual |
| exp | 0.10 | measured |
| exp-stage packs | 0.12 | measured |
| score packs | ~0.12 | as exp-stage packs |

**What the remaining levers can return, bounded by these shares.**
- Pipelining (sub+exp of member i on the pack thread under P@V of member i-1 on math): hides at most the exp and
  the two pack stages, 0.34 ms; realistically 0.2-0.3. -> ~5.85 ms.
- Wider matmul blocks (rows of two adjacent members per Q@K^T call so K tiles are unpacked once per pair; the
  same for V): the matmul shares are ~1.15 ms; sharing in1 unpacks across two members saves perhaps a quarter,
  -0.25-0.3 ms. Uncertain: it depends on how unpack-bound the calls are.
- Both: **~5.5 ms, 1.4x host** (device ~4.2 ms against dense's 6.4, ~1.5x, which is what a traced model sees).
- The stats could only be made cheaper by reducing the max once per two pieces or folding it into the Q@K^T
  batch; neither is small work.
Not reachable in this design with the gather untouched: 4.3 ms host, because the no-compute floor is ~4.15 and
the compute-dependent part cannot go below ~1 ms at tau 1 (the matmul work alone at the dense kernel's rate).

**M9 profile (chip 10, `chain_review.py`; m8x values in brackets):** device span 4.73 ms [4.84]; per-piece
compute mean 19.6 us [17.8], median 18.9 [17.7]; median member core 262-266 us of steps per 469-473 us
iteration [265-272 / 442-498]; chain interval mean 30.9 us [31.0]; slowest-of-8 / mean 1.65 [1.60]; steps are
fewer and longer (mean 24.0 us [15.4], the profiler's ~120-step window now covers more of the op). So the compact
step did exactly what it was designed to do structurally and did not reduce the compute per piece at all: the
per-member MOP setups and the width sort cost about what the halved step count saved. Consistent with the
ablation: the time is in the matmul and stats work itself and in the serial stage order, not in step overheads.

## 16. Traced timings (2026-10-23)

The bench timed the attention op as a separately dispatched op (wall clock between two synchronize calls), so
its "op" time carried the fused op's per-call dispatch: ~1.3 ms against ~0.5 ms for the plain all-gather op,
and the production model runs under trace. `SOL_BENCH_TRACE=1` now opens the 4x8 mesh with a trace region
(`_ring_8k_trace`), captures the ring op call after the compile run and replays it REPS times with its inputs
still live (the all-gather mode captures its own call); the time per replay is device time plus one synchronize.
`./run_sol_validation.sh trace` runs dense, the gather alone and resident at tau 1/2/3 (no rebuild). It gives
the end-to-end-relevant comparison (dense device ~6.4 expected, resident ~4.7, gather ~2.7) and splits the
untraced 6.1 ms into device time and dispatch.

**Traced, measured (2026-10-23, 5 s, 6 replays):**

| op | traced ms | untraced ms | dispatch | speedup vs dense (traced) |
|---|---|---|---|---|
| dense | 7.20-7.50 (drifting up over the replays), ~7.36 | 7.7 | 0.34 | 1x |
| gather alone | 3.08 | 3.2 | 0.12 | |
| resident tau 1 (20 %) | **4.97** | 6.12 | 1.15 | **1.48x** |
| resident tau 2 (7 %) | 4.03 | 5.17 | 1.14 | 1.83x |
| resident tau 3 (5 %) | 3.40 | 4.52 | 1.12 | 2.16x |

The fused op's per-call dispatch is 1.15 ms against dense's 0.34 and the gather's 0.12 (the 70 KB program and
the per-core tables), which the traced path removes. In the metric the production model sees, resident is at
1.48x at tau 1 and crosses 2x between tau 2 and tau 3. Two times dense traced is 3.68 ms; resident sits 1.9 ms
above the gather's 3.08, so meeting it at tau 1 means hiding ~1.3 of those 1.9 ms: the chain's slowest-member
pacing (1.65x on the ~2 ms of compute) is what stands in the way, and the compute levers alone (pipelining,
pooled phase, wider matmuls: ~0.7-0.8 ms together) reach ~4.2 ms, 1.75x.

**Decision points from the M6 profile (`m6_profile`, fine zones):** (a) S2-S4 time per event sizes item 1;
(b) the S1/S5/S6 split and the pack-thread vs math-thread occupancy size items 2 and 5; (c) the pooled zones
size item 3; (d) RD_CHAIN_RX / RD_SYNC confirm the reader stays hidden as compute shrinks (chain RX ~10 us per
piece is far below the ~37 us of compute per piece today).

So the op is compute-bound end to end: ~37 us of compute per piece (6.6 steps x ~5 us + 3.4 rotations) x 13 pieces x
8 shards + 0.47 ms pooled = 4.3 ms, against an all-gather that finishes at 2.7 ms. The reader never waits for the
gather (RD_SYNC 44 us total). Budget for 2x: M3 at 22-28 us per piece brings compute to ~3.0 ms, i.e. the op
becomes gather-bound at ~3.1-3.3 ms device; at today's 1.36 ms host overhead that is ~4.5 ms host (1.7x), and the
rest of the 2x is the host overhead, which trace capture of the op removes from both sides (dense ~6.4 vs sparse
~3.3 host -> ~1.95x). The pooled phase (0.47 ms, per chunk because the bias row differs per chunk) is the next
item after M3.

Dense-op accuracy (section 8) is fixed in code too (backward all-gather writers' op-signaler count); the nightly
test runs first in the `m2` step.

## 17. Branches for hand-off and M10 (2026-10-23)

**Branches.** `sosborne/sol-attn-q-opt` is the production branch: its head (dc69a2e) removes the debug code from
the M9 state (the `TT_SOL_ABLATE` knobs, the `TT_SOL_RES_CHUNK_MAJOR` bisection switch, the superseded per-piece
step) and documents `sol_row_max` as deprecated (accepted by resident mode, shape-validated, ignored by compute).
No functional change is intended; `./run_sol_validation.sh prod` (correctness set, 5 s parity, traced dense /
gather / tau 1-3) is its acceptance run. **Passed (2026-10-23, 00:43-01:07):** 11/11 (22 Wormhole-only skips),
resident vs compact 0.9999 on every geometry, resident vs dense 0.9996-0.9999 at 512 and 1024 rows (the 1184-row
geometries sit at 0.88 vs dense for compact and resident alike, keep-all included: the known partial-tail
ramp-data artifact of that test shape, and the same values line for line as the M9 log, not the kernel); 5 s parity
0.9996 / 0.9999 / 1.0 / 1.0, resident 10.9 ms vs compact 19.4 at 47 % density; traced dense 7.22-7.44, gather 3.06,
resident tau 1/2/3 = 4.97 / 4.00 / 3.39 ms: identical to the pre-cleanup numbers (section 16). No warnings beyond
the fabric routing-plane and pre-pass matmul-blocking notices every run prints. `sosborne/sol-attn-perf` is the optimization
branch: the production head plus one commit marked DEV ONLY that re-adds the ablation and bisection knobs and
tracks `run_sol_validation.sh`; every optimization goes on top of it, and the DEV ONLY commit is dropped
(revert, or `rebase --onto`) when a result is promoted to production, so the diff to production stays clean.

**M10 = interleaved exp / P@V (written 2026-10-23, unbuilt; `./run_sol_validation.sh m10` after a rebuild).**
Section 15's ablation says the step's stages run one after another and that the exp stage (SFPU exp + packs on
the pack thread, 0.22 ms of op time) and the P@V (math and unpack, ~0.55) load complementary threads. The
compact step now issues the P@V of the members whose last exp unit was packed in S5 batch k right after S5 batch
k+1 (`sol_res_pv_member`, outlined), so with the half-sync DEST the matmuls run while the pack thread is still on
the exp and packs of batch k+1, and the P@V's output packs overlap the next batch's subtracts. Ordering is one
PACK_DONE token per S5 batch that completes a member (posted after its packs; taken by UNPACK before the first
P@V of that batch's members). Finding on the way: `t6_semaphore_post` does not stall at the semaphore's maximum
(a SEMPOST at the max is lost), so PACK_DONE's capacity is a correctness limit, not flow control; resident mode
inits it at 15 (`init_sdpa_streaming_semaphores<15>`) while the streaming paths keep 2. The MOPs are re-issued on
every stage switch (subtract and matmul share the unpacker/math configuration), which is the cost that bounds the
gain.

**M10 first build (2026-10-23, 01:27-01:51):** the correctness set passed 11/11 with PCC values identical to the
production run's (the interleave changes issue order, not arithmetic), the traced dense op was unchanged
(7.22-7.48 ms), and every 5 s run failed at program creation: "Program size (73424) too large for kernel config
buffer (70656)", 2.8 KB over, as M9's first build did. The flush lambda, now carrying the P@V issue, was inlined
at four sites on each TRISC. Fix: the S1 flush, the S5 flush and the P@V issue are outlined (`sol_res_s1_flush`,
`sol_res_s5_flush`, one copy per TRISC) over a per-step state struct (`SolResStepState`); the `m10t` step re-runs
the 5 s parity (the program-size check), the traced timings and the profile without the 21-minute correctness set.
**Second build (01:59-02:02):** the outlining removed 2.4 KB (70992 B) and the 5 s runs were still 336 B over. The
compute kernel was being built at the framework default O3 except under `TT_SOL_STEP_PROFILE`, which set O2; the
factory now sets O2 for every resident build (~30 % smaller TRISC binaries). Every profiled resident run was at O2
and showed the same per-piece compute as the unprofiled O3 timings (the step is bound by Tensix instruction
throughput, not RISC issue), so no step-time cost is expected; the `m10t` traced numbers will say.
**Third build (02:06):** still 70992 B, byte for byte, with the JIT cache reporting 335/335 hits: the kernel build
hash (`ComputeKernel::config_hash`) did not include the optimization level, so the cache served the O3 binary.
Fixed in `tt_metal/impl/kernels/kernel.cpp` (the level is appended to the hash when it differs from the compute
default O3, so existing hashes are unchanged). A host rebuild is needed for both the factory and this fix.

**M10 measured (2026-10-23, 02:11-02:13, O2 build, compute kernel recompiled):** parity 0.9996 / 0.9999 / 1.0 /
1.0, resident 10.73-10.80 ms at 47 % density (10.88 before); traced dense 7.23-7.45 (unchanged), resident tau 1 /
2 / 3 = **4.90-4.94 / 4.02-4.07 / 3.40-3.41 ms** against 4.97 / 4.00 / 3.39: -1 % at tau 1, noise at tau 2 and 3.
The interleave bought nothing measurable, like M5-M9 before it: the step is not limited by which thread the exp
runs on. What the numbers leave: the matmul work at HiFi2 (~1.15 ms op-level), the row-max/lazy-test stats
(~0.4), and above all the chain's slowest-member pacing (iteration 470 us for 265 us of per-core steps, M9
profile), which the gather (437 us per iteration) sits just under. Not promoted to production; the tt_metal hash
fix is a real bug fix and stands on its own.

**M11 = the chain's counting protocol (written 2026-10-23, unbuilt; `./run_sol_validation.sh m11` after a
rebuild).** The chain link gets a second protocol next to the one-outstanding handshake: credits and arrivals as
monotonic counters (`ChainLink::grant`, `wait_arrival`, `forward_credited`; the sender/receiver semaphore words,
never reset within a launch, incremented by remote NoC atomics; the sender waits with `wait_min` on its running
credit count, the receiver on its running arrival count). The resident piece stream grants a credit for every K/V
page it can reserve ahead of its push, in the link's transfer order K(p), V(p), K(p+1), ... (a cursor that never
passes the iteration's last piece; a blocking reserve-and-grant right before a wait if the page was not free
earlier), so with the K/V CBs now 3 deep by default in resident mode the upstream can have two pieces in flight
while this core computes a third; fetchers forward on credits. `TT_SOL_CHAIN_CREDITS=0` is the A/B back to M8
(2 pages). Why it should do what M8 and m8x could not: those kept one transfer outstanding per link whatever the
depth, so a member's slow piece stalled its upstream at once; two pieces of slack per hop absorb one slow piece
before the stall propagates, and the per-hop handshake latency leaves the critical path. Bounded expectation:
the slowest-of-8 factor (1.65 on the ~1.1 ms of mean compute) falls toward ~1.3 and the hop latency hides:
−0.3 to −0.5 ms at tau 1 (4.4 to 4.6 ms, 1.6 to 1.67x). Risk: a hang (every protocol change so far has had one
first run); the 1024-row check runs first because two pieces per iteration is the shape that caught M8's.

**Expected (written before the measurement):** the exp stage's 0.22 ms and part of the P@V packs hidden, minus the per-member MOP switches:
−0.25 to −0.35 ms traced at tau 1, i.e. ~4.65 ms (1.58x) from 4.97. The pooled steps use the same code and gain
in proportion. This does not reach 1.8x (4.09 ms); the remaining compute levers (section 15: the pooled phase and
the stats) are each worth ~0.1-0.2, and the rest is the chain's slowest-member pacing (section 13), which only a
multi-outstanding chain protocol with deeper K/V pages would change (high risk, ≤ ~0.3 ms with 3 pages).
