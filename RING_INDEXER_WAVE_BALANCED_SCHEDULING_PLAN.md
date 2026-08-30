# Ring Indexer Arrival-Wave Load-Balancing Plan

## Status and scope

This plan starts from commit `61f9ff45a617f9640664d571cad3cda2e3def8c2` on
`pjosipovic/ring-indexer-dram-coalescing-multiworker`. It covers one change only: make the fused Ring Indexer
score scheduler distribute runtime-valid K work more evenly without changing the amount or granularity of work.

The proposed mechanism is a deterministic, row-block-preserving rotation of the compute-column assignment at
each ring arrival wave.
It is always enabled for a true bidirectional Ring with `ring_size > 2`, with no config or shape heuristic;
Linear and Ring-2 retain their existing schedule because they do not have paired bidirectional arrival waves.
It is a host-built static schedule. It must not inspect `kv_len`, select among cached programs, or add
device-side arbitration.

## Problem

The factory divides each physical K shard into KC-sized units and deals every shard from lane zero. A runtime
KV prefix skips units that are outside `kv_len`, but it does not rebuild the capacity-sized program. When the
number of valid units per shard is not divisible by the number of block-column lanes, the remainder therefore
lands on the same lanes for every shard. Those lanes determine the critical path while other lanes finish early.

For the current GLM-5.2 configuration, KC is 10 tiles (320 keys). A lane is one `(row block, compute column)`
pair, so QuietBox has 20 lanes and the measured LoudBox has 22:

| Case | Ring | Valid units per shard | Lanes | Current maximum / mean lane work |
| --- | ---: | ---: | ---: | ---: |
| QB 55K | 4 | 44 | 20 | 1.364 |
| QB 512K | 4 | 410 | 20 | 1.024 |
| LB 55K | 8 | 22 | 22 | 1.000 |
| LB 512K | 8 | 205 | 22 | 1.073 |
| LB 58,880 diagnostic | 8 | 23 | 22 | 1.913 |

This explains why QB 55K is substantially below the ideal-scheduling FPU model while LB 55K happens to be a
perfectly divisible case. The FPU model should continue to assume perfect work scheduling: that is the north
star this work is intended to approach.

## Goals

1. Improve QB Ring-4 55K scheduling and performance:
   - reduce the analytical maximum/mean lane-work ratio from 1.364 to at most 1.14;
   - improve median warm trace-replay FPU utilization by at least 10% relative, from the checked-in 46.26% to
     at least 50.9%; equivalently, reduce traced fused-program latency by at least 9.1% with the ideal work
     numerator held fixed.
2. Demonstrate that the mechanism also helps LB rather than merely fitting the QB geometry:
   - measure 58,880 keys as an A/B-only diagnostic LB point (23 valid units per shard over 22 lanes). This is
     46 proxy global slabs of `8 * 160` tokens and is valid for the 8x1 proxy with
     `chunk_start = kv_len - q_rows`; it is not a production 5,120-token chunk boundary;
   - reduce its analytical maximum/mean ratio from 1.913 to at most 1.23;
   - reduce its median warm trace-replay fused-program latency by at least 10% versus the unrotated scheduler.
3. Preserve performance where scheduling is already balanced or the tail is small:
   - no more than 2% median latency regression at QB 512K, LB 55K, or LB 512K;
   - no more than 2% median latency regression at any point in the QB/LB prefix sweep described below;
   - preserve both the default Fabric packet configuration and the 14 KiB configuration within the same 2%
     no-regression limit at the existing 55K and 512K points.
4. Preserve dynamic-prefix trace behavior:
   - all supported `kv_len` values for a fixed tensor capacity and program configuration must hit the same
     program-cache entry;
   - rotation must depend only on compile-time/cached geometry: ring size, arrival wave, KC, capacity, and the
     compute grid. It must never depend on runtime `kv_len` or `chunk_start_idx`.
5. Preserve correctness and readiness overlap:
   - every KC unit appears exactly once, stays inside one physical shard, and remains KC-aligned;
   - the local shard remains the first arrival wave;
   - the two shards arriving in the same bidirectional wave use the same rotation and retain the interleaved
     `A0, B0, A1, B1, ...` visit order;
   - first-half units continue to wait on the midpoint marker, and units crossing or following the midpoint
     continue to wait on shard completion;
   - every test scheduled in CI from
     `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` remains green without weakening its correctness or
     performance thresholds.
6. Make the schedule robust for a future Ring-32 deployment:
   - the algorithm must have no Ring-4/8 cases, lookup tables, or assumptions about the number of waves;
   - exhaustively simulate Ring-32 schedule coverage and load balance for every tile-aligned prefix up to the
     supported capacity, every local ring rank, and the relevant 20- and 22-lane geometries;
   - construct each rank's waves with the production `ring_writes_for` and `RingIdSequencer` rules, and compute
     prefix validity with the exact block-cyclic logical-tile mapping rather than assuming equal valid units in
     every shard;
   - for every simulated prefix, its maximum per-lane count of nonempty KC units must be no worse than the
     unrotated schedule, and its valid-K-tile distribution must also be reported;
   - preserve paired-wave ordering across all 17 Ring-32 arrival waves.
7. Require independent final approval of the implementation:
   - after correctness and performance acceptance passes, have a fresh Codex Sol instance and Claude Fable
     independently review the actual code and tests;
   - each reviewer must assess both whether the implementation achieves this plan's correctness, trace-stability,
     Ring-32, and measured performance goals, and whether there are remaining cleanup, simplification, or
     refactoring opportunities;
   - address actionable feedback, rerun validation in proportion to the resulting changes, and re-request review
     until both reviewers explicitly approve the final revision.

Ring-32 is a design and analytical acceptance goal in this environment, not a claim of hardware validation.
The change is not complete for production Ring-32 enablement until the same correctness, cache-reuse, and
performance tests run on a 32-device system.

## Non-goals

- Do not change the FPU performance model to price in imperfect scheduling.
- Do not change KC, slab boundaries, Fabric packet size, DRAM coalescing, partial-readiness markers, worker-core
  placement, or CCL routing.
- Do not add a runtime `kv_len` heuristic, a program variant, a new runtime argument, or a device work queue.
- Do not change the existing Linear or Ring-2 work assignment; arrival-wave rotation is specific to a true
  bidirectional Ring with more than two ranks.
- Do not tune CI targets downward. Raise a target only after a stable measured improvement, placing it at the
  midpoint of the observed range with the existing symmetric margin.
- Do not claim Ring-32 performance or correctness from simulation alone.

## Proposed schedule

Treat the columns in each row block as a cyclic space, applying the same rotation to every row block:

```text
lane_count = num_blocks * cols_used
lane       = block + column * num_blocks
wave_count = max_wave + 1
column_stride = max(1, floor(cols_used / (wave_count + 1)))
column_shift = (wave * column_stride) % cols_used
source_column = (column + cols_used - column_shift) % cols_used
source_lane = block + source_column * num_blocks
unit_in_shard = source_lane + round * lane_count
```

The destination remains `(block, column)`; only the source-unit residue assigned to it changes. Because
`source_column -> column` is a cyclic permutation within each row block, a wave still assigns every unit exactly
once. Every shard in a wave uses the same `column_shift`, so paired forward/backward shards retain matching
offsets and their existing fine-grained readiness-friendly order. Keeping the row-block parity fixed also
preserves the adjacent-block DRAM access pairing of the existing schedule; a preliminary full-lane rotation
that allowed odd shifts regressed LB 512K from about 60.0% to 58.0% FPU utilization and was rejected.

The bounded stride uses only cached geometry and deliberately minimizes displacement while still moving each
early arrival wave to a new residue. Expressed as lane shifts, Ring-4 with 20 lanes uses 0, 4, and 8; Ring-8
with 22 lanes uses 0, 2, 4, 6, and 8. At QB 55K this changes the expected
maximum/mean work ratio from 1.364 to 1.136. At the LB 58,880 diagnostic it
changes 1.913 to at most 1.196. The exact figures must be reproduced by the implementation-side simulator,
not trusted as hand-maintained constants.

The smaller stride is performance-driven but geometry-generic. A wider evenly-spaced Ring-8 rotation preserved
the same analytical tail balance but regressed the seven-replay LB 512K median by 3.3% (1.867 to 1.929 ms).
The bounded stride retained the 58,880 gain while keeping LB 512K within 0.9% of the unrotated schedule.

The local wave deliberately has shift zero. This keeps startup behavior unchanged and rotates only subsequent
waves whose data is Fabric-gated.

## Implementation steps

### 1. Add an executable schedule model before changing the factory

Add a small host-only test/helper that generates `shards_by_wave`, lane shifts, and per-lane physical starts.
Keep it independent of device execution so it can exhaustively check many prefixes and Ring-32.

For every tested geometry, assert:

- exact unit coverage with no duplicate or missing `(shard, unit)` pair;
- all physical starts are KC-aligned and less than the end of their shard;
- same-wave shards have identical unit offsets and remain adjacent in visit order;
- source-unit offsets are strictly increasing for every `(destination lane, wave, shard)`, so no first-half unit
  can appear after a unit that requires the shard-completion marker;
- local-wave shift is zero;
- the rotated maximum lane load is no greater than the unrotated maximum for every simulated prefix. Assert and
  report both nonempty KC units and valid K tiles. Nonempty KC units are the primary execution-load metric
  because the current reader/compute path pays a full KC for every nonempty boundary unit; valid K tiles are the
  semantic-volume and ideal-FPU-model metric.

Prefer a small pure helper near the existing factory scheduling code. Do not expose a new public API solely for
the test; if direct C++ unit coverage would require disproportionate plumbing, keep exhaustive schedule
validation as a checked-in Python host test that implements and documents the same integer formula, and retain
simple host assertions in the factory.

### 2. Rotate the factory's columns while preserving row blocks

Update `ring_indexer_score_dsa_program_factory.cpp` so `work_list[block][column]` uses the formula above when
the topology is Ring and `ring_size > 2`. Rotate the column residue identically in every row block; this moves
the repeated shard tail across the complete set of block-column lanes while preserving the current pairing of
adjacent row-block residues. Keep the current mapping unchanged for Linear and Ring-2.

Keep the existing outer arrival-wave order and emit all shards in `shards_by_wave[wave]` for a unit before
advancing to the next unit. Do not change reader, compute, writer, or all-gather kernel ABIs.

Name the lane/shift variables and document the permutation invariant. Avoid topology-specific constants and
avoid introducing a generic scheduling abstraction larger than this operation needs.

Document two structural properties rather than implementing special cases: wave zero has shift zero directly
from the formula, and a rotated lane cannot receive more than one remainder contribution from any shard, so its
maximum nonempty-unit load cannot exceed the unrotated maximum, where every shard's remainder starts on the
same lanes. Use one shift for every shard in a wave because their interleaved offsets must retain the same
midpoint-versus-completion gate relationship.

### 3. Verify correctness and program-cache stability

Run the full Ring Indexer correctness suites on Ring-4 and Ring-8, including nonzero/random K data, sampled
reference comparison, both ring directions, partial first/second shard halves, and at least two runtime KV
prefixes through one cached program.

Add or extend a test only if the existing Ring-8 partial-readiness/cache-hit test cannot distinguish a rotated
unit order from the current order. The test must check output semantics, not merely completion.

Run all CI-selected cases in `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`. No test exclusions, target
reductions, or relaxed tolerances are permitted to make this change pass.

### 4. Measure QB and LB

First collect an A/B baseline from the same commit, runner, packet configuration, and traced execution path by
toggling only lane rotation. Use at least seven measured warm replays per point and compare medians; repeat any
point within 1% of a pass/fail boundary in three fresh process/device sessions.

Measure:

- QB Ring-4: 55K and 512K, plus a prefix sweep;
- LB Ring-8: 55K, 58,880, and 512K, plus a prefix sweep;
- default and 14 KiB Fabric packet configurations at the existing 55K/512K endpoints.

The diagnostic prefix sweep should cover at least one value for every valid-unit remainder modulo the lane
count, plus the production endpoints. Generate these prefixes from the geometry rather than hand-picking only
favorable points. Record fused-program duration, ideal-model FPU utilization, lane count, valid units per shard,
remainder, and analytical maximum/mean lane load in both nonempty KC units and valid K tiles. Also record both
forms of per-lane work remaining after each wave arrival. The latter is a secondary arrival-timing diagnostic:
total-work balance alone does not guarantee improvement if additional work is assigned to a late Fabric wave.

The default-packet guard needs a distinct device setup because the checked-in perf test currently always passes
a 14 KiB `FabricRouterConfig`. Measure it either by parametrizing `device_params` over `{default, 14 KiB}` or by
a documented A/B run with `fabric_router_config` omitted, and record which path was used. Treat default-packet
numbers as A/B guards without new checked-in expected values unless that parametrization is intentionally added
to CI.

After the measurements pass, update `RING_INDEXER_EXPECTED_FPU_UTIL` in the same implementation change for
every existing point whose improved median falls outside its symmetric +/-2% band. Place each new target at the
midpoint of the stable observed range and never lower a target. The improvement would otherwise correctly fail
the old test's upper bound. Keep 58,880 A/B-only by default; if it is intentionally added to the checked-in perf
parametrization, add its post-change `(8, 58880)` expected value at the same time.

Accept the implementation only if every numerical goal above is met. If the analytical ratios improve but the
LB diagnostic or QB 55K measured goal does not, collect per-core timing/load evidence before considering a more
complex scheduler; do not retain complexity justified only by the analytical model.

### 5. Validate Ring-32 analytically and prepare hardware handoff

For Ring-32, simulate all 17 arrival waves for:

- 20 and 22 block-column lanes;
- every tile-aligned runtime prefix through the current maximum cache capacity;
- every `device_index` from 0 through 31, building its exact arrival waves from the same forward/backward write
  counts and `RingIdSequencer` order as the factory;
- exact wave endpoint cases, partial final KC units, lane-count divisibility boundaries, and the known
  block-cyclic chunk geometry;
- prefixes ending inside a global block-cyclic slab, using the same `logical_tile()`/`k_tiles()` rule as
  `ShardMajorWorkUnitSpan`, so rank-dependent unequal shard tails are included.

Produce a compact table containing worst maximum/mean load, the prefix where it occurs, coverage status, and
comparison with the unrotated schedule, reporting the worst local rank rather than one representative rank.
Document the unvalidated hardware matrix for the future Galaxy run: correctness with nonzero K,
one-cache-entry dynamic prefixes, 55K/512K performance, a remainder sweep, both Fabric packet configurations,
and semaphore/readiness progress under all 17 waves.

### 6. Run independent final code reviews

Only after the implementation has passed the correctness, cache-stability, analytical, and hardware-performance
acceptance above, send the final diff and supporting measurements independently to:

- a fresh Codex Sol instance; and
- Claude Fable in a separate review session.

Ask both reviewers for an explicit `APPROVE` or actionable findings in two distinct areas:

1. **Goal and correctness review:** exact work coverage, runtime-prefix behavior, paired readiness ordering,
   program-cache stability, Ring-4/8 correctness, Ring-32 analytical robustness, performance methodology, and
   whether every numerical acceptance goal in this plan was actually met without weakening another test.
2. **Code-quality review:** unnecessary abstractions or special cases, duplicated schedule arithmetic, cryptic
   naming or comments, avoidable branches, opportunities to make the permutation and invariants easier to audit,
   test complexity, and any safe cleanup, simplification, or refactoring that preserves performance.

Reviewers must inspect the implementation and tests, not only this plan or a prose summary. Provide the exact
commit, full diff, validation results, and raw A/B performance table. Keep the reviews independent: do not give
one reviewer the other reviewer's conclusions before both initial reviews are complete.

Address all accepted findings, rerun affected build/correctness/performance checks, and submit the revised code
to both reviewers again. The work is ready to merge only when both independently approve the same final code
revision. A review that says the implementation is correct but leaves actionable cleanup, simplification, or
refactoring findings unresolved does not satisfy this gate.

## Required validation commands

The implementation change will require, at minimum:

```bash
cmake --build build --target ttnn -j32
pre-commit run --files <changed-files>
scripts/run_safe_pytest.sh \
    tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_ring_indexer_score_dsa.py -s
scripts/run_safe_pytest.sh \
    tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_ring_indexer_score_dsa_4d.py -s
scripts/run_safe_pytest.sh \
    tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_ring_indexer_score_dsa_perf.py -s
```

Also run the Blackhole CI selection that exercises `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`. Record
hardware type, physical compute grid, packet configuration, commit, individual replay durations, median, and
program-cache entry count with every performance result.

## Stop/rollback criteria

Stop and revert the scheduler change if any of the following remains after root-cause analysis:

- output mismatch, hang, or semaphore/readiness ordering failure;
- a `kv_len` change creates another cached program;
- any required performance point or prefix-sweep point regresses by more than 2%;
- QB 55K or LB 58,880 misses its measured improvement goal;
- Ring-32 simulation finds a duplicate/missing unit, breaks paired-wave alignment, or produces a worse maximum
  nonempty-KC-unit lane load than the unrotated schedule;
- implementation requires a kernel ABI change or runtime strategy selection to obtain the gain.
- either independent final reviewer has unresolved correctness, goal-completion, cleanup, simplification, or
  refactoring findings, or has not explicitly approved the final code revision.

If the simple rotation passes correctness but fails the performance goals, preserve the measurements and return
to design. Do not compensate by changing KC, reducing CI targets, or teaching the FPU model about the imbalance.
