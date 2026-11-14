---
name: perf-test
description: Create, unify, repair, or review an LLK perf test on Wormhole, Blackhole, or Quasar so it measures its functional test - one kernel source, one Python runner, one shared sweep, a perf L1_TO_L1 that is the functional run, and isolate run types that are slices of L1_TO_L1. Use when adding perf_[op].py or perf_[op]_quasar.py, merging a separate *_perf.cpp into the functional kernel, implementing or debugging UNPACK_ISOLATE / MATH_ISOLATE / PACK_ISOLATE / SFPU_ISOLATE, fixing a perf hang or dvalid mock count, choosing Quasar PERF_TILE_SIZES or dest-fill matrices, or reviewing a perf test for drift from functional.
user_invocable: true
---

# Perf Test

## Goal

A perf test measures the functional test, not a separate program. One kernel
source and one Python runner serve both. Perf `L1_TO_L1` is the functional run
with `LOOP_FACTOR` repetitions. Each isolate run type runs one thread's
`L1_TO_L1` loop unchanged and replaces only the threads it does not measure.

This skill is the architecture-neutral contract. Read the reference for the
target architecture before editing:

- [references/wormhole-blackhole.md](references/wormhole-blackhole.md):
  MOP stream stand-ins, handshake counts, zeroed-SrcB paths.
- [references/quasar.md](references/quasar.md): the Quasar harness, tile and
  dimension selection, wait masks, Quasar stand-in counts.
- [references/perf-facts.md](references/perf-facts.md): run types, build
  kinds, CSV normalization, schema and `tile_cnt` accounting, report paths,
  determinism, CI. Every perf skill links here.

Related skills, in the order work usually needs them:

- `quasar-test-coverage`: choose the Quasar functional sweep. Perf inherits
  it, so its size sets the perf build count.
- `perf-coverage`: prove the perf sweep matches functional, or line up
  Blackhole and Quasar.
- `run-test`: run the test (`--perf` for perf sweeps).
- `perf-report`: produce, validate, and record a report.
- `perf-regression-check`: compare against the branch point.
- `perf-parameter-impact`: analyze a validated `.post.csv`.
- `perf-optimization-audit`: audit the kernel's own instruction costs.

## Choose the workflow

1. No perf test yet: **Inventory** (to read the functional test), then
   **Python**, **Kernel**, **Validate**.
2. A separate perf kernel or sweep exists: **Inventory** the drift, then fix
   each item.
3. A run type hangs or reports implausible numbers: **Repair**.
4. Tile sizes, dimensions or other sweep axes change: **Python**, then
   `perf-coverage`.

Finish every workflow with **Validate**.

## 1. Inventory the drift

Read the functional test, the perf test, and every kernel source they build.
List each difference under one heading below. Each one is a defect unless the
**Python** rules allow it.

Run the coverage comparer (`perf-coverage`) too, but do not rely on it
alone: it reads only `@parametrize`. Values set inside the test body
(`input_dimensions=[64, 64]`), runtimes, and run types fixed inside a
`PerfConfig` call are invisible to it.

| Area | Typical drift |
|---|---|
| Sources | a separate `[op]_perf.cpp` next to `[op]_test.cpp` |
| Sweep | perf-only values or axes, a different format set, hand-picked block shapes instead of the functional `input_dimensions` |
| Python checks | perf-only rejections (`perf does not support tilize`, `perf requires dest_index=0`) |
| Runtimes | perf omits a runtime functional emits (`NUM_FACES`, `NUM_BLOCKS`), or passes a different value |
| Run types | perf measures a subset (for example only `PACK_ISOLATE`), or a `static_assert` forbids one |
| INIT | a different init order, `PackMode`, tensor shape or literal (`num_faces = 4`) per run type; "prefill" work in isolate INIT |
| Addresses | any run type uses the `PERF_ADDRESS` ring or hard-coded L1 addresses instead of `buffer_A` / `buffer_Res` |
| Loop shape | isolates iterate differently from `L1_TO_L1`: other block sizes, `MAX_TILES_DEST` chunks instead of `NUM_BLOCKS × NUM_TILES_IN_BLOCK`, a hard-coded dst index |
| Parameters | the kernel picks values by build kind, such as `#ifdef LLK_PROFILER` stand-in zeros |

## 2. Python: one sweep, one runner

In the functional module:

```python
# Shared with the perf module so the two sweeps stay aligned.
OP_SWEEP = dict(formats=..., dest_acc=..., input_dimensions=[...])


def _run_op_test(<sweep args>, *, is_perf=False, perf_report=None, run_types=None, loop_factor=1):
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")
    if run_types is None:
        run_types = [PerfRunType.L1_TO_L1]
    # stimuli, golden, block split: computed the same way for both
    configuration = create_test_or_perf_config(
        is_perf=is_perf, run_types=run_types, test_config_kwargs={...}
    )
    if is_perf:
        configuration.run(perf_report)
        return
    # functional-only: result setup, collection, golden check
```

The perf module adds only measurement axes:

```python
@pytest.mark.perf
@parametrize(**OP_SWEEP, run_types=[ALL_PERF_RUN_TYPES], loop_factor=[32])
def test_perf_op(perf_report, <sweep args>, run_types, loop_factor):
    _run_op_test(<sweep args>, is_perf=True, perf_report=perf_report,
                 run_types=run_types, loop_factor=loop_factor)
```

Quasar has its own established shape for this (`_PERF_AXES`,
`PERF_RUN_TYPES_QUASAR`, `@pytest.mark.quasar`); see
[references/quasar.md](references/quasar.md).

Rules:

- Stimuli, golden, block split (`get_num_blocks_and_num_tiles_in_block`),
  templates and runtimes are built in one place. Perf never builds its own
  config.
- Emit every template and runtime the kernel reads, in both modes. If perf
  lacks a field such as `INPUT_DIMENSIONS`, pass it; do not make the kernel
  guess.
- No perf-only rejections. Reject a combination only when the hardware or the
  LLK cannot run it, and then reject it for functional too.
- Functional-only steps stay after the `is_perf` return: zero-filling the
  result for L1 accumulation, collecting results, the golden check.
- Narrow perf only as a documented subset: copy the sweep and filter an axis
  (`perf-coverage`, "Perf-only packaging"), or use the architecture's
  documented perf generator (Quasar tile shapes). Never add values that
  functional does not have.
- When perf sweeps values functional lacks (block shapes, `num_faces`),
  widen functional first, with a golden for each new value, and then share
  the sweep. Do not shrink perf to functional's smaller set without saying
  which perf points were dropped.
- Runtime values come from the same code in both modes. A value derived from
  the golden (a ReLU threshold, a clamp) is computed for perf too, so perf
  measures the configuration functional checks.
- `run_types` covers every stage the pipeline has. Leave one out only when the
  stage does not exist (unpack-to-dest without math).
- One `loop_factor` is enough. Use 32 (the convention) unless the module
  already publishes another value: changing it moves every published number,
  so do it in a separate, stated change. Each extra value multiplies the
  builds by the number of run types.
- Functional-only regression tests, such as a forced block shape, may stay
  unmatched. Name them in the PR.

## 3. Kernel: same INIT, same buffers, isolates as slices

Merge `[op]_perf.cpp` into `[op]_test.cpp` and delete it. Include `counters.h`,
`perf.h` and `profiler.h`. Open the setup scope with
`START_PERF_MEASURE("INIT")` and the repeated work with
`START_PERF_MEASURE("TILE_LOOP")`, each ending with `PROFILER_SYNC()`. The
macro opens the profiler zone and, in counter builds, the counter window; a
bare `ZONE_SCOPED` loses the counters. Branch with
`if constexpr (PERF_RUN_TYPE == ...)`.

**INIT.** One INIT sequence for every run type: the functional order, the
functional `PackMode`, the same runtime values, and no prefill. An isolate
thread with no partner neither needs nor gets real Dest or source contents.
The only allowed per-run-type difference is handshake setup: a thread that
runs without its partner clears or skips the Dest dvalid wiring that would
wait for it (see the architecture reference).

**Buffers.** Every run type, `L1_TO_L1` included, reads `buffer_A` (and
`buffer_B`) and writes `buffer_Res`, with the functional tile indexing. Do
not use `PERF_ADDRESS` or hard-coded L1 addresses; they change the L1
traffic you measure.

**Loop bodies.** Put each thread's per-block work in a helper (identical
inline copies drift the first time one is edited). `L1_TO_L1` and that
thread's isolate call the same helper with the same `NUM_BLOCKS`,
`NUM_TILES_IN_BLOCK`, `DST_INDEX` and addresses, and repeat it `LOOP_FACTOR`
times. The two branches then differ only in sync, plus the Dest bookkeeping
that pack's section done carries (below the table):

| Run type | Unpack | Math | Pack |
|---|---|---|---|
| `L1_TO_L1` | real loop | wait for Dest → block helper → section done | wait for math → block helper → section done |
| `UNPACK_ISOLATE` | real loop | stand-in: clear the source dvalids unpack posts | empty |
| `MATH_ISOLATE` | stand-in: post the source dvalids math consumes | block helper, no Dest sync | empty |
| `PACK_ISOLATE` | empty | empty | block helper, no math wait, no section done |

Adapt the table to the real pipeline: an unpack-to-dest kernel may bypass
math, and a 4-TRISC Quasar kernel adds an SFPU column.

On Wormhole and Blackhole, `_llk_pack_dest_section_done_` is more than the
math semaphore: it also stalls on pack, clears the packed Dest half
(`ZEROACC`) and flips the packer's Dest bank. Without it, `PACK_ISOLATE`
skips the Dest clear and always packs bank 0, so part of the gap to
`L1_TO_L1` is that bookkeeping, not handshake cost.

- Idle branches are empty and fall through to `PROFILER_SYNC()`; no early
  `return` inside the zone. Keep the same `if constexpr` chain shape in every
  thread.
- Uninit calls (`_llk_unpack_tilize_uninit_...`) go after the `TILE_LOOP`
  zone, for every run type.
- Options such as tilize work in every run type. A `static_assert` that a
  feature is functional-only is drift.

### Source dvalid accounting

The stand-in count must match what the real partner does, exactly:

```text
stand-in count =
    LOOP_FACTOR
  × real operation invocations per loop
  × dvalid pulses per invocation
```

- Count the tiles and blocks actually consumed.
- Determine from the MOP whether the real path pulses per tile or per face,
  and whether math consumes SrcA, SrcB, or both.
- Unpack's and math's counts must agree, or the threads deadlock.
- Use the shared count helper where one exists
  (`_perf_src_handshake_iters_` in `tests/helpers/include/llk_lib_unpack_wrappers.h`
  on Wormhole and Blackhole) and the stand-in helpers named in the
  architecture reference. Count from the MOP code, not its comments.
- Do not shrink `TILE_CNT` to change the published per-tile number
  ([references/perf-facts.md](references/perf-facts.md), "Accounting").
- Never add a synthetic SET/CLEAR pair to flush the pipeline. An extra token
  can race the final operation and pollute a later run.

### Destination dvalid rules

Signal completion only to an active consumer:

- `L1_TO_L1`: keep the real producer-to-pack setup and section-done calls.
- `UNPACK_ISOLATE` and `MATH_ISOLATE`: do not signal the idle pack.
- `PACK_ISOLATE`: pack without waiting for math and without Dest section
  done, which also drops its Dest clear and bank flip (see the run-type
  table).

Architecture-specific state (Quasar wait masks, `set_up_dest_dvalid_per_thread`)
is in the references.

## 4. Parameters that compile in every build

`run_kernel` is not a template, so a discarded `if constexpr` branch must
still compile. Every name it uses must exist in every build.

```cpp
#ifndef SPEED_OF_LIGHT
    const std::uint32_t block_ct_dim = params.BLOCK_CT_DIM;   // runtime struct field
#else
    constexpr std::uint32_t block_ct_dim = BLOCK_CT_DIM;       // generated constant
#endif
```

- Select only on `SPEED_OF_LIGHT` (runtime field versus generated constant).
- Never select parameter values on `LLK_PROFILER`
  ([references/perf-facts.md](references/perf-facts.md), "Build kinds"). A
  stand-in there silently returns wrong values when a config changes and
  hides untested code.
- If a field is missing in one mode, fix the Python side, not the kernel.
- Use sources of truth such as `get_dest_max_tiles<...>()` and
  `TILE_WIDTH * TILE_HEIGHT`. Label literal arguments (`false /*untilize*/`)
  and check each label against the callee's parameter name.
- Use the target architecture's current LLK signatures. Do not copy calls
  from another architecture's or an older perf kernel.

## 5. No dead paths

A kernel branch that no sweep value reaches is untested, and untested code
hides bugs. In the pack dest-bank test, tilize was never swept, and its read
offset stepped by `BLOCK_RT_DIM` instead of `BLOCK_CT_DIM`. Either sweep the
option in functional (and so in perf), with the matching golden
(`TilizeGolden`), or delete the branch.

## 6. Perf schema

Adding a template or runtime field adds CSV columns. Update the module's
entry in `helpers/perf/test_schemas.py` and bump its `version` once per PR,
relative to main. The hardware-free header gate fails on any drift.

## 7. Repair a run type

1. Read the kernel, both Python modules, and the latest
   `perf_data/latest/<module>/<module>.post.csv`.
2. Map every producer and consumer handshake: SrcA and SrcB dvalids between
   unpack and math, Dest dvalids between unpack or math and pack, and any
   semaphore-based unpack-to-dest sync.
3. Inspect the LLK MOP when pulse counts are not explicit, and derive the
   exact count for one invocation. Do not copy counts blindly.
4. Fix the branch. Compile a focused sweep (`run-test --k`), then run all run
   types together: sticky state shows only when one run type follows another.
5. Check the numbers against the plausibility list in `perf-report`
   ("Validate the artifact"). A hang means a stand-in count or a sync step is
   wrong. Values near 2048, 4096 or 8192, or a healthy first variant followed
   by slow ones, mean a handshake, section-done or wait-mask bug.
6. Run the functional test, including unpack-to-dest and Dest-accumulation
   variants when they exist.
7. Restore the full sweep after focused debugging.

Do not change LLK library code before ruling out a test-kernel orchestration
bug.

## 8. Validate

Run device and simulator tests through `run-test`; perf sweeps take `--perf`.

1. **Compile** every architecture the test runs on, normal and
   `--speed-of-light`, functional and perf.
2. **Functional** on hardware or the simulator, both modes.
3. **Perf**, all run types, through `run-test --perf`.
4. **Plausibility and determinism**: `perf-report`, "Validate the artifact".
5. **Coverage**: `perf-coverage`. Expect every coverage axis identical and
   only measurement axes ignored.
6. **Host-only suite**, which includes the schema header gate:
   `tests/run_hardware_free.sh`.
7. **Against main**: `perf-regression-check`. When the sweep or run types
   changed, few points match. Read deltas with its "Layout sensitivity"
   section before calling anything a regression.
8. **CI cost**: when the build count changes a lot, update the split costs
   ([references/perf-facts.md](references/perf-facts.md), "CI").

## PR description

State:

- which drift was removed;
- the isolate model (the table above, as implemented);
- any intentionally unmatched functional tests;
- the schema version change;
- the validation actually run, per architecture, and what was not measured
  (for example Blackhole on a Wormhole-only host).

## Do not

- keep a second kernel source for perf;
- add perf-only sweep values, rejections, stimuli or init steps;
- give an isolate a different loop structure, address set or dst index than
  `L1_TO_L1`;
- choose parameter values on `LLK_PROFILER`;
- keep a feature branch that no sweep exercises;
- weaken functional assertions;
- narrow the permanent sweep to make a run finish;
- declare success from pytest alone without the plausibility checks.
