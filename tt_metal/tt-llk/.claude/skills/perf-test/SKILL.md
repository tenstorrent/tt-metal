---
name: perf-test
description: Create, unify, or repair a Wormhole/Blackhole LLK perf test so it matches its functional test - one kernel source, one Python runner, one shared sweep, a perf L1_TO_L1 that is the functional run, and isolate run types that are slices of L1_TO_L1. Use when adding perf_[op].py, merging a separate *_perf.cpp into the functional kernel, aligning a perf sweep with its functional sweep, implementing UNPACK_ISOLATE / MATH_ISOLATE / PACK_ISOLATE, or reviewing a perf test for drift from functional.
user_invocable: true
---

# Perf Test (aligned with functional)

## Goal

A perf test measures the functional test, not a separate program. One kernel
source and one Python runner serve both. Perf `L1_TO_L1` is the functional run
with `LOOP_FACTOR` repetitions. Each isolate run type runs one thread's
`L1_TO_L1` loop unchanged and replaces only the threads it does not measure.

Reference implementations: `tests/sources/eltwise_unary_datacopy_test.cpp` with
`test_eltwise_unary_datacopy.py` / `perf_eltwise_unary_datacopy.py`, and, after
#63744, `tests/sources/pack_dest_bank_test.cpp` with `test_pack_dest_bank.py` /
`perf_pack_dest_bank.py`.

Related skills:

- `quasar-perf-test` — Quasar specifics: tile-size selection, dvalid wait
  masks, Quasar LLK signatures. The alignment contract below applies there too.
- `align-bh-qsr-tests` — the coverage comparer, and the cross-architecture
  alignment.
- `perf-report` — run a sweep and validate the CSV.
- `perf-regression-check` — compare against the branch point and read the
  deltas.

## 1. Inventory the drift

Read the functional test, the perf test, and every kernel source they build.
List each difference under one of these headings. Each one is a defect unless
it is in the allowed list in step 2.

| Area | Typical drift |
|---|---|
| Sources | a separate `[op]_perf.cpp` next to `[op]_test.cpp` |
| Sweep | perf-only values or axes, a different format set, hand-picked block shapes instead of the functional `input_dimensions` |
| Python checks | perf-only rejections (`perf does not support tilize`, `perf requires dest_index=0`) |
| Run types | perf measures a subset (for example only `PACK_ISOLATE`) |
| INIT | a different init order, or a different `PackMode`, per run type; extra "prefill" work in isolate INIT |
| Addresses | isolates use the `PERF_ADDRESS` ring while `L1_TO_L1` uses `buffer_A` / `buffer_Res` |
| Loop shape | isolate loops iterate differently from `L1_TO_L1`: other block sizes, `MAX_TILES_DEST` chunks instead of `NUM_BLOCKS × NUM_TILES_IN_BLOCK`, a hard-coded dst index |
| Parameters | the kernel picks values by build kind, such as `#ifdef LLK_PROFILER` stand-in zeros |

## 2. Python: one sweep, one runner

In `test_[op].py`:

```python
# Shared with perf_[op].py so the two sweeps stay aligned.
OP_SWEEP = dict(formats=..., dest_acc=..., tilize=get_valid_tilize_..., input_dimensions=[...])


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


@parametrize(**OP_SWEEP)
def test_op(<sweep args>):
    _run_op_test(<sweep args>)
```

In `perf_[op].py`, add only measurement axes:

```python
@pytest.mark.perf
@parametrize(**OP_SWEEP, run_types=[ALL_PERF_RUN_TYPES], loop_factor=[32])
def test_perf_op(perf_report, <sweep args>, run_types, loop_factor):
    _run_op_test(<sweep args>, is_perf=True, perf_report=perf_report,
                 run_types=run_types, loop_factor=loop_factor)
```

Rules:

- Stimuli, golden, block split (`get_num_blocks_and_num_tiles_in_block`),
  templates and runtimes are built in one place. Perf never builds its own
  config.
- Emit every template and runtime the kernel reads, in both modes. If perf
  lacks a field such as `INPUT_DIMENSIONS`, pass it. Do not make the kernel
  guess.
- No perf-only rejections. Reject a combination only when the hardware or the
  LLK cannot run it, and then reject it for functional too.
- Functional-only steps stay after the `is_perf` return: zero-filling the
  result for L1 accumulation, collecting results, the golden check.
- Narrowing perf is allowed only as a documented subset: copy the sweep and
  filter an axis (see `align-bh-qsr-tests`, "Perf-only packaging"). Never add
  values that functional does not have.
- `run_types` defaults to `ALL_PERF_RUN_TYPES`. Leave a run type out only when
  the pipeline does not have that stage (unpack-to-dest without math).
- `loop_factor`: one value is enough (32 is the convention). Each extra value
  multiplies the builds by the number of run types.
- Functional-only regression tests, such as a forced block shape, may stay
  unmatched. Name them in the PR.

## 3. Kernel: same INIT, same buffers, isolates as slices

Merge `[op]_perf.cpp` into `[op]_test.cpp` and delete it.

**INIT.** One INIT sequence for every run type: the functional order, the
functional `PackMode`, and no prefill. An isolate thread with no partner
neither needs nor gets real Dest or source contents.

**Buffers.** Every run type reads `buffer_A` (and `buffer_B`) and writes
`buffer_Res`, with the functional tile indexing. Do not switch to
`PERF_ADDRESS` for some run types. It changes the L1 traffic you are
measuring.

**Loop bodies.** Put each thread's per-block work in a helper. `L1_TO_L1` and
that thread's isolate call the same helper with the same `NUM_BLOCKS`,
`NUM_TILES_IN_BLOCK`, `DST_INDEX` and addresses. The two branches can then
differ only in sync:

| Run type | Unpack | Math | Pack |
|---|---|---|---|
| `L1_TO_L1` | real loop | `wait_for_dest_available` → block helper → `dest_section_done` | `wait_for_math_done` → block helper → `dest_section_done` |
| `UNPACK_ISOLATE` | real loop | stand-in: `_perf_math_stream_clear_valid` | empty |
| `MATH_ISOLATE` | stand-in: `_perf_unpack_stream_set_valid` | block helper, no Dest sync | empty |
| `PACK_ISOLATE` | empty | empty | block helper, no math wait, no section done |

- Stand-in counts: `LOOP_FACTOR * _perf_src_handshake_iters_(tilize_en, num_tiles, num_faces)`.
  Unpack and math must agree, or the threads deadlock.
- Stand-in SrcB: copy the established rule. In datacopy-style unary kernels
  unpack sets SrcB for `tilize_en || is_fp32_dest_acc_en`, and math clears
  `<true, true>`. Unpack-to-dest posts no source dvalids: math either copies
  without a source wait or is idle.
- Call the `_perf_*_stream_*<..., PERF_RUN_TYPE>` stand-ins. On Wormhole
  isolate runs they drive the handshakes through the MOP expander, so the
  stand-in does not set the measured value. Elsewhere they fall back to the
  `_perf_*_loop_*` RISC-V loop themselves, so call them on every architecture.
- Idle branches are empty and fall through to `PROFILER_SYNC()`. Keep the
  same `if constexpr` chain shape in every thread.
- Uninit calls (`_llk_unpack_tilize_uninit_...`) go after the `TILE_LOOP`
  zone, for every run type.
- Options such as tilize work in every run type. A `static_assert` that a
  feature is functional-only is drift.

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
- Never select parameter values on `LLK_PROFILER`. It means "profiler
  instrumentation is compiled", not "this is a perf config". A stand-in there
  silently returns wrong values when a config changes, and it hides untested
  code.
- If a field is missing in one mode, fix the Python side (step 2), not the
  kernel.
- Use sources of truth: `get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()`,
  `TILE_WIDTH * TILE_HEIGHT`. Label literal arguments (`false /* untilize */`,
  `true /* set_a */`) and check each label against the callee's parameter
  name.

## 5. No dead paths

A kernel branch that no sweep value reaches is untested, and untested code
hides bugs. In the pack dest-bank test, tilize was never swept, and its read
offset stepped by `BLOCK_RT_DIM` instead of `BLOCK_CT_DIM`. Either sweep the
option in functional (and so in perf), using the matching golden
(`TilizeGolden`), or delete the branch.

## 6. Perf schema

CSV columns come from the template and runtime dataclass fields. Adding a
runtime such as `INPUT_DIMENSIONS` adds its fields (`full_rt_dim`,
`full_ct_dim`, `block_rt_dim`, `block_ct_dim`). Update the module's entry in
`helpers/perf/test_schemas.py` and bump `version` relative to main, once per
PR. The hardware-free header gate fails on any drift.

## 7. Validate

From `tests/python_tests`, with `tests/.venv` and `CHIP_ARCH` set (or through
`run-test`):

1. **Compile.** Wormhole and Blackhole, normal and `--speed-of-light`,
   functional and perf (`--compile-producer`). Blackhole builds tilize
   variants that Wormhole skips.
2. **Functional on hardware.** Both modes.
3. **Perf on hardware.** All run types. A hang means a stand-in count or a
   sync step is wrong: see "Source dvalid accounting" in `quasar-perf-test`.
4. **Plausibility** of `TILE_LOOP`, per config:
   - every isolate ≤ `L1_TO_L1`;
   - `L1_TO_L1` ≥ the slowest isolate. The gap is the cross-thread handshake,
     largest for one-tile blocks (about 1.5–1.8× on Wormhole pack dest bank)
     and about 1.1–1.4× for 4–8 tile blocks;
   - repeated runs of one build give identical cycles on Wormhole. Any
     run-to-run spread means sticky state or a race.
5. **Coverage.** `python compare_test_and_perf.py --arch wormhole test_[op].py perf_[op].py`,
   then the same with `--arch blackhole`. Expect every coverage axis
   identical and only `run_types` / `loop_factor` ignored.
6. **Hardware-free perf suite**, which includes the header gate:

   ```bash
   LLK_HARDWARE_FREE=1 CHIP_ARCH=wormhole pytest -c pytest.ini --rootdir=. --confcutdir=. \
     -p helpers.llk_pytest_plugin -q -n 8 ../hardware_free/perf
   ```

7. **Perf against main** with `perf-regression-check`. When the sweep or run
   types changed, matching points are few. Read deltas with that skill's
   "Layout sensitivity" section before calling anything a regression.
8. `perf_split_costs.json` / `perf_split_costs_sol.json` weight CI groups by
   module. When the build count changes a lot, say so in the PR;
   `build_perf_split_costs.py` rebuilds them from CI timings.

## PR description

State:

- which drift was removed;
- the isolate model (the table above);
- any intentionally unmatched functional tests;
- the schema version change;
- the validation actually run, per architecture.

Say what was not measured on silicon (for example Blackhole on a Wormhole-only
host).

## Do not

- keep a second kernel source for perf;
- add perf-only sweep values, rejections, stimuli or init steps;
- give an isolate a different loop structure, address set or dst index than
  `L1_TO_L1`;
- choose parameter values on `LLK_PROFILER`;
- keep a feature branch that no sweep exercises;
- narrow the permanent sweep to make a run finish;
- declare success from pytest alone without the plausibility checks.
