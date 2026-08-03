---
name: quasar-perf-test
description: Create, extend, debug, and validate Quasar LLK performance tests and their PerfRunType kernel paths. Use when adding perf_[op]_quasar.py, wiring PerfConfig, implementing UNPACK_ISOLATE / MATH_ISOLATE / PACK_ISOLATE / L1_CONGESTION, or investigating implausible Quasar perf metrics and dvalid handshakes.
---

# Quasar Perf Test

## Goal

Create or repair a Quasar performance test while preserving its functional
`L1_TO_L1` behavior. Reuse the correctness harness, report through
`PerfConfig`, and give every applicable `PerfRunType` a balanced single-stage
or congestion path.

This skill covers two related workflows:

- **Create**: add a perf harness for an existing correctness test.
- **Repair or extend**: implement or debug run-type behavior in an existing
  Quasar kernel.

## Choose the workflow

1. If `tests/python_tests/quasar/perf_[op]_quasar.py` does not exist, follow
   **Create a perf test**.
2. If the perf harness exists but run types are missing, hanging, or producing
   implausible metrics, follow **Repair PerfRunType paths**.
3. In either case, finish with **Validation**.

## Create a perf test

1. Read:
   - `tests/python_tests/quasar/test_[op]_quasar.py`
   - `tests/sources/quasar/[op]_quasar_test.cpp`
   - the closest current Quasar perf test with the same pipeline shape
2. Move reusable parameter axes from inline lambdas into named helpers when
   the perf harness needs them.
3. Add `tests/python_tests/quasar/perf_[op]_quasar.py`.
4. Mark it with `@pytest.mark.perf` and `@pytest.mark.quasar`.
5. Import the correctness test as a helper, normally:

   ```python
   from test_[op]_quasar import test_[op]_quasar as run_[op]_quasar
   ```

6. Use a narrow perf-oriented sweep:
   - `run_types=PERF_RUN_TYPES_QUASAR` from `helpers.llk_params`
   - a fixed `loop_factor=32`, unless the operation has an established value
   - a fixed `is_perf=True`
   - stable dimensions and exact destination fill when normalization depends
     on shape
7. Pass `perf_report` and `is_perf=True` to the correctness helper.
8. Add keyword-only `is_perf=False` and `perf_report=None` to that helper.
   Build shared `test_config_kwargs`, then:

   ```python
   if is_perf:
       if perf_report is None:
           raise ValueError("perf_report must be provided when is_perf=True")
       PerfConfig(run_types=run_types, **test_config_kwargs).run(perf_report)
       return
   ```

9. Keep the correctness path on `TestConfig` with an explicit
   `PERF_RUN_TYPE(PerfRunType.L1_TO_L1)`.
10. Preserve dynamic shape coverage in correctness tests. Use fixed,
    comparable workloads only for the perf path.

## C++ structure

In `tests/sources/quasar/[op]_quasar_test.cpp`:

- Include `perf.h` and `profiler.h`.
- Put setup in `ZONE_SCOPED("INIT")`.
- Put repeated work in `ZONE_SCOPED("TILE_LOOP")`.
- Read `LOOP_FACTOR` and use it in steady-state work.
- Branch with compile-time `if constexpr (PERF_RUN_TYPE == ...)`.
- End each profiled zone with `PROFILER_SYNC()`.
- Preserve the functional `L1_TO_L1` path.
- Use current Quasar LLK signatures and TensorShape APIs; do not copy stale
  call signatures from older perf branches.

Use the target Quasar correctness kernel for current Quasar LLK signatures.
The generic perf references use older-architecture APIs and must not be copied
verbatim.

Do not copy loop counts blindly. Derive them from the target kernel's actual
MOP and handshake behavior.

## Required run-type behavior

Implement behavior independently in unpack, math, and pack:

| Run type | Unpack | Math | Pack |
|---|---|---|---|
| `L1_TO_L1` | Real | Real | Real |
| `UNPACK_ISOLATE` | Real | Clear exactly the source dvalids unpack produces | No-op |
| `MATH_ISOLATE` | Produce exactly the source dvalids math consumes | Real; do not signal inactive pack | No-op |
| `PACK_ISOLATE` | No-op | No-op | Real without waiting for math |
| `L1_CONGESTION` | Real | Clear source dvalids only | Real independently |

Adapt this table to the real pipeline. For example, an unpack-to-dest kernel
may bypass math entirely.

## Repair PerfRunType paths

1. Read the C++ kernel, correctness harness, perf harness, and latest
   `perf_data/<test>/<test>.post.csv`.
2. Map every producer/consumer handshake:
   - SrcA and SrcB dvalid between unpack and math
   - destination dvalid between unpack/math and pack
   - semaphore-based unpack-to-dest synchronization
3. Inspect the LLK MOP when pulse counts are not explicit.
4. Determine the exact pulse count for one kernel invocation.
5. Implement or correct each run-type branch.
6. Compile a focused sweep.
7. Run all run types together to expose sticky state.
8. Inspect `TILE_LOOP` metrics and run the functional test.
9. Restore the intended full sweep after focused debugging.

## Source dvalid accounting

The mock count must match hardware behavior exactly:

```text
total source handshakes =
    LOOP_FACTOR
  × real operation invocations per loop
  × dvalid pulses per invocation
```

- Use `_perf_unpack_loop_set_valid<set_a, set_b>(count)` only when unpack is
  mocked for `MATH_ISOLATE`.
- Use `_perf_math_loop_clear_valid<clear_a, clear_b>(count)` only when math
  mocks consumption for `UNPACK_ISOLATE` or `L1_CONGESTION`.
- Count tiles and blocks actually consumed.
- Determine whether the MOP pulses per tile or per face.
- Determine whether math consumes SrcA, SrcB, or both.

Known cases:

- Scalar binary broadcast clears SrcAB once on the final outer-loop
  iteration: one handshake per tile, not per face.
- Row/column broadcast may handshake per face; inspect the MOP.
- Quasar 32-bit unary datacopy uses ELWADD. Unpack produces real SrcA and
  dummy SrcB, so both mock sides use `<true, true>`.
- A block kernel commonly needs
  `LOOP_FACTOR * BLOCK_RT_DIM * BLOCK_CT_DIM`.

Never add a synthetic SET/CLEAR pair to flush the pipeline. An extra token can
race the final operation and pollute a later run.

## Destination dvalid rules

Only signal completion when the consumer is active:

- `L1_TO_L1`: retain real producer-to-pack setup and section-done calls.
- `MATH_ISOLATE`: run math without `_llk_math_set_dvalid_`.
- `PACK_ISOLATE`: pack without waiting for math and without destination
  section-done.
- `L1_CONGESTION`: unpack and pack run independently; math does not pulse
  destination dvalid.
- `UNPACK_ISOLATE`: do not signal inactive pack.

Quasar CFG state can persist between run types. For independent pack
execution, clear the wait mask during `INIT` when required:

```cpp
auto cfg = (volatile std::uint32_t*)TENSIX_CFG_BASE;
cfg[PACK_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
```

Configure `set_up_dest_dvalid_per_thread` only when the matching producer and
consumer are active.

## Unpack-to-dest

- A missing math-isolate metric is expected when math is compiled out.
- `L1_TO_L1` performs real unpack and signals active pack.
- `UNPACK_ISOLATE` performs real unpack without signaling inactive pack.
- `L1_CONGESTION` runs unpack and pack independently.
- `MATH_ISOLATE` must not emit dummy source dvalid without a math consumer.

## Diagnose metrics

Analyze `marker == TILE_LOOP` first.

Healthy relationships:

- `L1_CONGESTION[UNPACK]` is close to `UNPACK_ISOLATE`.
- `L1_CONGESTION[PACK]` is close to `PACK_ISOLATE`.
- Format and destination-mode changes have bounded, explainable effects.

Failure signals:

- Values near large powers of two such as 2048, 4096, or 8192
- Healthy first variant followed by very slow variants
- Isolate cost orders of magnitude above the corresponding real stage
- Congestion cost orders of magnitude above isolate
- A passing run that leaves the next run slow

Treat these as likely handshake, mock-count, section-done, or wait-mask bugs;
pytest completion alone does not validate the metrics.

## Validation

From the `tt-llk` root:

1. Use the repository LLK test runner workflow, not direct pytest.
2. Compile a focused perf variant, then run all run types together.
3. Confirm each focused pytest case passes.
4. Inspect every focused `TILE_LOOP` CSV row.
5. Run the shared non-perf correctness test, including unpack-to-dest and
   destination-accumulation variants when applicable.
6. Check edited files for lint errors and run `git diff --check`.

Do not:

- weaken functional assertions;
- change LLK library code before isolating a test-kernel orchestration bug;
- leave the permanent sweep narrowed accidentally;
- declare success from pytest alone when metrics are implausible.
