---
name: update-quasar-perf-run-types
description: Update existing Quasar LLK C++ test kernels so UNPACK_ISOLATE, MATH_ISOLATE, PACK_ISOLATE, and L1_CONGESTION work correctly alongside L1_TO_L1. Use when adding PerfRunType branches, debugging implausible Quasar perf CSV metrics, fixing dvalid handshake pollution, or validating tests under tests/sources/quasar.
---

# Update Quasar PerfRunType Kernels

## Goal

Extend an existing, functionally correct `L1_TO_L1` Quasar kernel to all
`PerfRunType` modes without changing functional behavior or leaving hardware
state that pollutes later runs.

Use these validated references:

- `tests/sources/quasar/pack_untilize_quasar_test.cpp`
- `tests/sources/quasar/eltwise_binary_broadcast_quasar_test.cpp`

Do not copy their loop counts blindly. Derive counts from the target kernel's
actual unpack and math MOP behavior.

## Required run-type behavior

Implement compile-time `if constexpr` branches independently in unpack, math,
and pack.

| Run type | Unpack | Math | Pack |
|---|---|---|---|
| `L1_TO_L1` | Real | Real | Real |
| `UNPACK_ISOLATE` | Real | Clear exactly the source dvalids unpack produces | No-op |
| `MATH_ISOLATE` | Produce exactly the source dvalids math consumes | Real, but do not signal inactive pack | No-op |
| `PACK_ISOLATE` | No-op | No-op | Real without waiting for math |
| `L1_CONGESTION` | Real | Clear source dvalids only | Real independently, without math-to-pack handshake |

Some kernels bypass math, such as unpack-to-dest. Adapt the table to the real
pipeline instead of manufacturing work for a stage that is not used.

## Workflow

1. Read the C++ kernel, correctness harness, perf harness, and latest
   `perf_data/<test>/<test>.post.csv`.
2. Map every real producer/consumer handshake:
   - SrcA and SrcB dvalid between unpack and math
   - destination dvalid between unpack/math and pack
   - any semaphore-based unpack-to-dest synchronization
3. Determine how many dvalid pulses one real kernel invocation produces and
   consumes. Inspect the LLK MOP implementation when this is not explicit.
4. Add each run-type branch using the table above.
5. Compile a focused sweep before simulation.
6. Run all run types together to detect cross-run sticky state.
7. Inspect `TILE_LOOP` metrics and run the normal functional test.
8. Restore the full parameter sweep after fast iteration unless the user asks
   to keep it focused.

## Source dvalid accounting

The mock count must equal hardware behavior exactly:

```text
total source handshakes =
    LOOP_FACTOR
  × real operation invocations per loop
  × dvalid pulses per invocation
```

Use `_perf_unpack_loop_set_valid<set_a, set_b>(count)` only when unpack is being
mocked for `MATH_ISOLATE`.

Use `_perf_math_loop_clear_valid<clear_a, clear_b>(count)` only when math is
mocking consumption for `UNPACK_ISOLATE` or `L1_CONGESTION`.

Verify all three parts of the formula:

- Count tiles and blocks actually consumed, not just outer-loop iterations.
- Determine whether the MOP pulses once per tile or once per face.
- Determine whether math requires SrcA, SrcB, or both.

Known cases:

- Scalar binary broadcast clears SrcAB once on the final outer-loop iteration:
  one handshake per tile, not one per face.
- Row/column broadcast may handshake per face; inspect the MOP.
- Quasar 32-bit unary datacopy uses ELWADD. Unpack produces real SrcA plus
  dummy SrcB, so mocks must use `<true, true>` on both sides.
- A block kernel commonly needs
  `LOOP_FACTOR * BLOCK_RT_DIM * BLOCK_CT_DIM`, equivalent to
  `LOOP_FACTOR * TILE_CNT` only when that invariant is guaranteed.

Never add a synthetic SET/CLEAR pair to "flush" the pipeline. An extra token
can race the final real operation and cause a later run to stall.

## Destination dvalid rules

Only signal destination completion when a consumer is active.

- `L1_TO_L1`: keep the real producer-to-pack setup and section-done calls.
- `MATH_ISOLATE`: run real math but do not call `_llk_math_set_dvalid_` when
  pack is inactive.
- `PACK_ISOLATE`: pack without waiting for math and without destination
  section-done.
- `L1_CONGESTION`: unpack and pack run independently; math must not pulse
  destination dvalid.
- `UNPACK_ISOLATE`: do not signal destination completion to inactive pack.

Quasar CFG state can persist across run types. For independent pack execution,
clear the pack wait mask during `INIT`:

```cpp
auto cfg = (std::uint32_t volatile *)TENSIX_CFG_BASE;
cfg[PACK_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
```

Configure `set_up_dest_dvalid_per_thread` only for paths that have the matching
producer and consumer.

## Unpack-to-dest

Treat unpack-to-dest separately:

- Math may be compiled out; a missing `MATH_ISOLATE` metric is then expected.
- `L1_TO_L1` performs real unpack and signals pack completion.
- `UNPACK_ISOLATE` performs real unpack but must not signal inactive pack.
- `L1_CONGESTION` runs unpack and pack independently without destination
  dvalid coupling.
- `MATH_ISOLATE` must not emit dummy source dvalid when no math consumer exists.

Preserve the functional `L1_TO_L1` path exactly. Correctness tests normally
compile only that branch, so isolate-only logic must disappear at compile time.

## Fast iteration sweep

Temporarily select five or six combinations that cover distinct handshake
paths. Prefer:

1. 16-bit destination without destination accumulation
2. 16-bit destination with destination accumulation
3. output-format conversion
4. alternate input format
5. block/tile shape with different loop counts, when supported
6. 32-bit or unpack-to-dest path

Run all run types in the same pytest case. This is essential: individually
healthy modes can still leave sticky state that breaks the next mode.

Use a single run type only to isolate which mode introduces pollution, then
restore the combined run.

## Diagnosing CSV output

Analyze `marker == TILE_LOOP` first.

Healthy patterns:

- `L1_TO_L1` is plausibly related to the active stage costs. Pipeline overlap
  means it need not equal their sum.
- `L1_CONGESTION[UNPACK]` is close to `UNPACK_ISOLATE`.
- `L1_CONGESTION[PACK]` is close to `PACK_ISOLATE`.
- Format and destination-mode changes cause explainable, bounded differences.

Strong failure signals:

- Repeated values near 2048, 4096, 8192, or larger multiples
- First variant is healthy but later variants are huge
- One isolate is huge while the corresponding real L1 path is small
- Congestion is orders of magnitude larger than its isolate
- A run passes but leaves the following run slow

These usually indicate unmatched dvalids, a wrong mock count, an inactive
consumer receiving section-done, or a persistent wait mask. Treat implausible
numbers as a kernel/test-harness correctness issue, not as valid performance.

## Validation

From the `tt-llk` root:

1. Compile the focused perf module with `.cursor/scripts/run_test.sh`.
2. Run the focused module with all run types together; use `--no-split` when
   compile/simulation artifact handoff is unreliable.
3. Confirm every focused pytest case passes.
4. Inspect every focused `TILE_LOOP` row in the post-processed CSV.
5. Run the shared non-perf correctness test, especially the unpack-to-dest and
   destination-accumulation variants.
6. Check edited files for lint errors and run `git diff --check`.

Do not declare the kernel fixed based only on pytest passing. Perf mocks can
complete after long hardware timeouts and still produce meaningless metrics.

## Scope guardrails

- Fix the test kernel's run-type orchestration first.
- Change LLK library code only when the functional `L1_TO_L1` behavior or an
  independently reproduced LLK primitive is wrong.
- Do not weaken or remove functional assertions.
- Do not leave the permanent perf sweep narrowed accidentally.
- Explain any intentionally absent metric, such as math for unpack-to-dest.
