# Step 2 — moreh softmax family

Migrate the max-phase partial-tile workaround in the moreh softmax kernels onto
`ReducePartialScaler`. Unlike Step 1 this is a **partial** migration, for a reason worth recording.

## Why softmax can only be partially migrated

The mask CB is used for two different jobs:

1. **Max phase** — `mask_tile_to_cb(in0, mask, tmp, ...)` masks the last tile into a scratch CB, and
   the MAX reduce consumes that scratch CB. The scratch CB feeds *nothing else*, so this is a clean
   swap to a partial scaler.
2. **Exp phase** — after computing `exp(x - max)`, the last tile is masked **in place** in `cb_exps`.
   `cb_exps` then feeds *both* the SUM reduce *and* the final `exps * recipsumexps` multiply that
   produces the op output.

A partial scaler on the sum would make the sum correct, but nothing would zero the padding lanes of
`cb_exps`, so the **output tile's padding** would change from zeros to `exp(garbage) * recip`. That
is an observable change to the op's output buffer, not just to an internal reduction, so the exp-phase
mask stays. The mask CB and its `generate_mask_{h,w}` call therefore remain for all softmax kernels.

Consequence: Step 2 does **not** get softmax to "no partial-tile workarounds". It removes one of the
two workarounds per kernel.

## A new helper overload was required

`prepare_partial_reduce_scalers` takes the fill count as a **template** parameter. moreh softmax
derives `mask_h` / `mask_w` from the tensor shape on the host but passes it as a **runtime** arg — the
op compiles one kernel per core-split, not per shape. Making it compile-time would mean re-indexing
`TensorAccessorArgs<N>` in the readers and factories of all eight softmax variants.

So a runtime-count overload was added to `reduce_helpers_dataflow.{hpp,inl}`:

```cpp
template <uint32_t cb_id, PoolType pool_type, ReduceDim reduce_dim>
FORCE_INLINE void prepare_partial_reduce_scalers(float scaler_f, uint32_t partial_positions);

template <uint32_t cb_id, PoolType pool_type, ReduceDim reduce_dim, uint32_t reduce_factor = ...>
FORCE_INLINE void calculate_and_prepare_partial_reduce_scalers(uint32_t partial_positions);
```

This is additive and mirrors the existing pair; `prepare_reduce_scaler` already took its count at
runtime, so nothing new had to be built underneath. One deliberate difference: the runtime form
**allows `partial_positions == full_dim`**, producing a tile 1 identical to tile 0. The compile-time
form rejects that (it wants the caller to use the cheaper single-tile path), but a shape-generic
kernel needs the aligned case to be expressible without a separate build.

## What changed per kernel

Before (softmax_h_small, and structurally identical in softmax_w_small):

```cpp
if (Ht == 1) {
    mask_tile_to_cb(in0, mask, tmp, 0, 0, 0, 0);
    reduce<MAX, REDUCE_COL, cb_tmp, cb_max_scaler, cb_max>(single());
} else {
    reduce<MAX, REDUCE_COL, cb_in0, cb_max_scaler, cb_max, WaitUpfrontNoPop>(col(Ht - 1));
    mask_tile_to_cb(in0, mask, tmp, Ht - 1, 0, 0, 0);
    reduce<MAX, REDUCE_COL, cb_tmp, cb_max_scaler, cb_max>(single(), ..., Accumulate::at(cb_max, 1));
}
```

After:

```cpp
reduce<MAX, REDUCE_COL, cb_in0, cb_max_scaler, cb_max, WaitUpfrontNoPop>(
    col(Ht), contiguous(), NoAccumulation{}, NoOp{}, ReducePartialScaler::last_tile_at(1));
```

The `Ht == 1` / `Ht > 1` branch, the scratch CB `c_28`, and the accumulating second reduce all go
away. The max-scaler CB grows to two tiles.

### Files changed

| File | Change |
|---|---|
| `kernel_lib/reduce_helpers_dataflow.hpp/.inl` | runtime-count overloads (additive) |
| `moreh_softmax/device/kernels/moreh_softmax_h.cpp` | max phase → one reduce with partial scaler; `cb_tmp` dropped |
| `moreh_softmax/device/kernels/moreh_softmax_w.cpp` | same, REDUCE_ROW |
| `moreh_softmax/device/kernels/reader_moreh_softmax_h.cpp` | max scaler → partial pair (runtime count) |
| `moreh_softmax/device/kernels/reader_moreh_softmax_w.cpp` | same |
| `.../softmax_h_small/softmax_h_small.cpp` | max-scaler CB → 2 tiles; `c_28` removed |
| `.../softmax_w_small/softmax_w_small.cpp` | same |
| `tests/.../test_moreh_softmax.py` | new `test_softmax_non_tile_aligned` |

## Issues encountered

**The existing test suite had zero coverage of the migrated path.** Every shape in
`test_moreh_softmax.py` is a multiple of 32 (`[32, 32]`, `[3, 32, 32*5]`, `[10, 20, 32*3, 32*5]`, …),
so `mask_h`/`mask_w` was always `TILE_DIM` and the ragged tail never ran. The suite would have passed
identically with the migration completely broken.

This was found by inspection, not by a failure, and is the most important finding of this step. A new
`test_softmax_non_tile_aligned` case was added covering ragged H and ragged W, single-core and
multi-core, from 1 to 31 valid elements. Softmax normalises, so a leaked padding element shows up
both as a per-element mismatch and as a row sum that is not 1.0.

**Latent MAX-over-negatives bug fixed as a side effect.** `mask_tile` masks with `0`, so the old code
reduced an all-negative column to `max(values, 0) == 0`. The partial scaler genuinely excludes the
padding (verified on device — see the top-level README). Softmax is shift-invariant, so the final
result was unaffected either way; the fix matters only if this pattern is copied to an op that returns
the max.

## Scope actually completed

Done: `softmax_h_small`, `softmax_w_small` (i.e. `moreh_softmax_h.cpp`, `moreh_softmax_w.cpp`).

**Not done, deliberately deferred:** the four `_large` variants
(`moreh_softmax_{h,w}_large.cpp`) and the four `moreh_softmax_backward_*` kernels. The `_large`
kernels call `mask_tile_to_cb` with `pop0=1` — they *consume* the input as they stream and then make a
second pass over it, so collapsing their two-phase max into one resident-block reduce changes the CB
consumption pattern rather than just the scaler. That needs its own analysis and its own test run, and
is tracked as follow-up rather than rushed in alongside the straightforward small-kernel swap.

## Test results

| Suite | Result |
|---|---|
| `test_moreh_softmax.py` after `softmax_h` migration | **93 passed, 32 skipped** in 274.26s |
| `test_moreh_softmax.py -k non_tile_aligned` (new cases) | **14 passed** in 22.63s |
| `test_moreh_softmax.py` full, after both migrations | **107 passed, 32 skipped** in 103.19s |

Plus an ad-hoc device probe of ragged softmax before the test was written, confirming
`allclose` against torch and row sums within `[0.958, 1.005]` for H = 33 / 47 / 63 / 45-multicore.
