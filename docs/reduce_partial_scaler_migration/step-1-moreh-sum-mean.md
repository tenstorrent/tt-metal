# Step 1 — `moreh_sum_h` and `moreh_mean_h`

Migrate the two simplest mask-the-data kernels onto `ReducePartialScaler`. These were chosen first
because their masked scratch buffer feeds nothing but the reduce, so the workaround can be removed
outright rather than partially.

## What the workaround did

Both kernels split every column reduction into two phases:

```cpp
// Phase 1: reduce the first Ht-1 tiles into an accumulator CB
reduce<REDUCE_OP, REDUCE_DIM, input, scaler, accum_dst>(ReduceInputBlockShape::col(Ht - 1));

// Phase 2: mask the last tile, stage it, then fold it in with an accumulating reduce
if constexpr (do_mask_h) {
    tile_regs_acquire();
    copy_tile(input, 0, reduce_dst_idx);       // last tile -> DST
    copy_tile(mask_h, 0, mask_dst_idx);        // 0/1 mask  -> DST
    mask_tile(reduce_dst_idx, mask_dst_idx);   // zero the padding rows
    tile_regs_commit(); ...
    pack_tile(reduce_dst_idx, masked_input);   // stage into a scratch CB
    reduce<..., masked_input, scaler, out>(single(), ..., Accumulate::at(accum_dst, 1));
} else {
    reduce<..., input, scaler, out>(single(), ..., Accumulate::at(accum_dst, 1));
}
```

That required four supporting resources: a `mask_h` CB, a `masked_input` scratch CB, an `accum_dst`
accumulator CB, and a reader-side `generate_mask_h` call. Note the two-phase split ran even when
masking was off — the `else` branch still reduced `Ht-1` tiles and then folded in the last one.

## What it is now

```cpp
constexpr auto partial_scaler = do_mask_h ? ReducePartialScaler::last_tile_at(1)
                                          : ReducePartialScaler::none();
for (nc...) for (wt...) {
    reduce<REDUCE_OP, REDUCE_DIM, input, scaler, out>(
        ReduceInputBlockShape::col(Ht), contiguous(), NoAccumulation{}, NoOp{}, partial_scaler);
}
```

One call over the whole column. The reader emits the scaler pair instead of a mask:

```cpp
calculate_and_prepare_partial_reduce_scalers<scaler, POOL, REDUCE_COL, PARTIAL_H, reduce_factor>();
```

For `moreh_mean_h` both scaler tiles carry `1/origin_H`; the partial tile just fills fewer rows. So
summing the valid rows and scaling still divides by the true element count — the mean is correct
without any separate divisor bookkeeping.

## Files changed

| File | Change |
|---|---|
| `moreh_sum/.../moreh_sum_h.cpp` | two-phase masked reduce → one `reduce()` with `partial_scaler` |
| `moreh_sum/.../reader_moreh_sum_h.cpp` | `generate_mask_h` → `calculate_and_prepare_partial_reduce_scalers` |
| `moreh_sum/device/moreh_sum_h_program_factory.cpp` | `PARTIAL_H` define; scaler CB sized to 2 tiles; dropped CBs `c_3`, `c_24`, `c_25` |
| `moreh_mean/device/kernels/moreh_mean_h.cpp` | same collapse |
| `moreh_mean/device/kernels/reader_moreh_mean_h.cpp` | same reader swap |
| `moreh_mean/device/moreh_mean_h_program_factory.cpp` | `partial_h` CT arg; scaler DFB → 2 entries; dropped `mask_h`/`accum_dst`/`masked_input` DFBs and their bindings |

Net **−174 lines** (82 added, 256 removed).

## Notes and issues encountered

**The fill count has to reach the reader at compile time.** `prepare_partial_reduce_scalers` takes
`partial_positions` as a template parameter, but both readers received `mask_h` as a *runtime* arg.
`origin_H` is known at program-creation time, so the host already had the value — it just wasn't
being passed as compile-time data. Resolved differently per op because the two factories use
different arg systems:

- `moreh_sum_h` uses positional compile-time args followed by `TensorAccessorArgs<3>()`. Inserting a
  CT arg would have shifted the accessor's base index, so the value goes through a **define**
  (`PARTIAL_H`) instead.
- `moreh_mean_h` uses the named-CT-arg system, so it takes a plain `{"partial_h", mask_h}` entry.

This is a genuine ergonomic wrinkle in the helper: an op that only knows its ragged count at runtime
cannot use `prepare_partial_reduce_scalers` at all. Not a problem for any Step 1–3 caller, but a
runtime-count overload may be worth adding if a later step needs one.

**The scaler CB had to grow.** Both factories allocated exactly one scaler tile. The partial path
needs two, so the CB is now sized `do_mask_h ? 2 : 1`. Missing this would deadlock the reader on
`reserve_back` for the second tile.

**Dead runtime arg left in place.** `mask_h` is still passed as a runtime arg by both factories but
is no longer read by either reader. Removing it would mean touching the per-core runtime-arg emission
loops and (for `moreh_mean_h`) the `runtime_arg_schema`, which is churn disproportionate to the
benefit. The kernels carry a comment saying so.

**`accum_dst`'s `UnpackToDest` entry went away.** `moreh_mean_h`'s factory set
`unpack_modes = {{ACCUM_DST_DFB, UnpackMode::UnpackToDest}}` under `fp32_dest_acc_en`, because the
accumulator was a Float32 buffer read back through a 32-bit dest register. With no accumulator buffer
the entry is meaningless and was removed along with it.

## Verification

- `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_mean.py`
- `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_sum.py`

Both suites parametrise on `[3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1]` = `[3, 2, 319, 319]`.
`319 % 32 = 31`, so `dim=2` (and every dim list containing 2) drives the H reduce down the partial
path with 31 valid rows — the migrated code is genuinely covered, across both `bfloat16` and
`bfloat8_b` and both compute-kernel option sets.

## Test results

| Suite | Result |
|---|---|
| `test_moreh_mean.py` | **76 passed, 72 skipped** in 154.70s |
| `test_moreh_sum.py` | **229 passed, 155 skipped** in 422.22s |

No failures. The skips are pre-existing parametrisation guards (e.g. `test_moreh_sum_non_4d` skipping
when `dim >= input_rank`), not anything this change introduced.

Also unchanged: the 36-case `toy_reduce_partial` suite and the 638-case layernorm/softmax regression
from the helper commit still pass, since Step 1 touches no shared code — only these two ops and their
factories.
