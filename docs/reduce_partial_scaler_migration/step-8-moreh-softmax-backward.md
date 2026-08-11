# Step 8 — `moreh_softmax_backward`: full mask removal on the small kernels, and why the `_large` ones cannot follow

Scope: `moreh_softmax_backward_h.cpp` and `moreh_softmax_backward_w.cpp` (the SMALL_H / SMALL_W
kernels), across the SOFTMAX, SOFTMIN and LOG variants. Unlike the forward small kernels in Step 2
this is a **complete** removal of the mask-the-data workaround.

The `_large` kernels were attempted, failed on device, and are **not** migrated. That is the most
useful result in this step and it is written up below — it also rules out the forward `_large`
migration that Step 2 deferred, and it is a general constraint on `ReducePartialScaler`.

## Why backward's small kernels can go all the way and the forward ones could not

Step 2 had to keep the forward exp-phase mask because `cb_exps` feeds both the sum reduce *and* the
op's output tile, so unmasking it would change the output buffer's padding lanes.

Backward has no such consumer. In the small kernels the masked tile feeds **only** the reduce:

| variant | what was masked | who consumes it |
|---|---|---|
| SOFTMAX / SOFTMIN | last tile of `cb_ydy` (`y*dy`) | the SUM reduce, then popped |
| LOG | last `dy` tile, copied into a scratch CB | the SUM reduce |

`dx` is recomputed from `y`, `dy` and `sum` in a separate epilogue loop which never masked anything,
so the output's padding lanes are unchanged by this step.

What it looks like now — one reduce over the whole axis, replacing a two-phase split *and* its
`Ht == 1` special case:

```cpp
// before: reduce(col(Ht-1)) -> inter0; mask last tile -> inter1; reduce(single()) -> inter2;
//         add_tiles_to_cb(inter0, inter2) -> sum      (plus a separate Ht == 1 branch)
compute_kernel_lib::reduce<SUM, REDUCE_COL, cb_dy, cb_bcast_scaler, cb_sum, WaitUpfrontNoPop>(
    ReduceInputBlockShape::col(Ht), contiguous(), NoAccumulation{}, NoOp{}, partial_scaler);
```

## The `_large` kernels: a partial scaler cannot follow an element-wise accumulation

The `_large` kernels are large precisely because they do not keep the row resident. They fold every
tile into a **single** accumulator tile and then reduce that one tile:

```cpp
for (uint32_t h = 0; h < Ht; ++h) {
    if (h == Ht - 1) { mul_tiles_and_mask_tile_to_cb(y, dy, mask, ydy, ...); }  // mask the ragged tile
    else             { mul_tiles_to_cb(y, dy, ydy); }
    if (h == 0) { copy_tile_to_cb(ydy, add); } else { add_tiles_to_cb(add, ydy, add); }
}
reduce<SUM, REDUCE_COL, cb_add, scaler, cb_sum>(single());   // full scaler, deliberately
```

Removing the mask and putting `last_tile_at(1)` on that `single()` reduce looks equivalent. It is not,
and the device says so:

```
pcc=0.8589, Max RTOL Delta 5.78   ([1,1,10,74] dim=3, forced LARGE_W)
```

**Why:** the accumulation is *element-wise across tiles*, so `cb_add[j]` is
`in0[j] + in1[j] + … + inLast[j]`. For a lane `j >= mask_w`, `inLast[j]` is padding — but `in0[j]`,
`in1[j]`, … are **valid elements of the same row**. A partial scaler zeroes lane `j` for the whole
accumulated tile, discarding all of those valid contributions. The mask is load-bearing here in a way
it was not in the small kernels: it zeroes the ragged tile's invalid lanes *before* the accumulation
so that the subsequent full-scaler reduce sums every lane correctly.

The failure signature matches exactly: the single-tile ragged cases (`Wt == 1`, `Ht == 1`) **passed**,
because there are no earlier tiles to lose; every multi-tile ragged case failed.

### The general rule this establishes

> `ReducePartialScaler` describes the last tile **along the reduce dimension of the reduce itself**.
> If a kernel collapses several tiles into one by element-wise accumulation *before* reducing, the
> raggedness stops being expressible as a scaler mask, because the accumulated tile's padding lanes
> also carry valid data from the other tiles. Such kernels must keep masking the ragged tile before
> the accumulation.

Two consequences beyond this step:

1. **The forward `_large` kernels are also out of reach** for the "remove the mask entirely" idea
   floated in the plan for Step 2's deferred work: `moreh_softmax_{h,w}_large.cpp` accumulate `cb_exps`
   into `cb_add` in exactly the same shape. Their max phase (which reduces the input tiles directly,
   with no accumulation) is still a candidate; their exp/sum phase is not.
2. It is worth stating in the `ReducePartialScaler` docs, which currently warn only about combining a
   partial scaler with `Accumulate` (a related but distinct trap: that one is about accumulation
   *between* `reduce()` calls, this one is about accumulation *before* a single call).

## Two side-cleanups that came with the small-kernel change

**The `mask_h`/`mask_w` count reaches compute as a compile-time arg**, exactly as in Step 7a: `Ht`/`Wt`
is already a compile-time arg for these kernels, so they are built per shape and the partial-scaler
path constant-folds away entirely on aligned shapes. The reader keeps a runtime count (a compile-time
arg there would shift the `TensorAccessorArgs` base index) and branches once per launch.

**A dead packed-`1.0f` scaler runtime arg is gone** (small kernels only). The `h` reader called
`generate_bcast_scaler(scaler)` with a host-supplied `std::bit_cast<uint32_t>(1.0f)`; the `w` reader had
already moved to `calculate_and_prepare_reduce_scaler` and simply ignored the value the host still
passed. Both now compute the scaler in-kernel and the arg is gone from both factories.

`generate_bcast_scaler` and `prepare_reduce_scaler`'s full-fill path were checked to be equivalent
before swapping: both write the scaler into row 0 of each of the four faces
(`ptr[k*256 + j]`, `j < 16` vs. `fill_each_face_row0<num_faces>`), so the aligned case is byte-identical.

## Files changed

| File | Change |
|---|---|
| `kernels/moreh_softmax_backward_{h,w}.cpp` | two-phase masked reduce + `Ht==1` branch → one `reduce()` with a partial scaler (LOG and non-LOG); mask CB dropped |
| `kernels/reader_moreh_softmax_backward_{h,w}.cpp` | scaler pair when ragged; `generate_mask_{h,w}` gone; dead scaler arg gone |
| `device/softmax_backward_{h,w}_small/*.cpp` | scaler CB `1 or 2` tiles; mask CB `c_3` dropped; `mask_{h,w}` as compute CT arg; dead RT arg dropped |
| `tests/.../test_moreh_softmax.py`, `test_moreh_logsoftmax.py` | new ragged × forced-LARGE cases |

The four `_large` files and their two factories are unchanged.

## Coverage: the Step 2 trap, again — and this time it paid immediately

- `test_softmax_backward_not_multiple_of_32_for_dim_hw` **does** cover ragged H and W (15, 10, 74 …),
  and the same cases exist for logsoftmax, so the LOG path is covered. But its shapes are small, so the
  op always picks a SMALL strategy.
- `test_softmax_backward_large_algorithmfor_dim_hw` is the only thing that reaches the `_large`
  kernels, and every shape in it is `32 * k`.

So the `_large` kernels had **zero** ragged coverage. Added
`test_softmax_backward_large_algorithm_not_multiple_of_32_for_dim_hw` (and the logsoftmax twin), which
takes the ragged shapes and forces `SoftmaxBackwardOpParallelizationStrategy.LARGE_{W,H}`. Forcing
LARGE is always legal — only the SMALL strategies carry an availability `TT_FATAL`.

**That new test is what caught the bad `_large` migration**, on its first run, before anything was
committed. It stays in the suite: it now guards the masked `_large` implementation against exactly this
class of change.

## Test results

| Suite | Result |
|---|---|
| `test_moreh_softmax.py` | **115 passed, 32 skipped** in 40.11s |
| `test_moreh_logsoftmax.py` | **100 passed, 32 skipped** in 112.37s |
| `test_moreh_softmin.py` | **92 passed, 32 skipped** in 97.23s |

(The 115 includes the 4 new large-ragged cases. The intermediate run *with* the `_large` migration was
`4 failed, 111 passed` — the four multi-tile ragged large cases.)
