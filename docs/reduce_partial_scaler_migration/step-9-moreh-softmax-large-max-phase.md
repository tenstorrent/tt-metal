# Step 9 — `moreh_softmax_{h,w}_large`: the max phase

The forward `_large` kernels, i.e. the last piece of the moreh softmax family. Only the **max phase**
moves; the exp/sum phase cannot, for the reason established in step 8.

## Why the max phase can and the sum phase cannot

The two phases consume their input differently, and that is the whole story:

| phase | how it consumes | partial scaler? |
|---|---|---|
| max | reduces the `Ht`/`Wt` **input tiles directly**, streaming and popping one at a time | **yes** — the last tile it sees *is* the last tile of the reduce dim |
| exp/sum | folds every tile into `cb_add` **element-wise**, then reduces that one tile | **no** — lane `j >= mask` of the accumulator holds valid data from the earlier tiles |

So the mask CB and the exp-phase `mask_tile` stay. This step removes the max phase's two-phase split,
not the workaround as a whole — the same partial outcome as Step 2 had on the small kernels.

## What changed

```cpp
// before (w_large): a Wt == 1 branch, plus a Wt > 1 branch that reduced Wt-1 tiles,
// masked the last one into a scratch CB, and folded it in with an accumulating reduce
reduce<MAX, REDUCE_ROW, cb_in0, cb_max_scaler, cb_max>(row(Wt - 1));
mask_tile_to_cb(in0, mask, tmp, 0, 0, /*pop0=*/1, /*popm=*/0);
reduce<MAX, REDUCE_ROW, cb_tmp, cb_max_scaler, cb_max>(row(1), ..., Accumulate::at(cb_max, 1));

// after
reduce<MAX, REDUCE_ROW, cb_in0, cb_max_scaler, cb_max>(
    row(Wt), contiguous(), NoAccumulation{}, NoOp{}, max_partial_scaler);
```

The default `WaitAndPopPerTile` policy is what makes this equivalent: the old code popped `Wt-1` tiles
in phase 1 and the last one via `mask_tile_to_cb(pop0=1)`, so `Wt` pops either way — which matters
because these kernels exist precisely to stream the input and have the reader re-send it for the later
passes.

Dropping the `Accumulate` is a bonus: `MAX + REDUCE_ROW` accumulation is rejected on Quasar by a
`static_assert` (the reload needs a within-16x16-face transpose), so `w_large` was carrying an
arch-limited construct it no longer needs.

## Files changed

| File | Change |
|---|---|
| `kernels/moreh_softmax_{h,w}_large.cpp` | max phase → one streaming `reduce()` with the partial scaler; `mask_{h,w}` CT arg |
| `kernels/reader_moreh_softmax_{h,w}_large.cpp` | max scaler emitted as a pair only when ragged |
| `moreh_softmax/device/softmax_{h,w}_large/*.cpp` | `c_2` sized 1-or-2; `mask_{h,w}` hoisted out of the per-core loop and passed to compute |
| `normalization/softmax/device/softmax_program_factory_general_{h,w}_large.cpp` | the same, because they share these kernels |

Four factories for two kernels — the Step 7b lesson applied up front this time: `ttnn.softmax`'s
general path builds these same `_large` kernels, and a host/kernel disagreement on the scaler-tile
count is a hang, not a wrong answer. The general large factories have no L1-fit estimate to update
(they are the fallback when `*Small` does not fit), so nothing else moved.

## Coverage, checked first

`test_softmax_large_algorithm_for_dim_hw` only uses multiples of 32, and `test_softmax_non_tile_aligned`
does not force a strategy, so its ragged shapes take a SMALL kernel. The forward `_large` kernels had
no ragged coverage at all. Added
`test_softmax_large_algorithm_not_multiple_of_32_for_dim_hw`, which forces
`SoftmaxOpParallelizationStrategy.LARGE_{W,H}` on ragged shapes including the single-tile cases (those
are the ones that used to take the separate `Wt == 1` / `Ht == 1` branch that this step deletes).

**It was run against the pre-migration code first and passed (16 cases)**, so it is a real gate rather
than a test written to fit the new behaviour.

## Test results

| Suite | Result |
|---|---|
| `test_moreh_softmax.py` | **123 passed, 32 skipped** in 44.01s (115 + 8 new) |
| `test_moreh_logsoftmax.py` + `test_moreh_softmin.py` + `tests/ttnn/.../fused/test_softmax.py` | **565 passed, 65 skipped** in 191.04s |
