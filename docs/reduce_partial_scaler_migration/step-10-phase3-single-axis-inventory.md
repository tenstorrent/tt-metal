# Step 10 — the remaining single-axis mask kernels: one migration, the rest blocked

Phase 3 of the plan listed roughly a dozen single-axis mask-the-data moreh kernels as
"same pattern as Step 1, one op per commit". Applying the step-8 rule to each of them first turns that
into **one** migration. This document is mostly the negative result, because that is the part worth
recording.

## The filter

Step 8 established: a partial scaler describes the last tile along the reduce dimension **of the
reduce() call**. A kernel that folds several tiles into one by element-wise accumulation *before*
reducing cannot use one, because lane `j >= partial_positions` of the accumulator holds valid data from
the earlier tiles as well as the ragged tile's padding.

Almost every remaining moreh reduce is written in exactly that shape: build `f(x)` per tile, fold it
into a one-tile accumulator, then `reduce(single())` the accumulator.

## Inventory

| Kernel | Reduce input | Verdict |
|---|---|---|
| `moreh_norm_h`, `moreh_norm_w` (main variants) | `cb_xpowadd`, built by `add_tiles(cb_correct_xpow, cb_xpowadd)` per tile | **Blocked** — accumulate-then-reduce |
| `moreh_norm_h`, `moreh_norm_w` (`ord_other` variants) | `dfb::cal`, built by `add_tiles` / `binary_max_tile` per tile | **Blocked** — same |
| `moreh_layer_norm` small (and large) | `cb_xsum`, `cb_xmm2sum`, built by `add_tiles` per tile | **Blocked** — same |
| `moreh_layer_norm_backward` gamma/beta grad | `cb_ydyadd`, `cb_dyadd` | **Blocked** — same |
| `moreh_abs_pow` | — | **N/A** — the kernel has no reduce at all; its `mask_tile` shapes an elementwise output, so `ReducePartialScaler` does not apply |
| `moreh_group_norm` (forward) | — | **N/A** — does not use `mask_tile`; it was on the Phase 3 list only because its *backward* readers call `generate_mask_h_w` |
| `moreh_bias_backward_h` | one tile per `reduce()` call, accumulated **between** calls via `Accumulate` | **Migrated** — see below |

The blocked ones are not close calls: for `moreh_norm_h` the accumulator is literally the reduce input
(`reduce<..., cb_xpowadd, cb_one, cb_xpowsum>(single())`), and the ragged tile was folded in 30 lines
earlier. Their masks stay.

## The one migration: `moreh_bias_backward_h`

This kernel is the counter-example that makes the rule precise. It does **not** accumulate before the
reduce — it reduces one tile per `reduce()` call and accumulates *between* calls:

```cpp
const auto reduce_accum = compute_kernel_lib::Accumulate::at(cb_intermed1, num_tile_done);
reduce<SUM, REDUCE_COL, cb_in0, cb_scaler, cb_intermed1>(single(), contiguous(), reduce_accum);
```

So "the last tile along the reduce dim of this call" is that single tile, and the ragged H tile of each
batch can take scaler tile 1 while every other tile takes tile 0. That is exactly the per-call form the
`ReducePartialScaler` docs prescribe for accumulating reduces:

```cpp
const auto reduce_partial = (do_mask_h && last_row) ? ReducePartialScaler::last_tile_at(1)
                                                    : ReducePartialScaler::none();
```

**`mask_w` stays.** The reduce is over H, so W padding columns survive into the output's padding lanes;
zeroing them is observable output state, the same argument that kept the forward softmax exp-phase mask
in Step 2. The `mask_h_w` CB therefore stays too (tile 1, the W mask, is still used) — but the H-masking
branch, and with it the copy-mask-restage detour through `cb_intermed0` for every ragged-H tile, is
gone. When only H is ragged the kernel now takes the plain no-mask path for every tile.

Host side: `in1_t` (the scaler CB) is `do_mask_h ? 2 : 1`, and the reader emits the pair under the same
condition.

## Coverage

`test_moreh_linear.py`'s bias-grad cases already cover ragged H — but every one of them has `Ht == 1`
(H ∈ {2, 4, 31} or aligned 32/96), so the last H tile is never preceded by full tiles. That is precisely
the configuration that distinguishes a correct per-call partial scaler from the broken
accumulate-then-reduce form step 8 caught, so a case with `H = 95` (`Ht = 3`, 31 valid rows in the last
tile) was added.

## Test results

| Suite | Result |
|---|---|
| `test_moreh_linear.py` | **219 passed, 212 skipped** in 91.56s |
| `-k 95` (the new shape only) | **86 passed** |

## Consequence for the migration as a whole

"No kernels using partial-tile workarounds" is now bounded by two separate limits, not one:

1. **2-D masks** (`generate_mask_h_w` feeding a `REDUCE_SCALAR`) — needs the 2-D partial scaler feature.
2. **Accumulate-then-reduce** — needs nothing; it is simply outside what a scaler mask can express.
   The kernels in the table above will keep their masks permanently unless they are restructured to
   reduce the tiles directly (which would cost the L1 residency that made them accumulate in the first
   place).

Limit 2 is the larger group, and it was invisible until step 8 measured it.
