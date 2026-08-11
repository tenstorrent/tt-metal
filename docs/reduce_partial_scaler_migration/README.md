# ReducePartialScaler migration

Tracking document for the migration of kernels off hand-rolled partial-tile workarounds and off
direct `reduce_tile` calls, onto `compute_kernel_lib::reduce<>` + `ReducePartialScaler`.

## Background

When a reduce dimension is not a multiple of `TILE_DIM`, the padding lanes in the last tile pollute
the result. Historically each op solved this on its own. Three distinct workarounds existed in-tree:

1. **Mask the data** (the moreh family) — copy the last tile into DST, multiply it by a generated
   0/1 mask tile with `mask_tile`, pack the result into a scratch CB, then reduce that scratch CB
   separately and fold it into the main result with an accumulating second `reduce()` call.
   Costs: a mask CB, a scratch CB, an accumulator CB, a reader-side mask generator, and a two-phase
   split of every reduction.
2. **Hand-rolled dual scaler** (layernorm + `normalization/kernel_util/compute/numeric.h`) — emit two
   scaler tiles and select the partial one on the last tile. This is the same mechanism the helper
   now provides, implemented independently against raw `reduce_tile`.
3. **Pad the data** (`reduce_rm`) — pad the input during tilize so no ragged tile ever reaches the
   reduce. Out of scope for this migration.

`ReducePartialScaler` (added in `kernel_lib: support non-tile-aligned reduce dims via
ReducePartialScaler`) makes mechanism 2 a first-class part of the reduce helper, so ops can express
a ragged reduce dim in a single `reduce()` call with no extra buffers.

## Two properties verified on device before starting

Both were measured with the `toy_reduce_partial` op rather than reasoned about, because they decide
whether the moreh migrations are behaviour-preserving.

- **The partial scaler is correct for `PoolType::MAX`.** With all-negative input it returns the true
  negative maximum, not `0` — it genuinely excludes the padding rather than multiplying it by zero.
  This matters because `mask_tile` masks with **0**, so the workaround it replaces returns `0` for an
  all-negative MAX reduce. Migrating a MAX reduce is therefore a correctness *improvement*.
  (It is benign in softmax specifically, which only uses max for shift-invariant stability.)
- **Padding contents are irrelevant.** `+inf` and `NaN` in the pad lanes both produce clean, correct
  results for SUM and MAX. There is no `inf * 0 = NaN` poisoning, so no migration needs a guarantee
  about what the padding holds.

## A second structural limit, found on device in Step 8

`ReducePartialScaler` describes the last tile along the reduce dimension **of the reduce call itself**.
A kernel that folds several tiles into one by element-wise accumulation *before* reducing cannot use it:
lane `j >= partial_positions` of the accumulated tile holds the ragged tile's padding *and* valid data
from every earlier tile, so zeroing that lane discards real values (measured: pcc 0.86 on
`moreh_softmax_backward_w_large`). Those kernels must keep masking the ragged tile before accumulating.

This rules out the `_large` variants of the moreh softmax family — forward and backward — beyond their
max phase, and it is now documented on `ReducePartialScaler` itself. Step 9 migrates exactly that max
phase and leaves the exp/sum phase masked.

## Reachability of the stated end goal

"No kernels using partial-tile workarounds" is **not** fully reachable with the helper as it stands.
Eight kernels mask *both* axes (`generate_mask_h_w`) feeding a `REDUCE_SCALAR`-shaped reduce.
`ReducePartialScaler` rejects that by design: it selects one scaler tile along one axis, and a single
row/col tile cannot encode a 2-D corner mask. Closing that gap is a new helper feature (2-D partial
support), not a migration. Those eight are documented and deliberately left alone.

## Steps

| Step | Scope | Doc |
|---|---|---|
| 1 | `moreh_sum_h`, `moreh_mean_h` | [step-1-moreh-sum-mean.md](step-1-moreh-sum-mean.md) |
| 2 | `moreh_softmax_*` (small variants) | [step-2-moreh-softmax.md](step-2-moreh-softmax.md) |
| 3 | layernorm readers | [step-3-layernorm-readers.md](step-3-layernorm-readers.md) |
| 4 | `numeric.h` off direct `reduce_tile` | [step-4-numeric-h-analysis.md](step-4-numeric-h-analysis.md) — **blocked, no code change** |
| 5 | remaining direct `reduce_tile` callers | [step-5-remaining-reduce-tile-callers.md](step-5-remaining-reduce-tile-callers.md) — **assessment only** |
| 6 | perf comparison vs `main` | [step-6-perf-vs-main.md](step-6-perf-vs-main.md) |
| 7 | Phase 1 cleanups: `softmax_w` perf fix, a hang in the shared `ttnn` general softmax, dead RT args, `topk_router_gpt` | [step-7-phase1-cleanups.md](step-7-phase1-cleanups.md) |
| 8 | `moreh_softmax_backward` small kernels — full mask removal; why the `_large` ones cannot follow | [step-8-moreh-softmax-backward.md](step-8-moreh-softmax-backward.md) |
| 9 | `moreh_softmax_{h,w}_large` — max phase only (exp/sum phase blocked by step 8) | [step-9-moreh-softmax-large-max-phase.md](step-9-moreh-softmax-large-max-phase.md) |
