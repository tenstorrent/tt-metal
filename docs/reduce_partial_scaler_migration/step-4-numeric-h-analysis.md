# Step 4 — `numeric.h`: analysis, and why it is not migrated

**Outcome: no code change.** This step is an analysis. Migrating
`normalization/kernel_util/compute/numeric.h` onto `compute_kernel_lib::reduce<>` is blocked on two
capabilities the helper does not have, and a partial migration would leave the file with two reduce
mechanisms instead of one — strictly worse than the status quo.

## Why it was the headline target

`numeric.h` is the largest direct `reduce_tile` caller in the tree, and it contains an independent
reimplementation of the partial-scaler rule this whole migration is about:

```cpp
// numeric.h:86
const auto scaler_tile_idx = block.to_global(j) == num_tiles - 1 && last_tile_partial ? 1 : 0;
```

That is `ReducePartialScaler::last_tile_at(1)` written by hand. Unifying it would remove the last
duplicate of the mechanism and take two more kernels (`layernorm.cpp`, `layernorm_large_tensor.cpp`)
off direct `reduce_tile`.

## Blocker 1 — accumulation across multiple CBs into one DST

`row_wise_accumulate_with_epilogue` acquires DST **once**, accumulates the input CB into `dst0`, then
accumulates every additional CB into the *same* `dst0`, runs a caller-supplied epilogue, and packs
**one** tile:

```cpp
tile_regs_acquire();
accumulate_compute_loop(...);   // dfb_in, then each of dfb_additional..., all into dst0
epilogue();                     // e.g. multiply by 1/N
tile_regs_commit(); tile_regs_wait();
pack_tile(dst0, dfb_out);
tile_regs_release();
```

`compute_kernel_lib::reduce<>` owns its own DST lifecycle: it acquires, reduces one block, runs
`post_reduce_op`, packs, and releases. There is no way to say "reduce CB A and CB B into the same
accumulator before packing". `Accumulate` is not a substitute — it round-trips the partial through a
CB between calls, which is both different numerically and slower than staying resident in DST.

Used by `row_wise_mean_with_pre_add` (`layernorm_large_tensor.cpp:114`), which sums two CBs to compute
`E[x + b]` in one pass.

## Blocker 2 — block-wise streaming granularity

The accumulate loop waits and pops in units of `block_size` (a compile-time arg, i.e. the tuned
pipelining granularity):

```cpp
for (auto block : generic::blocks(num_tiles, block_size)) {
    dfb.wait_front(num_previous_tiles + curr_block_size);
    for (auto j : block.local()) { reduce_tile(...); }
    if constexpr (pop_input) { dfb.pop_front(curr_block_size); }
}
```

Mapping onto `ReduceInputPolicy`:

| numeric.h policy | closest `ReduceInputPolicy` | correct? | cost |
|---|---|---|---|
| `*WithoutPopPolicy` (pop=false) | `WaitUpfrontNoPop` | yes — the cumulative waits already reach all tiles by the last block | none |
| `*WithPopPolicy` (pop=true) | `WaitAndPopPerTile` | yes — 1-tile residency is strictly more conservative than `block_size` | **finer-grained CB handshake per tile instead of per block** |

`BulkWaitBulkPop` is *not* usable for the pop case: it waits for all `num_tiles` upfront, which for
`layernorm_large_tensor` — the kernel that exists precisely because the row does not fit in L1 —
would require residency the CB was never sized for.

So the pop path can only be expressed by dropping to per-tile synchronisation. That is a real
pipelining regression on exactly the large-tensor kernel that most needs the batching, and this
migration has no performance mandate to spend that.

## What would unblock it

Two additive helper features, in increasing order of usefulness:

1. **A block-wise input policy** — e.g. `BlockWaitBlockPop` carrying a `block_size`, waiting and
   popping in that unit. This alone unblocks every single-CB call site.
2. **Multi-input accumulation** — a `reduce<>` form that takes several input CBs and folds them into
   one DST before the post-op and the single pack.

Neither is a migration; both are helper features and should be scoped as such.

## Why not migrate partially

Three of the four call sites are single-CB, and two of those are no-pop, which maps cleanly. It is
tempting to convert those and leave `row_wise_mean_with_pre_add` on `reduce_tile`.

That is rejected: `numeric.h` is one small framework whose whole value is that every normalization
kernel reduces the same way. Splitting it into "some paths go through `reduce<>`, one path still
hand-rolls `reduce_tile`, and the partial-scaler rule is written twice in the same file" is worse for
a reader than the current single consistent implementation. The file should move as a unit, once the
helper can express what it does.

## Consequence for the stated end goal

"No kernels calling `reduce_tile` directly" is **not** reachable today. `numeric.h` and its two
layernorm consumers stay on `reduce_tile` until the helper gains at least the block-wise policy.
The remaining direct callers are assessed separately in Step 5.
