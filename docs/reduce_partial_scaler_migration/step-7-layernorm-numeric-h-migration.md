# Step 7 — migrating the layernorm `numeric.h` calls onto the reduce helpers

Experimental branch: `malimpic/experimental-reduce-helpers-migration-layernorm`.

Step 4 recorded `numeric.h` as blocked on two missing helper capabilities. This step builds both and
migrates all four call sites, so the conclusion there is now superseded: it was a *missing feature*,
not an impossibility.

## What was missing, and what was added

### 1. Block-granular CB consumption

`accumulate_compute_loop` walks the reduce dimension in chunks of a tuned compile-time `block_size`,
waiting (and optionally popping) one chunk at a time. None of the four existing `ReduceInputPolicy`
values expresses that:

- `BulkWaitBulkPop` waits for the whole reduce dimension — impossible in `layernorm_large_tensor`,
  whose CBs are deliberately capped **below `Wt`** (`layernorm_op_multi_core.cpp:292-296`); with
  `FUSE_PRE_ADD`, `in0_t` is only `2 * block_size`.
- `WaitAndPopPerTile` is correct but degrades one handshake per block into one per tile, on exactly
  the kernel that needs batching most.

Added `ReduceInputPolicy::BlockWaitBlockPop` / `BlockWaitNoPop` plus a runtime `ReduceBlockConfig`
(`block_size`, `sync_full_block`).

`sync_full_block` is the subtle half. The layernorm producers pad their pushes to a whole number of
blocks — `layernorm.cpp:154-162` reserves and pushes `full_block_size()` while packing only
`block.size()` real tiles. A consumer that pops only the real tiles would leave the padding behind
and desynchronise the CB across the `NCHt` loop. The flag makes the last short block wait and pop as
a whole block while still reducing only its real tiles, so the padding never reaches the result.

Implemented for `REDUCE_ROW` (the dim all four calls use) via a shared
`reduce_row_block_walk()`, which `reduce()`'s ROW body and `reduce_multi_input()` both call — so the
block/pop/partial-scaler rules exist in exactly one place. `REDUCE_COL` / `REDUCE_SCALAR` and the
Int32 SFPU path `static_assert` rather than silently doing the wrong thing.

### 2. Multi-input accumulation into one DST

`row_wise_mean_with_pre_add` folds two CBs into a single DST register, runs one epilogue, and packs
one tile. It computes `E[x+b]` **without ever materialising `x+b`**, by linearity:

```
E[x + b] = (1/N) · Σ(xᵢ + bᵢ) = (1/N) · (Σxᵢ + Σbᵢ)
```

That is the whole point of the fused form: the large-tensor kernel has no room for a `Wt`-sized
temporary of the elementwise sum.

Chaining two `reduce()` calls through `Accumulate` would work but round-trips `Σx` through the output
CB between them — an extra pack, an extra `copy_tile` reload, and, when `fp32_dest_acc_en` is off,
a **truncation of the partial sum to bfloat16** (`cb_ex` is `fp32_dest_acc_en ? Float32 : Float16_b`).
That is a numerical regression on the largest-magnitude intermediate in the computation.

So `reduce_multi_input()` was added instead: it takes an array of input CB ids, reduces each in turn
into the same DST accumulator, then runs the post-op once and packs once. The running sum never
leaves DST.

Its correctness precondition is documented on the function: folding partial reductions must equal
reducing the concatenation, i.e. SUM/AVG (linear) or MAX/MIN (associative). It is **not** a general
`reduce(f(a, b))`.

## The four call sites

| Site | Before | After |
|---|---|---|
| `layernorm.cpp:180` E[x] | `row_wise_mean<…, FullBlockWithoutPop>` | `reduce<…, BlockWaitNoPop>` + `ReduceBlockConfig::of(block_size)` |
| `layernorm.cpp:244` Var[x] | `row_wise_mean<…, FullBlockWithPop>` | `reduce<…, BlockWaitBlockPop>` |
| `large_tensor.cpp:121` E[x] | `row_wise_mean<…, FullBlockWithPop>` | `reduce<…, BlockWaitBlockPop>` |
| `large_tensor.cpp:114` E[x+b] | `row_wise_mean_with_pre_add<…>` | `reduce_multi_input<…, 2, BlockWaitBlockPop>` |

Two caller-side details had to move out of the framework and into the call sites:

- **The `1/N` epilogue.** `row_wise_mean` hardcoded it; `reduce()` takes it as `post_reduce_op`, so
  each site now passes `[](uint32_t dst) { scale_dest(dst, bit_cast<uint32_t>(1.0f / W)); }`. This
  makes the divisor visible at the call site rather than buried in a wrapper.
- **`WaitAtEndPolicy::WAIT`.** `numeric.h` waited on its own output after pushing; `reduce()` does
  not. `layernorm.cpp`'s E[x] site and `large_tensor.cpp`'s mean site both consume `cb_ex`
  immediately afterwards and relied on that wait, so each gained an explicit `dfb_ex.wait_front(1)`.
  (`layernorm.cpp`'s Var site already re-waited, so it needed nothing.)

The partial-scaler selection, previously hand-rolled at `numeric.h:86` and again at
`numeric.h:169-171`, is now a single `constexpr auto partial_scaler = (W % tile_width > 0) ?
last_tile_at(1) : none()` per kernel.

## Issues hit

**One compile error, one round.** `reduce_row_block_walk` was defined outside `namespace detail` but
called as `detail::reduce_row_block_walk`. The kernel JIT reported it as 122 test failures, all of
them the same `trisc0 build failed`; a stray cascade error (`no match for 'operator>'` between
`ReduceInputPolicy` and `ReduceBlockConfig`) was just the parser reading `<` as less-than after the
name lookup failed. Fixed by dropping the qualification.

Worth noting for anyone reading the earlier run: a JIT compile failure surfaces as a large number of
*correctness* failures, not as a build error, because the kernel is only compiled when a test runs it.

**A tainted intermediate test run.** The first clean 265-pass result was collected while the
`large_tensor` edits were mid-flight, so it could not be attributed to the `layernorm.cpp` changes
alone. Re-run after all four sites were in place rather than trusting it.

## Correctness

| Suite | Result |
|---|---|
| `test_layer_norm.py`, all four calls migrated | **265 passed** |
| `test_large_layer_norm_with_weight_bias_and_residual_input` (the `reduce_multi_input` path) | **28 passed** |
| moreh softmax + mean + `toy_reduce_partial` (other `reduce<>` callers, shared helper changed) | **219 passed, 104 skipped** |

265 is exactly the pre-migration baseline, so nothing regressed and nothing silently stopped running.

Coverage was checked before trusting it: `test_layer_norm.py` parametrises `w` over 24, 42, 127,
519, 31, 487, 3821 and pairs like (19, 2865), so the ragged path — the one the partial scaler and
`sync_full_block` actually change — is genuinely exercised.

## Performance

Same harness as Step 6, measured against `malimpic/reduce-partial-scaler` (the branch this one
forked from) so the delta isolates *this* migration rather than re-measuring the earlier work.

| case | migrated | baseline | delta |
|---|---:|---:|---:|
| `layernorm.aligned_4096` | 648.5 | 651.5 | −0.45% |
| `layernorm.ragged_4095` | 649.9 | 650.2 | −0.05% |
| `layernorm_wb.aligned_4096` | 934.7 | 933.8 | +0.10% |
| `layernorm_wb.ragged_4095` | 934.4 | 934.8 | −0.05% |
| `layernorm_residual.aligned_4096` | 1092.7 | 1092.9 | −0.01% |
| `layernorm_residual.ragged_4095` | 1097.5 | 1096.9 | +0.06% |

**Every layernorm case is within ±0.5%** — inside the 0.1–0.5% run-to-run noise established in Step 6.
The block-granular policy reproduces `numeric.h`'s pipelining exactly, and `reduce_multi_input`
reproduces the fused pre-add without the extra pack/reload that an `Accumulate` chain would have
cost. `layernorm_residual` is the pre-add case and it is flat, which is the specific thing that
would have regressed had the two-`reduce()` workaround been used instead.

The non-layernorm rows in the same run (moreh, softmax) moved by −1.4% to +1.6%, all within the
noise measured for those cases in Step 6 and none of them touched by this branch beyond the shared
helper.

## Status of the Step 4 conclusion

Step 4 recorded `numeric.h` as blocked. That is now **superseded**: both missing capabilities were
buildable, and with them all four call sites migrate with no correctness or performance cost.
`numeric.h` has no remaining callers.

What Step 4 got right is that a *partial* migration would have been bad — the value came from moving
all four together, which is only possible once both capabilities exist.

Not done here, and still open: `layernorm_large_tensor.cpp:195-198` open-codes its variance reduce
with its own `reduce_init`/`reduce_tile` loop and its own copy of the partial-scaler rule. It is a
CB-accumulator reduce across blocks, which the new `BlockWaitBlockPop` policy could likely express in
one call with no accumulator CB at all — but that is a separate change with its own measurement, and
this branch was scoped to the four `numeric.h` calls.
