# Step 7b — putting `numeric.h` on `reduce<>` with **no helper changes**

Experimental branch: `malimpic/experimental-reduce-helpers-migration-layernorm-2`,
branched from `malimpic/reduce-partial-scaler`.

A second, independent attempt at the same goal as
[step 7](step-7-layernorm-numeric-h-migration.md). Step 7 migrated the four *call sites* and grew the
helper to suit. This one leaves the helper completely alone and rewrites `numeric.h` itself to sit on
top of `reduce<>`.

**Result: it worked, with no helper-library changes at all.** `git diff --name-only` against the base
returns zero files under `ttnn/cpp/ttnn/kernel_lib/`, and `numeric.h` contains no `reduce_tile` or
`reduce_init` call.

## The one API change that was unavoidable

`compute_kernel_lib::reduce<>` takes its CB ids as **template** parameters:

```cpp
template <PoolType, ReduceDim, uint32_t input_dfb_id, uint32_t scaler_dfb_id, uint32_t output_dfb_id, ...>
```

`numeric.h` took `DataflowBuffer&` **runtime** references. There is no way to bridge that without
either changing `reduce<>` (ruled out here) or changing `numeric.h`'s signature. So the adapters are
now templated on ids and callers pass ids instead of objects — a mechanical change at the four call
sites, and arguably the more honest signature since the ids were `constexpr` at every call site
already.

## How each piece of `numeric.h` maps onto the helper

| `numeric.h` concern | Now |
|---|---|
| the `reduce_tile` loop | `reduce<>` |
| `reduce_init` / `reduce_uninit` / operand swap | inside `reduce<>` |
| DST acquire / commit / wait / pack / release | inside `reduce<>` |
| the hand-rolled partial-scaler rule (line 86, and the wait count at 169) | `ReducePartialScaler` + `detail::partial_scaler_for()` |
| the zero-arg `epilogue()` | `reduce<>`'s `post_reduce_op(dst)` |
| `WaitAtEndPolicy` | kept in `numeric.h`, one `wait_front(1)` after the call |
| block-granular wait/pop | **not expressible** — see below |
| multi-CB accumulation into one DST | **not expressible** — see below |

### Block granularity → per-tile streaming

`input_policy::pop` maps to `WaitAndPopPerTile`, `!pop` maps to `WaitUpfrontNoPop`.

The pop case is the compromise. `numeric.h` waited and popped `block_size` tiles at a time;
`WaitAndPopPerTile` does one at a time. `BulkWaitBulkPop` is not usable: the large-tensor kernel
deliberately sizes its input CB *below* the reduce extent, so a bulk wait would deadlock. So this
approach necessarily trades the block-granular handshake for a per-tile one — the exact cost step 4
predicted and step 7 avoided by adding a policy.

Whether that costs anything measurable is the point of this experiment; see Performance below.

### The padding drain

The normalization producers reserve and push `full_block_size()` for the final short block while only
filling the real tiles, so the CB carries `total_with_remainder()` entries. `reduce<>` pops exactly
what it reduces, so `numeric.h` now drains the difference itself
(`detail::drain_block_padding`). Without it the CB desynchronises across the enclosing `NCHt` loop.
This is bookkeeping the old code got for free from `sync_full_block`.

### Multi-CB accumulation → an `Accumulate` chain

`row_wise_mean_with_pre_add` folded two CBs into one DST register. With no helper change available,
it is now two `reduce<>` calls, the second using `Accumulate::at(out_dfb_id, 1)` to reload the first
pass's result before folding in the second CB and dividing.

This is the known trade: the intermediate `sum(x)` makes a round trip through `out_dfb_id`, which is
`Float16_b` whenever `fp32_dest_acc_en` is off. Step 7's `reduce_multi_input` avoided that by never
leaving DST. Whether it shows up in the tests is, again, measured rather than assumed.

## Dead parameter removed

`FLOAT32_REDUCTION` was a template parameter on all three functions, threaded into
`accumulate_compute_loop`, and referenced **zero** times in any body. Since the signatures were
changing anyway, it is gone. It never did anything; note that `reduce<>` has the working version of
what it presumably intended (`ReduceFp32Mode::Accurate`, which genuinely routes fp32 SUM through the
SFPU).

## Results

### Correctness

`test_layer_norm.py`: **265 passed** — identical to baseline and to approach 1.
`test_large_layer_norm_with_weight_bias_and_residual_input` + the small-kernel residual test (the
pre-add / `Accumulate`-chain path): **88 passed**.

### Performance

Same harness and baseline as step 7, so the three columns are directly comparable.

| case | baseline | approach 1 | approach 2 | Δ1 | Δ2 |
|---|---:|---:|---:|---:|---:|
| `layernorm.aligned_4096` | 651.5 | 648.5 | 648.9 | −0.45% | −0.40% |
| `layernorm.ragged_4095` | 650.2 | 649.9 | 649.6 | −0.05% | −0.10% |
| `layernorm_wb.aligned_4096` | 933.8 | 934.7 | 933.6 | +0.10% | −0.03% |
| `layernorm_wb.ragged_4095` | 934.8 | 934.4 | 935.4 | −0.05% | +0.06% |
| `layernorm_residual.aligned_4096` | 1092.9 | 1092.7 | 1094.8 | −0.01% | +0.17% |
| `layernorm_residual.ragged_4095` | 1096.9 | 1097.5 | 1096.5 | +0.06% | −0.03% |

**Both approaches are within ±0.5% of baseline and of each other.**

This refutes a prediction made in step 4 and repeated in step 7: that dropping from block-granular to
per-tile CB handshakes would cost measurable performance on the large-tensor kernel. It does not, at
these shapes. In hindsight the reason is unsurprising — `block_size` is 4 or 8, and the per-tile
handshake is small next to the `reduce_tile` work it guards. The `BlockWait*` policies added in
approach 1 are therefore **not justified by performance** on this evidence.

### Precision — the one place the two approaches actually differ

The `Accumulate` chain packs the first pass's row sum to `cb_ex` and reloads it, and `cb_ex` is
`Float16_b` whenever `fp32_dest_acc_en` is off. Measured against a float64 reference, same seed and
shapes on both branches (device compute is deterministic, so these are exact comparisons, not noise):

| shape | baseline mean\|err\| | approach 2 | Δ |
|---|---:|---:|---:|
| 32 × 1024 | 0.002065 | 0.002065 | **bit-identical** |
| 32 × 4095 | 0.002775 | 0.002848 | +2.6% |
| 1024 × 4096 | 0.002807 | 0.002918 | +4.0% |
| 32 × 4096 | 0.002661 | 0.002851 | +6.8% |

PCC drops correspondingly (0.99998592 → 0.99998533 at 32 × 4096).

Identical at W = 1024 and diverging as W grows is exactly the truncation signature: the wider the
row, the larger the intermediate sum, the more bfloat16 loses. The absolute error stays small and
every test passes at its tolerance, but the regression is systematic and grows with width.

Approach 1's `reduce_multi_input` avoids it entirely by never leaving DST.

## Which approach to keep

| | approach 1 (call sites + new policies) | approach 2 (numeric.h on stock reduce<>) |
|---|---|---|
| helper changes | +274 lines, 2 new features | **none** |
| `numeric.h` | untouched, left with no callers | rewritten, −30 lines, still the kernels' entry point |
| call-site churn | 4 sites rewritten inline | 4 sites change arguments only |
| performance | baseline | baseline |
| pre-add precision | preserved | **degrades with W** |

Approach 2 is the smaller and cheaper change, and it disproves the performance argument for the
`BlockWait*` policies. The remaining case for approach 1 rests entirely on `reduce_multi_input` and
the precision result above.

A third option this comparison suggests, not built here: take approach 2 and add **only**
`reduce_multi_input` to the helper — skipping the `BlockWait*` policies, which the measurement shows
buy nothing. That would keep the small diff and the precision.
