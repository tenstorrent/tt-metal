# Step 5 — remaining direct `reduce_tile` callers: assessment

**Outcome: assessment, no code change in this commit.** This records the state of every remaining
direct `reduce_tile` caller and whether `compute_kernel_lib::reduce<>` can express it, so the
remaining work is scoped rather than guessed at.

## Inventory

14 files call `reduce_tile` directly (excluding the helper itself and the `api/` declarations).

| File | `reduce_tile` calls | Verdict |
|---|---|---|
| `normalization/kernel_util/compute/numeric.h` | 1 (in a framework) | **Blocked** — see [step-4](step-4-numeric-h-analysis.md) |
| `layernorm/.../compute/layernorm_large_tensor.cpp` | via `numeric.h` | **Blocked** — consumer of the above |
| `layernorm/.../compute/layernorm_sharded.cpp` | 5 | Needs inspection — already mixes `reduce<>` and `reduce_tile` |
| `layernorm/.../compute/layernorm_sharded_pre_allgather.cpp` | — | Needs inspection |
| `layernorm/.../compute/layernorm_sharded_post_allgather.cpp` | 2 | Needs inspection |
| `layernorm_distributed/.../compute/layernorm_post_allgather.cpp` | 2 | Needs inspection |
| `experimental/topk_router_gpt/device/kernels/compute.cpp` | 2 | **Clean candidate** — see below |
| `experimental/ccl/rms_allgather/.../rms_compute.cpp` | 3 | Needs inspection |
| `experimental/deepseek_prefill/moe_grouped_topk/.../moe_gate_common_compute.hpp` | 1 | Likely clean (single tile) |
| `experimental/deepseek_prefill/per_token_cast_to_fp8/.../compute_per_token_cast_to_fp8.cpp` | 1 | Likely clean (single tile) |
| `transformer/sdpa/device/kernels/compute/compute_common.hpp` | 2 | Needs inspection — indexed `i * cols + j` into a 2-D block |
| `reduction/generic/.../compute/reduce_{h,hw,w}_neg.cpp` | 1 each | Needs inspection |

## The clean candidate

`topk_router_gpt/compute.cpp` is the textbook case — two single-tile reduces written out longhand:

```cpp
tile_regs_acquire();
reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_softmax_tmp_id, cb_bcast_scaler_id, cb_reduce_scalar_id);
reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_softmax_tmp_id, cb_bcast_scaler_id, 0, 0, 0);
reduce_uninit(cb_reduce_scalar_id);
tile_regs_commit();
tile_regs_wait();
pack_tile(0, cb_reduce_scalar_id);
tile_regs_release();
cb_reduce_scalar.push_back(1);
```

This is exactly `reduce<MAX, REDUCE_ROW, in, scaler, out>(ReduceInputBlockShape::single())`. The one
wrinkle is that the caller currently does its own `reserve_back` / `push_back` around the block, which
the helper also does — so the surrounding CB calls have to come out at the same time, not just the
`reduce_tile` line.

## Why this is an assessment and not a migration

These are independent one-off conversions in ops with varying (and in the experimental cases, thin)
test coverage. The Step 2 finding — a suite that looked green while providing no coverage of the
changed path — makes it clear that converting a kernel without first establishing that its tests
exercise the reduce is not worth much. Each of these needs its own coverage check and test run, which
is per-file work rather than a sweep.

Given the explicit request to finish with a perf comparison against `main`, measuring the three
migrations that are actually done was the better use of the remaining pass than adding a fourth
untested conversion.

## Recommended order when this is picked up

1. `topk_router_gpt` — clean, self-contained, establishes the single-tile pattern.
2. The two `deepseek_prefill` single-tile calls — same pattern.
3. `reduce_{h,hw,w}_neg.cpp` — these are the generic reduction op's own kernels, so they have real
   test coverage in `tests/ttnn/unit_tests/operations/reduce/`.
4. `layernorm_sharded*` and `layernorm_post_allgather` — these already call `reduce<>` elsewhere in
   the same file, so the remaining `reduce_tile` calls are probably patterns the helper could not
   express at the time; worth checking whether the partial-scaler parameter closes that gap.
5. `sdpa/compute_common.hpp` — the indexed `i * cols + j` access into a 2-D block maps to
   `ReduceInputBlockShape::of(rows, cols)` with a row stride, but SDPA is performance-critical and
   should not be touched without a perf run.
6. `numeric.h` + `layernorm_large_tensor` — only after the helper gains a block-wise input policy.
