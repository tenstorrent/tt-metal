# Cumulative `cb_wait_front` Pattern Search

**Date**: 2026-04-30
**Conclusion**: 6 production kernels use cumulative wait. Phase 2 claim **REFUTED**.

## Pattern

```cpp
for (uint32_t var = 0; var < MAX; var += STEP) {
    cb_wait_front(cb, var + STEP);  // count grows each iter
    // process block
}
```

Semantics: cumulative count grows; by end of loop, all MAX tiles fronted. Reduces per-iter sync overhead by pre-staging entire buffer before bulk reduce.

## Findings (6 kernels)

| # | File | Line | Call | Use case |
|---|---|---|---|---|
| 1 | `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp` | 52 | `cb_wait_front(cb_inp, wt + blk)` | RMSNorm variance staging |
| 2 | `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather_2d.cpp` | 58 | `cb_wait_front(cb_inp, wt + blk)` | RMSNorm 2D distributed |
| 3 | `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather.cpp` | 47 | `cb_wait_front(cb_inp, wt + blk)` | LayerNorm E[x²] staging |
| 4 | `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_2d.cpp` | 49 | `cb_wait_front(cb_inp, wt + blk)` | LayerNorm 2D distributed |
| 5 | `ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/device/kernels/compute/layernorm_post_allgather_welford.cpp` | 136, 164 | `cb_wait_front(cb_gamma, col_tile + block_size)` | DIT layernorm gamma/beta broadcast |
| 6 | `ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_post_allgather.cpp` | 133 | `cb_wait_front(weight_cb, col_tile + block_size)` | RMSNorm + ROPE fused weights |

## Eltwise-adjacent count

4 of 6 are in eltwise-adjacent normalization (RMSNorm + LayerNorm pre-allgather variants). Remaining 2 in experimental transformer fusions.

## Implication for helper

`CopyTilePolicy::CumulativeWait` (or equivalent) IS needed if helper aspires to cover normalization eltwise inner steps. Phase 3 proposal must add this policy.

Pattern signature: `wait_count = base + step` per iteration, pop at end of outer block. Helper-side support requires:
- Wait shape: cumulative count (base + i*step)
- Pop shape: typically upfront-end at loop completion (after all MAX tiles fronted)
- Index mode: BlockIter — caller reads tile `i` within the cumulative window

Compatible with `WaitUpfrontPopAtEnd` semantics if wait is moved up-front. If kernel intentionally interleaves wait+compute, need a `CumulativeWait` policy.
