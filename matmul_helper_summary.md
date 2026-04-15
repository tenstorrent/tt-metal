# Matmul Helper Library Summary

**Branch:** `wransom/matmul_op_integ_verf` | **Hardware verified:** Blackhole P100A

## What This Library Does

`ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.{hpp,inl}` provides DST-safe matmul helpers for compute kernels. The helpers encapsulate the full DST register lifecycle (`tile_regs_acquire/commit/wait/release`) and CB management (`cb_reserve_back/push_back`) so kernel authors cannot introduce acquire/release bugs that cause device hangs.

## Public API (DST-managed, safe for kernel use)

| Helper | What it does | Used by |
|--------|-------------|---------|
| `matmul_accumulate_and_pack` | acquire + [reload] + accumulate + PostComputeFn + pack + release | conv, minimal_matmul |
| `matmul_compute_one_tile` | Per-tile streaming matmul with full CB management | moreh_matmul, bmm.cpp |
| `matmul_compute_inner_block` | Double subblock loop with spill/reload | bmm_large_block_zm |
| `matmul_reduce_w_and_pack` | Reduce-W accumulation + pack | reduce_w.cpp |
| `matmul_reduce_subblock_inplace` | Per-subblock inplace reduce | SDPA matmul_reduce |
| `matmul_single_and_pack` | Single matmul + PostComputeFn + pack | SDPA normalize_row |
| `matmul_and_pack_absolute` | Subblock matmul + absolute-offset pack | SDPA streaming |
| `matmul_blocks_absolute` | Full blocked matmul + absolute-offset pack + CB management | SDPA non-streaming |
| `matmul` | Full automated blocked matmul | bmm.cpp, bmm_large_block_zm |
| `matmul_init`, `matmul_init_short`, etc. | Hardware init/reinit | All matmul kernels |

## PostComputeFn / PostPackFn Callbacks

Helpers accept callback functors that fire at specific DST lifecycle points:
- **PostComputeFn**: fires after accumulation, before commit. For fused SFPU ops (relu, recip) or elementwise ops (mask addition) on tiles in DST.
- **PostPackFn**: fires after pack, before release. For hardware semaphore posting.

**Rules**: No DST locking calls, no pack_tile, no matmul calls inside functors. See header docs for full constraints.

## detail:: Namespace (internal building blocks)

Functions in `detail::` do NOT manage DST. They exist for composition inside the public helpers. Kernel code that calls `detail::` functions is flagged for future migration to DST-managed helpers.

## Test Results (Blackhole P100A)

1,505 passed, 0 failed across: SDPA decode+prefill, test_matmul, conv2d, moreh matmul+mean+sum, reduce sum.
