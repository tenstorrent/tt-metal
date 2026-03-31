# Phase 1 Instance 3: Existing Helper & Composition Pattern Analysis

## Part A -- Existing kernel_lib Helpers

### A1. matmul_block_helpers.hpp/inl

**Public API:**
```cpp
template <uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t interm_cb,
          bool transpose = false, typename PostComputeFn = matmul_block_config::NoPostCompute>
ALWI void matmul_block(
    uint32_t block_w, uint32_t in0_num_subblocks, uint32_t in1_num_subblocks,
    uint32_t num_k_blocks, uint32_t out_subblock_h, uint32_t out_subblock_w,
    uint32_t batch = 1, PostComputeFn post_compute = {});
```

**Template params:** 4 CB indices (compile-time), transpose bool, PostComputeFn functor type.

**Runtime params:** 5 independent dimensions + batch count. Derived quantities (out_num_tiles, in0_subblock_num_tiles, etc.) computed internally per PR feedback.

**Feature variant handling:**
- `transpose` -- compile-time template bool, passed through to `ckernel::matmul_block()` and `mm_block_init()`
- `PostComputeFn` -- compile-time functor type. Called once per output sub-block on the last K-block, before packing. Receives `out_num_tiles` as argument. Tiles sit in DST[0..num_tiles-1]. Default: `NoPostCompute` (empty functor that compiles away).

**Init/uninit lifecycle:**
- Calls `mm_block_init(in0_cb, in1_cb, interm_cb, transpose, out_subblock_w, out_subblock_h, block_w)` once at function start. Note: packer is configured for `interm_cb` (not `out_cb`) because spill path packs to interm.
- No explicit uninit -- the matmul LLK API has no `mm_uninit`.
- Re-init via `mm_block_init_short_with_dt()` during spill/reload (after `copy_tile_to_dst_init_short_with_dt` corrupts matmul unpacker state).

**CB synchronization strategy:**
- `cb_wait_front(in0_cb, in0_block_num_tiles)` + `cb_wait_front(in1_cb, in1_block_num_tiles)` -- waits for full input blocks per K-iteration.
- `cb_pop_front()` for both inputs at end of each K-block.
- Output: `cb_reserve_back(out_cb, out_num_tiles)` + `pack_tile` + `cb_push_back(out_cb, out_num_tiles)` per sub-block on last K-block.
- Spill: `cb_reserve_back(interm_cb, out_num_tiles)` + `pack_tile` + `cb_push_back(interm_cb, out_num_tiles)` for intermediate results.
- Special: on the first non-last K-block (`block == 0`), reserves out_cb space to prevent interm from overwriting (they share L1 memory). Accumulates `out_num_tiles_to_wait`.

**Composition with other helpers:** None. matmul_block is fully self-contained. It does not call or delegate to any other kernel_lib helper.

**PostComputeFn enables:** Any per-sub-block SFPU operation (relu, gelu, etc.) on DST tiles before packing. The functor receives the tile count, not individual tile indices.

**PostComputeFn cannot express:**
1. Operations requiring additional CB data (e.g., bias addition needs a bias CB)
2. Per-tile operations (receives total tiles, not individual dst indices)
3. Operations that need different packing targets (always packs to out_cb)
4. Multi-phase operations (e.g., pack to staging CB, then do bias add, then pack to output)
5. Operations that need data format reconfiguration (the matmul format is still active)

---

### A2. tilize_helpers.hpp/inl

**Public API:**
```cpp
template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb,
          tilize_config::InitUninitMode init_uninit_mode = InitAndUninit,
          tilize_config::WaitMode wait_mode = WaitBlock,
          tilize_config::ReconfigureRegisterDatatypeMode reconfig_mode = UnpackAndPackReconfigure,
          tilize_config::Fp32Mode fp32_mode = Fast>
ALWI void tilize(uint32_t num_blocks, std::optional<uint32_t> total_input_pages = std::nullopt);
```

**Template params:** Block width (compile-time dimension), 2 CB indices, 4 mode enums.

**Runtime params:** `num_blocks` (height in tiles), optional `total_input_pages` for asymmetric CB pages.

**Feature variant handling:**
- Fast vs standard tilize: auto-detected at compile time from `can_use_fast_tilize<>()` which checks tile size (32x32), format (Float32/Float16_b), and sync mode (not full-sync). Controlled by `Fp32Mode::Lossless` override.
- Asymmetric CB pages: runtime branch when `total_input_pages.has_value()` -- input CB has non-tile pages (e.g., row-sized pages).
- All mode selections are compile-time via `if constexpr`.

**Init/uninit lifecycle:** `InitUninitMode` enum with 4 values. Enables back-to-back tilize calls with shared init/uninit (InitOnly -> Neither -> UninitOnly). Implementation is `if constexpr` on the mode value.

**CB synchronization:** `WaitMode` enum: WaitBlock (per-block wait), WaitUpfront (wait all before loop), NoWait (caller manages). Uses `experimental::CircularBuffer` wrapper objects for wait/pop/reserve/push.

**Composition:** Standalone. Does not call other helpers. Uses `cb_helpers.hpp` utilities for validation.

---

### A3. untilize_helpers.hpp/inl

**Public API:**
```cpp
template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb,
          untilize_config::InitUninitMode init_uninit_mode = InitAndUninit,
          untilize_config::WaitMode wait_mode = WaitBlock,
          untilize_config::ReconfigureRegisterDatatypeMode reconfig_mode = UnpackAndPackReconfigure>
ALWI void untilize(uint32_t num_blocks);
```

Follows the same pattern as tilize. Additionally provides standalone `untilize_init<>()` and `untilize_uninit<>()` wrapper functions for manual lifecycle control.

**Unique feature:** Automatic block splitting when `block_width_tiles > DEST_AUTO_LIMIT`. Computes optimal sub-block width via `compute_num_blocks()` (largest divisor of total width <= DEST limit). Two dispatch paths:
- Single-pass: `pack_untilize_block<block_width_tiles, block_width_tiles>(...)` when width fits in DEST.
- Block-based: splits into sub-blocks, calls `pack_untilize_block<sub_block_width, block_width_tiles>(...)` in inner loop.

---

### A4. reduce_helpers_compute.hpp/inl

**Public API:**
```cpp
template <PoolType reduce_type, ReduceDim reduce_dim,
          ReduceInputPolicy input_policy = WaitAndPopPerTile,
          ReduceDataFormatReconfigMode reconfig_mode = INPUT_AND_OUTPUT,
          typename AccumulateT = NoAccumulation, typename PostReduceOp = NoOp>
ALWI void reduce(
    uint32_t input_cb, uint32_t scaler_cb, uint32_t output_cb,
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout = contiguous(),
    AccumulateT accumulate = {}, PostReduceOp post_reduce_op = {});
```

**Template params:** reduce type (SUM/AVG/MAX), reduce dim (ROW/COL/SCALAR), input policy, reconfig mode, accumulation type, post-reduce op type.

**Runtime params:** 3 CB IDs, ReduceInputBlockShape struct (rows, cols, batches), ReduceInputMemoryLayout struct (row_stride), accumulation config, post-reduce callback.

**Feature variant handling -- the most sophisticated in kernel_lib:**
- **Input policy** (4 modes): WaitAndPopPerTile (streaming), BulkWaitBulkPop (bulk), WaitUpfrontNoPop (persistent), NoWaitNoPop (caller-managed). Selected via template param, dispatched via `if constexpr` on helper predicates (`waits_per_tile()`, `waits_bulk()`, etc.).
- **Accumulation**: Type-based dispatch. When `AccumulateT = Accumulate`, accumulation code is compiled in. When `AccumulateT = NoAccumulation`, it compiles away via `if constexpr(is_accumulate_v<AccumulateT>)`. Accumulate carries config (CB, DST index) + iteration index (0 = first/skip reload, >0 = reload from accumulator CB).
- **Reduce dim**: Three completely different code paths in one function, selected by `if constexpr`. REDUCE_COL adds auto-chunking by DEST_AUTO_LIMIT.
- **FP32 accumulation**: Auto-detected from JIT define via `get_fp32_dest_acc_enabled()`.

**Init/uninit lifecycle:** `reduce_init()` at start, `reduce_uninit()` at end. No fine-grained lifecycle control (unlike tilize/untilize).

**CB synchronization:** Policy-based, matching the 4 input policy modes. Output synchronization also varies: pop modes do per-tile reserve/push, no-pop modes do bulk reserve upfront + bulk push at end.

**Composition:** Uses `copy_tile_to_dst_init_short_with_dt()` + `copy_tile()` for accumulator reload. After reload, must call `reduce_init_short_with_dt()` to restore reduce unpacker/math config (because copy_tile corrupts SRCA config). This is the most complex inter-operation reconfiguration in kernel_lib.

**PostReduceOp:** Called per output tile with `dst_idx`. For REDUCE_ROW: once per row (dst_idx=0). For REDUCE_COL: once per column in chunk (dst_idx in [0, current_chunk)). Not called for REDUCE_SCALAR. Enables transformations like `recip_tile` for softmax.

---

### A5. binary_op_helpers.hpp/inl

**Public API:**
```cpp
template <BinaryOpType op_type, BroadcastDim bcast_dim = NONE,
          BinaryInputPolicy input_a_policy = WaitAndPopPerTile,
          BinaryInputPolicy input_b_policy = input_a_policy,
          BinaryOutputPolicy output_policy = PerTile,
          BinaryDataFormatReconfig reconfig = INPUT_AND_OUTPUT,
          bool init = true, typename PostOp = NoOp, typename AccumT = NoAccumulation>
ALWI void binary_op(uint32_t icb_a, uint32_t icb_b, uint32_t ocb,
                     BinaryInputBlockShape shape, PostOp post_op = {}, AccumT accum = {});
```

Plus convenience aliases: `add`, `sub`, `mul`, `square`.

**Template params (9 total):** Op type, broadcast dim, 2 independent input policies, output policy, reconfig mode, init flag, PostOp type, AccumT type.

**Runtime params:** 3 CB IDs, BinaryInputBlockShape (rows, cols), post_op callback, accumulation config.

**Feature variant handling -- the widest combinatorial space in kernel_lib:**
- **4 op types** (ADD, SUB, MUL, SQUARE) mapped to LLK EltwiseBinaryType
- **4 broadcast dims** (NONE, ROW, COL, SCALAR) mapped to LLK BroadcastType. B-tile indexing depends on broadcast dim. ROW/SCALAR always wait for B upfront.
- **6 input policies per operand**, independently configured for A and B
- **3 output policies** (PerTile, PerChunk, Bulk)
- **Accumulation** via BinaryAccumulate struct (CB + DST index)
- All dispatch via `if constexpr` -- zero-overhead for unused branches

**Init/uninit lifecycle:** `binary_init<op_type, bcast_dim>(icb_a, icb_b)` when `init=true`. No explicit uninit. MUL uses configured MATH_FIDELITY; ADD/SUB always use LoFi.

**CB synchronization:** Most complex in kernel_lib:
- A policy: per-tile, per-chunk, upfront, or caller-managed
- B policy: independent of A. ROW/SCALAR broadcasts force B to wait upfront regardless of policy.
- Output: per-tile, per-chunk, or bulk reserve/push
- B pop timing depends on broadcast dim: NONE pops per tile, COL pops per row, ROW/SCALAR pop at end

**DEST register management:** Chunks by `effective_dest_limit = DEST_AUTO_LIMIT - base_dst`. Per-tile policy: full acquire/commit/wait/release per tile. Per-chunk/bulk: one acquire/release cycle per chunk.

**PostOp:** Called per tile with `dst_idx`. Applied after binary exec, before packing. Enables per-tile SFPU (rsqrt, recip, etc.).

---

### A6. copy_tile_helpers.hpp/inl

**Public API:**
```cpp
template <CopyInputPolicy input_policy = WaitAndPop,
          CopyDataFormatReconfig reconfig_mode = INPUT_AND_OUTPUT,
          typename PostOp = NoOp>
ALWI void copy_tiles(uint32_t input_cb, uint32_t output_cb, uint32_t num_tiles, PostOp post_op = {});
```

**Simplest helper.** Copies tiles one at a time through DST: unpack -> optional post_op -> pack. Two input policies (WaitAndPop, NoWaitNoPop). Processes one tile per acquire/release cycle.

**PostOp:** Called per tile with `dst_idx=0`. Same pattern as binary_op but always on DST[0].

---

### A7. Shared utility files

**cb_helpers.hpp/inl:** `get_full_tile_size<format>()`, `get_cb_num_pages()`, `is_block_float_format()`, `is_valid_cb_tile_page_size()`. Used by tilize, untilize, reduce for validation.

**dest_helpers.hpp:** `get_fp32_dest_acc_enabled()`, `get_dst_full_sync_enabled()`, `get_dest_limit()`, `DEST_AUTO_LIMIT`. Auto-detects from JIT-generated headers. Used by untilize (block splitting), reduce (chunk size), binary_op (chunk size), matmul_block (validation).

**common_types.hpp:** `NoAccumulation`, `NoOp`. Tag types for type-based dispatch. Used by reduce and binary_op.

**l1_helpers.hpp:** `zero_faces()`, `addr_to_l1_ptr()`. Dataflow namespace. Used by reduce_helpers_dataflow.

**reduce_helpers_dataflow.hpp/inl:** `prepare_reduce_scaler()`, `calculate_and_prepare_reduce_scaler()`. Dataflow-side scaler tile generation. Separate namespace (`dataflow_kernel_lib`).

---

### A8. Prior attempt: matmul_block_fused_bias (from wransom/llk3)

**Public API (llk3):**
```cpp
template <uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t interm_cb, uint32_t bias_cb,
          matmul_block_fused_bias_config::InitUninitMode init_uninit_mode = InitAndUninit,
          matmul_block_fused_bias_config::ReconfigureRegisterDatatypeMode reconfig_mode = UnpackAndPackReconfigure,
          bool transpose = false, typename PostComputeFn = NoPostCompute>
ALWI void matmul_block_fused_bias(
    In0BlockParams in0, In1BlockParams in1, uint32_t num_blocks,
    OutSubblockParams out, uint32_t batch = 1, PostComputeFn post_compute = {});
```

**Critical finding: FULL CODE DUPLICATION, not composition.**

The fused bias helper does NOT call `matmul_block()`. It duplicates the entire matmul K-blocking/spill-reload logic (~85 lines of matmul phase) verbatim, then appends a separate bias-add phase (~40 lines). Specific duplicated sections (comparing matmul_block_helpers.inl with matmul_block_fused_bias_helpers.inl):

| Section | matmul_block | fused_bias | Identical? |
|---------|-------------|------------|------------|
| Static asserts | lines 36-41 | lines 37-43 | Yes (+ bias_cb) |
| Derived quantities | lines 44-48 | Used from structs | Different (structs) |
| K-block outer loop | lines 77-168 | lines 85-152 | Nearly identical core logic |
| Spill/reload logic | lines 96-105 | lines 100-109 | Identical |
| Inner matmul_block call | lines 116-128 | lines 113-125 | Identical |
| Last-K-block packing | lines 130-140 | lines 127-133 | Different target (interm vs out) |

**What fused_bias ADDS beyond matmul_block (Phase 2 -- bias addition, lines 155-186):**
1. `reconfig_data_format(in1_cb, interm_cb, in0_cb, bias_cb)` -- 4-arg format switch from matmul to bias-add
2. `pack_reconfig_data_format(out_cb)` -- packer target switch from interm to out
3. `add_bcast_rows_init_short(interm_cb, bias_cb)` -- init bias broadcast op
4. `cb_wait_front(bias_cb, in1.per_core_w)` -- wait for bias tiles (persist across batches)
5. Sub-block loop: `cb_wait_front(interm_cb)` -> `tile_regs_acquire` -> `add_tiles_bcast_rows` row-broadcast loop -> `post_compute()` -> `tile_regs_commit` -> `cb_pop_front(interm_cb)` -> `cb_reserve_back(out_cb)` -> `tile_regs_wait` -> pack -> `tile_regs_release` -> `cb_push_back(out_cb)`
6. `reconfig_data_format(interm_cb, in1_cb, bias_cb, in0_cb)` + `pack_reconfig_data_format(interm_cb)` + `mm_block_init_short()` -- reconfigure back for next batch

**Also used param structs and enums rejected by PR review:**
- `In0BlockParams`, `In1BlockParams`, `OutSubblockParams` -- caller computes derived quantities
- `InitUninitMode`, `ReconfigureRegisterDatatypeMode` -- unused modes with no call sites

**Why this likely caused device hangs:**

The key difference from `matmul_block` is that the fused bias version ALWAYS packs matmul output to `interm_cb` (even the final K-block), because the bias phase needs to read from it. In the regular matmul_block, the final K-block packs directly to `out_cb`, avoiding any contention.

The shared-memory hazard: `out_cb` and `interm_cb` share L1 memory (per the API contract documented in matmul_block_helpers.hpp lines 39-43). The spill path reserves `out_cb` space to prevent `interm_cb` overwrites. But in the fused bias version:
1. During spill: `cb_reserve_back(out_cb, out_num_tiles_to_wait)` holds out_cb space
2. On last K-block: `cb_reserve_back(interm_cb, out_num_tiles)` + `cb_push_back(interm_cb)` for ALL sub-blocks
3. In bias phase: `cb_wait_front(interm_cb)` reads staged matmul output
4. Then `cb_pop_front(interm_cb)` BEFORE `cb_reserve_back(out_cb)` to release shared memory
5. Then `cb_reserve_back(out_cb, out_num_tiles)` + pack + `cb_push_back(out_cb)`

The problem is that on the last K-block, ALL sub-blocks pack to interm_cb before any bias-add runs. With multiple sub-blocks (in0_num_subblocks * in1_num_subblocks > 1), the total interm_cb writes can exceed the CB capacity allocated for a single sub-block's worth of tiles, causing the CB to overflow before the bias phase can drain it. The matmul phase produces (in0_num_subblocks * in1_num_subblocks) sub-block outputs to interm_cb back-to-back, but interm_cb only has `out_num_tiles` pages (one sub-block's worth). The `cb_reserve_back` in the matmul phase will stall waiting for space that will never be freed (because the bias phase hasn't started yet).

---

## Part B -- DeepSeek V3 Unified Kernels

### B1. Architectural Overview

The DeepSeek V3 unified kernels (`models/demos/deepseek_v3_b1/unified_kernels/`) use a fundamentally different architecture from kernel_lib:

**Unified kernel model:** Each operation (Matmul, DRAMStreamingMatmul, etc.) is a single struct containing all three RISC core implementations in one file:
- `ReaderCTArgs` (NCRISC) / `WriterCTArgs` (BRISC) / `ComputeCTArgs` (TRISC) -- compile-time args
- `ReaderArgs` / `WriterArgs` / `ComputeArgs` -- runtime args
- `Op<CTArgs, IsActiveCore, ...>` class template -- the operation itself

Core selection happens via `#if defined(COMPILE_FOR_TRISC)` guards within the `Op::impl()` method and via `SelectByRISCV<Reader, Writer, Compute>` type selection (from `kernel_op_api.hpp`).

**Contrast with kernel_lib:** kernel_lib helpers are compute-only (TRISC). Dataflow is separate. The DeepSeek model co-designs reader/writer/compute in one unit, enabling tighter CB coordination (e.g., NCRISC's triple-buffered DRAM streaming is synchronized with TRISC's subblock-K consumption).

### B2. How Multiple Matmul Variants Are Handled

Six matmul files examined. The variant space is:

| File | Variant dimensions |
|------|-------------------|
| matmul.hpp | out_w, transpose, fused_activation (NONE/SIGMOID/SILU) |
| dram_streaming_matmul.hpp | subblock_w, num_subblocks_k, fuse_silu, fp32_dest_acc_en, tile_r_dim |
| dram_streaming_matmul_compressed.hpp | subblock_k, per_core_n, num_subblocks_k |
| dram_streaming_experts_matmul.hpp | same as dram_streaming_matmul + selected_experts_k |
| kn_sliced_matmul.hpp | out_w |
| (matmul_compressed not found -- listed in orchestration doc but not present) | -- |

**Variant selection mechanism:** All variants are compile-time template parameters on `ComputeCTArgs`. Runtime dispatch uses `if constexpr`:

```cpp
// From matmul.hpp, lines 142-193:
if constexpr (CTArgs::fuse_sigmoid || CTArgs::fuse_silu) {
    // Per-tile: matmul -> activation on PACK -> pack
    for (uint32_t w = 0; w < out_w; w++) { ... }
} else {
    // Batch processing - all tiles at once
    tile_regs_acquire();
    custom_mm_block<finalize, read_transposed>(...);
    ...
}
```

No enums or policy objects -- raw bool/uint32_t template params.

### B3. How Fused Operations Are Plugged In

**Fused SiLU/Sigmoid (matmul.hpp, dram_streaming_matmul.hpp, dram_streaming_experts_matmul.hpp):**

When `fuse_silu=true` or `fuse_sigmoid=true`, the matmul switches from batch processing to per-tile pipelining with SFPU-on-PACK:

1. `tile_regs_acquire()` -- acquire DST for one tile
2. `custom_mm_block<finalize>()` -- compute one output tile
3. `tile_regs_commit()` -- hand DST to PACK thread
4. `TTI_SEMWAIT(STALL_TDMA | STALL_CFG, ...)` -- stall until PACK thread owns DST
5. `PACK(TT_SETC16(..., packer_dest_offset))` -- configure PACK's view of DST
6. `PACK(llk_math_eltwise_unary_sfpu_silu<...>(...))` -- run SFPU on PACK thread
7. `PACK(TTI_STALLWAIT(STALL_PACK, WAIT_SFPU))` -- wait for SFPU completion
8. `pack_tile(0, cb_out, w)` -- pack from DST to output CB
9. `tile_regs_release()` -- release DST

This is a very different pattern from PostComputeFn: the activation runs on the PACK thread, not the MATH thread. It uses low-level TTI instructions and thread semaphores for MATH-PACK synchronization. The kernel_lib PostComputeFn pattern cannot express this because:
- PostComputeFn runs inline (between matmul and pack, on MATH thread)
- The DeepSeek pattern overlaps SFPU with the MATH-to-PACK handoff
- It uses thread-specific macros (PACK(...)) not available in the PostComputeFn context

**Fused eltwise multiply:** Not directly in the matmul ops. Would be composed by chaining Matmul + EltwiseMul ops in the unified kernel pipeline.

### B4. Thread Model Differences

| Aspect | kernel_lib helpers | DeepSeek unified kernels |
|--------|-------------------|--------------------------|
| Scope | Compute only (TRISC) | All 3 RISCs in one file |
| Dataflow | Separate reader/writer kernels | NCRISC reader + BRISC writer in same struct |
| CB coordination | Implicit (wait/pop at helper boundary) | Explicit (NCRISC triple-buffers, TRISC streams) |
| Init | `compute_kernel_hw_startup()` + per-helper init | `deepseek_compute_kernel_hw_startup()` or per-op init |
| SFPU fusion | PostComputeFn (inline, MATH thread) | PACK-thread SFPU with TTI semaphore control |
| Pop control | Helper always pops its inputs | Template bool params (pop_in0, pop_in1) |

### B5. Composition Patterns DeepSeek Uses That kernel_lib Doesn't

1. **Pop-control for input reuse:** `pop_in0=false` allows one activation buffer to be consumed by multiple matmuls (e.g., gate_proj and up_proj share in0). kernel_lib's matmul_block always pops both inputs -- no way to reuse in0 across calls.

2. **CB address override:** `in1_address_override` / `weights_address_override` for pointing the read pointer at an arbitrary L1 address (e.g., sharded tensor). kernel_lib has no equivalent.

3. **DRAM streaming with triple-buffering:** NCRISC triple-buffers in1 from DRAM using transaction IDs (`noc_async_read_set_trid`, `noc_async_read_barrier_with_trid`). The compute side consumes subblock-K at a time. This streaming-and-compute overlap is impossible to express in kernel_lib (compute only).

4. **Expert indexing:** Runtime weight selection via index CB. kernel_lib has no concept of dynamic weight selection.

5. **FP32 accumulation override:** `if constexpr (CTArgs::fp32_dest_acc_en != DST_ACCUM_MODE)` -- can override the JIT-generated accumulation mode per-op. kernel_lib's matmul_block always uses the JIT default.

6. **custom_mm_block with split_acc:** Uses `custom_mm_block<finalize=true/false>()` for K-accumulation without explicit spill/reload. The finalize flag controls whether partial sums are "committed" in DST. This avoids the `copy_tile_to_dst_init_short_with_dt` -> `copy_tile` -> `mm_block_init_short_with_dt` reload dance that kernel_lib must perform.

7. **Compressed matmul:** Entirely different LLK path (`compressed::custom_mm_compressed_block_init_short`, `compressed::custom_mm_compressed_block_runtime`) with per-tile format metadata. No kernel_lib equivalent.

---

## Part C -- Cross-Cutting Patterns

### C1. DEST Register Acquire/Release/Commit Sequences

Two distinct DEST management APIs are used across the codebase:

**Pattern A: tile_regs_acquire/commit/wait/release (4-phase)**
Used by: copy_tile_helpers.inl:44-72, binary_op_helpers.inl:219-357, reduce_helpers_compute.inl:213-248 (REDUCE_SCALAR), reduce_helpers_compute.inl:294-331 (REDUCE_ROW), reduce_helpers_compute.inl:389-437 (REDUCE_COL), all DeepSeek matmul ops.

```
tile_regs_acquire();    // Acquire DST registers for compute
// ... compute into DST ...
tile_regs_commit();     // Hand DST to PACK thread
tile_regs_wait();       // Wait for PACK to be ready
// ... pack_tile() ...
tile_regs_release();    // Release DST registers
```

Enables MATH-PACK pipelining: MATH can start next tile while PACK processes current.

**Pattern B: acquire_dst/release_dst (2-phase)**
Used by: matmul_block_helpers.inl:93-156, matmul_block_fused_bias_helpers.inl (llk3) matmul phase.

```
acquire_dst();
// ... matmul_block LLK + pack_tile ...
release_dst();
```

Simpler: no commit/wait phase. MATH and PACK run sequentially within the acquire/release window.

**Observation:** The matmul helper is the ONLY helper using the 2-phase pattern. All other helpers and all DeepSeek kernels use the 4-phase pattern. This matters because:
- The 4-phase pattern allows SFPU-on-PACK (DeepSeek's fused activation approach)
- The 2-phase pattern blocks MATH while packing
- A redesigned matmul helper should consider switching to 4-phase for better pipeline utilization

### C2. CB Wait/Pop Patterns

Four distinct policies appear across 3+ helpers:

| Policy | Helpers using it | Implementation |
|--------|-----------------|----------------|
| Per-tile streaming | copy_tile, binary_op, reduce | `cb_wait_front(cb, 1)` inside tile loop, `cb_pop_front(cb, 1)` after each tile |
| Bulk wait/pop | reduce (BulkWaitBulkPop), binary_op (WaitAndPopPerChunk) | `cb_wait_front(cb, N)` before block, `cb_pop_front(cb, N)` after block |
| Upfront wait | tilize (WaitUpfront), untilize (WaitUpfront), reduce (WaitUpfrontNoPop), binary_op (WaitUpfrontNoPop) | `cb_wait_front(cb, total)` before all processing |
| Caller-managed | tilize (NoWait), untilize (NoWait), reduce (NoWaitNoPop), binary_op (NoWaitNoPop), copy_tile (NoWaitNoPop) | No wait/pop -- caller does it |

Additionally for matmul_block: waits for full input blocks per K-iteration (block-granularity, not tile or upfront).

### C3. Pack Loops

**Sequential pack (used by 5+ helpers/kernels):**
```cpp
for (uint32_t i = 0; i < num_tiles; i++) {
    pack_tile(i, out_cb);
}
```
Used in: matmul_block_helpers.inl:137-139, matmul_block_fused_bias (llk3):177-179, reduce_helpers_compute.inl:245, copy_tile_helpers.inl:68.

**Offset pack (used by 3+ helpers/kernels):**
```cpp
for (uint32_t i = 0; i < num_tiles; i++) {
    pack_tile(dst_idx + i, out_cb, offset + i);
}
```
Used in: binary_op_helpers.inl:325-332, DeepSeek matmul.hpp:189-191 (`pack_tile(dst_idx, args.out, dst_idx)`), DeepSeek dram_streaming_matmul.hpp:358-360.

**Out-of-order pack:** `pack_tile<true>(i, out_cb)` -- not used in kernel_lib. Used in some experimental kernels (e.g., minimal_matmul `pack_tile<true>`).

### C4. Data Format Reconfiguration Sequences

**Unpack reconfig (3+ helpers):**
```cpp
reconfig_data_format_srca(input_cb);  // Reconfigure unpacker source A
```
Used in: matmul_block_helpers.inl:68-70, tilize_helpers.inl:136-142, untilize_helpers.inl:107-109, copy_tile_helpers.inl:33.

**Combined reconfig (3+ helpers):**
```cpp
reconfig_data_format(input_cb, scaler_cb);  // Reconfigure both srcA and srcB
```
Used in: reduce_helpers_compute.inl:169, binary_op_helpers.inl:160.

**Pack reconfig (5+ helpers):**
```cpp
pack_reconfig_data_format(output_cb);
```
Used in: matmul_block_helpers.inl:70, tilize_helpers.inl:147, untilize_helpers.inl:113, copy_tile_helpers.inl:38, reduce_helpers_compute.inl:172, binary_op_helpers.inl:163.

**Phase-switch reconfig (fused_bias only):**
```cpp
reconfig_data_format(in1_cb, interm_cb, in0_cb, bias_cb);  // 4-arg: switch both sources
pack_reconfig_data_format(out_cb);
add_bcast_rows_init_short(interm_cb, bias_cb);
```
Used only in matmul_block_fused_bias (llk3):155-158. This is the pattern needed for switching between matmul and bias-add operations. No current kernel_lib helper supports multi-phase reconfiguration.

**DeepSeek reconfig:**
```cpp
reconfig_data_format<false, true>(cb_b, cb_a);  // Template version
pack_reconfig_data_format<true>(cb_out);
```
Used in all DeepSeek matmul ops. The template bools control which registers to reconfigure.

### C5. Spill/Reload Sequences (K-Blocking)

**Pattern A: Explicit spill/reload via interm_cb (kernel_lib)**
Used in: matmul_block_helpers.inl:96-105 (reload), matmul_block_helpers.inl:148-153 (spill). Also duplicated in matmul_block_fused_bias (llk3).

Reload sequence:
```cpp
copy_tile_to_dst_init_short_with_dt(in1_cb, interm_cb);    // Switch unpacker to interm format
cb_wait_front(interm_cb, out_num_tiles);                     // Wait for spilled partial results
for (i = 0..out_num_tiles) copy_tile(interm_cb, i, i);     // Load partials into DST
cb_pop_front(interm_cb, out_num_tiles);                      // Release interm CB space
mm_block_init_short_with_dt(in0_cb, in1_cb, interm_cb, ...); // Restore matmul unpacker config
```

Spill sequence:
```cpp
cb_reserve_back(interm_cb, out_num_tiles);
for (i = 0..out_num_tiles) pack_tile(i, interm_cb);
cb_push_back(interm_cb, out_num_tiles);
```

Reduce uses a similar reload pattern (reduce_helpers_compute.inl:68-84, `reload_accumulator_if_needed`):
```cpp
cb_wait_front(accum_cb, 1);
copy_tile_to_dst_init_short_with_dt(input_cb, accum_cb);
copy_tile(accum_cb, 0, dst_index);
cb_pop_front(accum_cb, 1);
reduce_init_short_with_dt<...>(accum_cb, input_cb, scaler_cb);  // Restore reduce config
```

**Pattern B: split_acc accumulation (DeepSeek)**
Used in: all DeepSeek matmul ops (dram_streaming_matmul.hpp:262-366, etc.).

```cpp
custom_mm_block_init_short<transpose, split_acc=true, dense_packing>(...);
for (sb_k = 0; sb_k < num_subblocks_k - 1; sb_k++) {
    custom_mm_block<finalize=false>(...);  // Partial accumulation in DST
}
custom_mm_block<finalize=true>(...);       // Final accumulation
```

No explicit spill/reload -- DST accumulates across K subblocks via the `finalize` flag. This is fundamentally simpler but requires the `custom_mm_block` LLK (not available in kernel_lib which uses `mm_block_init` + `ckernel::matmul_block`).

### C6. Per-Helper Init/Uninit Lifecycle Summary

| Helper | Init | Uninit | Lifecycle control? |
|--------|------|--------|--------------------|
| matmul_block | `mm_block_init()` | none | No |
| tilize | `tilize_init()` / `fast_tilize_init()` | `tilize_uninit()` / `fast_tilize_uninit()` | Yes -- InitUninitMode enum (4 modes) |
| untilize | `pack_untilize_init<>()` | `pack_untilize_uninit()` | Yes -- InitUninitMode enum (4 modes) |
| reduce | `reduce_init<>()` | `reduce_uninit<>()` | No (always init+uninit) |
| binary_op | `binary_init<>()` | none | Partial -- `init` bool template param |
| copy_tile | `copy_tile_to_dst_init_short()` | none | No |
| fused_bias (llk3) | `mm_block_init()` | none | Yes -- InitUninitMode enum |

### C7. PostComputeFn / Callback Pattern Comparison

| Helper | Callback name | Receives | Called when | Can access CB data? |
|--------|--------------|----------|------------|---------------------|
| matmul_block | PostComputeFn | `out_subblock_num_tiles` | Per sub-block, last K-block, before pack | No |
| fused_bias (llk3) | PostComputeFn | `out_subblock_num_tiles` | Per sub-block, after bias add, before pack | No |
| binary_op | PostOp | `dst_idx` | Per tile, after binary exec, before pack | No |
| reduce | PostReduceOp | `dst_idx` | Per output tile (ROW/COL), before pack | No |
| copy_tile | PostOp | `dst_idx=0` | Per tile, after copy, before pack | No |

**What the callback pattern enables:**
- Zero-overhead SFPU ops on DST tiles before packing
- Composing unary ops (relu, gelu, exp, recip, rsqrt) with compute ops
- Default `NoOp`/`NoPostCompute` compiles away entirely

**What it CANNOT express:**
1. Multi-input operations (bias add needs a separate CB -- bias_cb)
2. Data format reconfiguration (callback runs in the compute op's format context)
3. Different packing targets (callback cannot redirect output to a different CB)
4. Multi-tile coordination (matmul_block's PostComputeFn gets tile count but no per-tile control)
5. Operations that need their own init/uninit (e.g., add_bcast_rows_init_short)
6. Cross-thread operations (PACK-thread SFPU as in DeepSeek)

### C8. Shared API Pattern Summary

Patterns consistent across all kernel_lib helpers:

1. **Namespace:** `compute_kernel_lib` for compute helpers, `dataflow_kernel_lib` for dataflow helpers
2. **Config sub-namespace:** `tilize_config`, `untilize_config`, `matmul_block_config`, `matmul_block_fused_bias_config`
3. **File structure:** `.hpp` has types + function declaration + `#include "*.inl"`; `.inl` has implementation
4. **Template params:** CB indices are compile-time; dimensions are runtime
5. **Compile-time dispatch:** `if constexpr` on enum values and type traits
6. **Runtime asserts:** `ASSERT()` for dimension > 0, DEST limits, CB capacity
7. **TRISC-guarded asserts:** `PACK(ASSERT(...))` or `UNPACK(ASSERT(...))` for thread-specific checks
8. **Static asserts:** CB index validity (< 32), CB distinctness (in != out)
9. **ALWI attribute:** All helper functions are `ALWI` (Always Inline)
10. **Data format reconfig at function start:** Most helpers reconfigure unpack and pack data formats as their first action
