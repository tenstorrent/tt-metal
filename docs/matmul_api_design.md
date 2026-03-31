# Matmul Helper API Design

Phase 2, Instance 1 output. Synthesizes all phase 1 analysis into a concrete API design.

---

## Section A -- Design Principles

### P1: L1 accumulation is a compile-time strategy, not a runtime option

Phase 1 analysis (Instance 2, Paths 3-4; Instance 4, kernel #1) shows PACKER_L1_ACC fundamentally changes the K-blocking loop's spill/reload behavior:

- **Without L1_ACC**: software spill to interm CB on every non-last K-block, reload on every block after the first. `enable_reload = true` after block 0.
- **With L1_ACC, no bias**: hardware accumulates in L1, no explicit reload. Only reload on the K-1 block to pack to output. `cb_wait_front/cb_pop_front` between blocks to advance FIFO pointer.
- **With L1_ACC + bias**: hardware accumulates across ALL K-blocks, `enable_reload` is always false. Last block's output stays in interm for the bias phase.

These are three different code paths, not runtime variations. The `packer_l1_acc` flag must be a compile-time template parameter so the irrelevant paths compile away via `if constexpr`.

### P2: Post-matmul phases are separate helpers, not matmul parameters

The llk3 `matmul_block_fused_bias` attempt failed because it combined matmul K-blocking and bias addition in one function. Instance 3's analysis identified the root cause: the monolithic function packed ALL subblocks to interm before the bias phase drained any, causing CB overflow with multi-subblock configurations.

The production kernel avoids this by running bias as a distinct phase AFTER the complete K-loop. Design accordingly:
- `matmul_block` handles the K-blocking loop (subblock iteration, spill/reload or L1_ACC, pack)
- `add_bias_bcast_rows` handles the post-matmul bias-add phase (read from interm, add bias, pack to output)
- The caller composes them: `matmul_block(pack to interm) --> add_bias_bcast_rows(read from interm)`

### P3: Use 4-phase DST management, matching production code and all other helpers

Instance 3 (Part C, Section C1) found that the matmul helper is the ONLY kernel_lib helper using 2-phase DST (`acquire_dst/release_dst`). All other helpers (binary_op, reduce, copy_tile, tilize, untilize) and the production kernel use 4-phase DST (`tile_regs_acquire/commit/wait/release`).

4-phase DST allows MATH to start the next subblock while PACK finishes the previous one, and is required for correct pipelining with `cb_reserve_back` between commit and wait (the production kernel's pattern). Switching to 4-phase is a performance improvement and aligns with the codebase convention.

### P4: Caller-managed features stay outside the helper

Several production kernel features are caller-level concerns that should NOT be absorbed into the matmul helper:

| Feature | Why caller-managed | Reference |
|---------|-------------------|-----------|
| `in0_transpose_tile` | Single call site (production kernel only, not conv/gathered). ~20 lines of self-contained transpose + format reconfig + L1_ACC toggle. Absorbing it would add template params with only 1 use. | Instance 1, only kernel #3 uses this |
| `SKIP_COMPUTE` | Degenerate case (bias-only). Caller skips matmul helper entirely. | Instance 2, Path 8 |
| `untilize_out` | Separate post-processing phase. Untilize helper already exists. | Instance 2, Path 10 |
| `MATMUL_DRAM_SHARDED` | Core-type early exit. Checked before any compute. | Instance 2, Section A |
| Outer H/W loops | Loop structure with per-iteration reconfig. Caller composes. | Instance 2, Section B |
| `get_batch_from_reader` | Per-batch validity mailbox check. Orthogonal to matmul. | Instance 2, Section A |

### P5: Always reconfigure pack format on the last K-block

Instance 2 (Paths 3-7) shows that `pack_reconfig_data_format(mm_out_cb)` is called on the last K-block whenever `FP32_DEST_ACC_EN` or `PACKER_L1_ACC` is active. Since `get_fp32_dest_acc_enabled()` (from `dest_helpers.hpp`) auto-detects FP32 mode from JIT headers, and `packer_l1_acc` is a template param, the helper can determine internally whether reconfig is needed. The caller doesn't need a separate `FP32_DEST_ACC_EN` flag.

---

## Section B -- Type System

### Existing types (unchanged)

```cpp
// common_types.hpp (stable, colleague-owned)
namespace compute_kernel_lib {
    struct NoAccumulation {};   // Tag type: no accumulation
    struct NoOp {               // Default no-op functor
        ALWI void operator()(uint32_t = 0) const {}
    };
}
```

### Modified type: matmul_block_config namespace

```cpp
// matmul_block_helpers.hpp
namespace compute_kernel_lib {
namespace matmul_block_config {

// Default no-op post-compute functor (UNCHANGED from current).
// Called per output sub-block on the last K-block, before packing.
// Receives out_subblock_num_tiles. Tiles are in DST[0..num_tiles-1].
struct NoPostCompute {
    ALWI void operator()(uint32_t /* out_subblock_num_tiles */) const {}
};

}  // namespace matmul_block_config
}  // namespace compute_kernel_lib
```

No changes to `matmul_block_config`. The `NoPostCompute` type stays as-is.

### New type: bias_add_config namespace

```cpp
// bias_add_helpers.hpp (NEW FILE)
namespace compute_kernel_lib {
namespace bias_add_config {

// Default no-op post-bias functor.
// Called per output sub-block after bias addition, before packing.
// Receives out_subblock_num_tiles. Tiles are in DST[0..num_tiles-1].
// Use for fused SFPU activation after bias (e.g., gelu, silu).
struct NoPostBias {
    ALWI void operator()(uint32_t /* out_subblock_num_tiles */) const {}
};

}  // namespace bias_add_config
}  // namespace compute_kernel_lib
```

### What changed from current types and why

**No param structs.** The llk3 attempt used `In0BlockParams`, `In1BlockParams`, `OutSubblockParams`. These were rejected by PR review: "take only independent params, derive computed values internally." The redesign follows the current `matmul_block` approach: flat independent runtime params, derived quantities computed in the implementation.

**No unused enums.** The llk3 attempt used `InitUninitMode` and `ReconfigureRegisterDatatypeMode` with no call sites using non-default values. The redesign adds only template params with real call sites (see Section C).

**Separate PostBias functor.** The production kernel applies SFPU activation at different points depending on whether bias is fused: after matmul (no bias) or after bias add (with bias). The matmul helper's `PostComputeFn` fires after matmul. The bias helper's `PostBiasFn` fires after bias add. The caller decides which to use.

---

## Section C -- Function Signatures

### C1. matmul_block (MODIFIED -- extended with new template params)

```cpp
// matmul_block_helpers.hpp
namespace compute_kernel_lib {

/**
 * matmul_block: sub-blocked tiled matrix multiplication C = A x B with K-blocking.
 *
 * Performs matrix multiplication using mm_block_init + matmul_block LLK with
 * sub-block indexing and automatic K-dimension blocking. Supports two blocking
 * strategies selected at compile time:
 *
 *   packer_l1_acc=false: Software spill/reload via interm_cb (current behavior)
 *   packer_l1_acc=true:  Hardware L1 accumulation via packer (avoids spill/reload)
 *
 * -- Template Parameters --
 *
 *   in0_cb            Input CB for matrix A (0-31).
 *   in1_cb            Input CB for matrix B (0-31).
 *   out_cb            Output CB for result C (0-31). Also used for shared memory
 *                     protection with interm_cb (they overlap in L1).
 *   interm_cb         Intermediate CB for K-blocking (0-31). Used for spill/reload
 *                     (software) or L1 accumulation (hardware). Must differ from out_cb.
 *   transpose         If true, transpose B tiles before multiplication (default: false).
 *   packer_l1_acc     If true, use packer L1 accumulation instead of software
 *                     spill/reload. Changes K-blocking strategy fundamentally:
 *                     - Block 0: llk_pack_reconfig_l1_acc(0) (no accumulation)
 *                     - Block 1+: llk_pack_reconfig_l1_acc(1) (accumulate)
 *                     - Between blocks: cb_wait_front/cb_pop_front to advance FIFO
 *                     - Last block: reload only when !pack_last_to_interm
 *                     (default: false)
 *   pack_last_to_interm  If true, the last K-block packs to interm_cb instead of
 *                     out_cb. Use when a post-processing phase (bias add, untilize)
 *                     reads from interm_cb. When combined with packer_l1_acc,
 *                     enable_reload is always false (L1 accumulates across all blocks).
 *                     (default: false)
 *   pack_relu         If true, enable PACK_RELU on the last K-block when
 *                     !pack_last_to_interm. Calls llk_pack_relu_config(ZERO_RELU)
 *                     before the last block's pack phase. Has no effect when
 *                     pack_last_to_interm=true (RELU is deferred to the post-
 *                     processing phase in that case). (default: false)
 *   PostComputeFn     Functor called per output sub-block on the last K-block,
 *                     after matmul but before packing. For fused SFPU activations
 *                     when no bias follows. (default: NoPostCompute)
 *
 * -- Runtime Parameters --
 *
 *   block_w           Inner block dimension in tiles (K-dimension block size).
 *   in0_num_subblocks Number of sub-blocks along the M dimension.
 *   in1_num_subblocks Number of sub-blocks along the N dimension.
 *   num_k_blocks      Number of blocks along the K dimension.
 *   out_subblock_h    Output sub-block height in tiles.
 *   out_subblock_w    Output sub-block width in tiles.
 *   batch             Number of independent batch slices (default: 1).
 *   post_compute      PostComputeFn instance (default: {}).
 */
template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t interm_cb,
    bool transpose = false,
    bool packer_l1_acc = false,
    bool pack_last_to_interm = false,
    bool pack_relu = false,
    typename PostComputeFn = matmul_block_config::NoPostCompute>
ALWI void matmul_block(
    uint32_t block_w,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t num_k_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t batch = 1,
    PostComputeFn post_compute = {});

}  // namespace compute_kernel_lib
```

**How the new template params interact:**

| packer_l1_acc | pack_last_to_interm | Spill strategy | Last block target | enable_reload |
|:---:|:---:|---|---|---|
| false | false | Software spill/reload | out_cb | true after block 0 |
| false | true | Software spill/reload | interm_cb | true after block 0 |
| true | false | L1 accumulation | out_cb (reload on K-1) | true only on K-1 |
| true | true | L1 accumulation | interm_cb (no reload) | always false |

| pack_relu | pack_last_to_interm | Effect |
|:---:|:---:|---|
| false | any | No RELU |
| true | false | RELU enabled on last K-block's pack to out_cb |
| true | true | No RELU (deferred to post-processing) |

### C2. add_bias_bcast_rows (NEW)

```cpp
// bias_add_helpers.hpp (NEW FILE)
namespace compute_kernel_lib {

/**
 * add_bias_bcast_rows: row-broadcast bias addition on matmul output.
 *
 * Reads matmul output sub-blocks from partials_cb, adds bias with row broadcast,
 * optionally applies a post-bias operation (e.g., SFPU activation), and packs
 * the result to out_cb.
 *
 * This is the bias-add phase of the production matmul kernel
 * (bmm_large_block_zm_fused_bias_activation.cpp lines 404-462). It composes
 * with matmul_block by reading from the same interm_cb that matmul_block
 * packed to (when pack_last_to_interm=true).
 *
 * CB flow: partials_cb (wait+pop per subblock) + bias_cb (wait upfront, no pop)
 *          --> out_cb (reserve+push per subblock)
 *
 * -- Template Parameters --
 *
 *   partials_cb    CB containing matmul output (= interm_cb from matmul_block).
 *   bias_cb        CB containing bias tiles. One tile per output column, row-broadcast.
 *   out_cb         Output CB for biased result.
 *   PostBiasFn     Functor called per output sub-block after bias addition, before
 *                  packing. For fused SFPU activation (e.g., gelu, silu). Receives
 *                  out_subblock_num_tiles. Tiles are in DST[0..num_tiles-1].
 *                  (default: NoPostBias)
 *
 * -- Runtime Parameters --
 *
 *   in0_num_subblocks  Number of sub-blocks along M dimension (from matmul config).
 *   in1_num_subblocks  Number of sub-blocks along N dimension (from matmul config).
 *   out_subblock_h     Output sub-block height in tiles.
 *   out_subblock_w     Output sub-block width in tiles.
 *   bias_width_tiles   Number of bias tiles to wait for (= in1_num_subblocks * out_subblock_w).
 *
 * -- Notes --
 *
 *   - Caller must wait for bias_cb tiles before calling, OR the helper waits internally.
 *     The helper calls cb_wait_front(bias_cb, bias_width_tiles) which is a no-op if
 *     tiles are already available.
 *   - The helper does NOT pop bias_cb. The caller manages bias tile lifetime
 *     (bias may persist across multiple W-block iterations).
 *   - The helper performs data format reconfiguration at the start:
 *       reconfig_data_format_srca(partials_cb)
 *       reconfig_data_format_srcb(bias_cb)
 *       pack_reconfig_data_format(out_cb)
 *     If reconfiguration was already done by the caller, this is a no-op
 *     (reconfiguring to the same format is safe).
 *   - Uses 4-phase DST management (tile_regs_acquire/commit/wait/release).
 *   - PostBiasFn placement: without PostBiasFn, tile_regs_commit() is called
 *     immediately after bias add. With PostBiasFn, the functor runs first, then
 *     tile_regs_commit(). This matches the production kernel's SFPU placement
 *     (Instance 2, Path 7: commit AFTER activation when SFPU is fused with bias).
 */
template <
    uint32_t partials_cb,
    uint32_t bias_cb,
    uint32_t out_cb,
    typename PostBiasFn = bias_add_config::NoPostBias>
ALWI void add_bias_bcast_rows(
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t bias_width_tiles,
    PostBiasFn post_bias = {});

}  // namespace compute_kernel_lib
```

### C3. Composition diagram

```
Caller (kernel_main):
  mm_block_init(in0, in1, interm, transpose, osw, osh, bw)
  |
  for batch / H / W:
  |  [caller: PACK_RELU disable, format reconfig if multi-iteration]
  |  [caller: optional in0_transpose phase]
  |
  |  matmul_block<in0, in1, out, interm, ...>()
  |    |-- K-loop (internal)
  |    |   |-- subblock iteration
  |    |   |-- spill/reload OR L1 accumulation
  |    |   |-- PostComputeFn on last K-block (SFPU if !bias)
  |    |   |-- pack to out_cb or interm_cb
  |    |-- returns
  |
  |  [caller: optional bias phase]
  |  add_bias_bcast_rows<interm, bias, out>()
  |    |-- format reconfig + init
  |    |-- subblock loop
  |    |   |-- add_tiles_bcast_rows (row broadcast)
  |    |   |-- PostBiasFn (SFPU if bias)
  |    |   |-- pack to out_cb
  |    |-- returns (does NOT pop bias)
  |
  |  [caller: optional untilize phase]
  |  [caller: format reconfig for next iteration]
```

### C4. File structure

Following the established kernel_lib pattern (`.hpp` has types + declaration + `#include "*.inl"`; `.inl` has implementation):

| File | Purpose | Status |
|------|---------|--------|
| `matmul_block_helpers.hpp` | matmul_block types + declaration | **Modified** (add template params) |
| `matmul_block_helpers.inl` | matmul_block implementation | **Modified** (rewrite for L1_ACC, 4-phase DST) |
| `bias_add_helpers.hpp` | add_bias_bcast_rows types + declaration | **New** |
| `bias_add_helpers.inl` | add_bias_bcast_rows implementation | **New** |
| `common_types.hpp` | NoAccumulation, NoOp | **Unchanged** |
| `dest_helpers.hpp` | DEST_AUTO_LIMIT, get_fp32_dest_acc_enabled | **Unchanged** |

---

## Section D -- Coverage Matrix

| # | Kernel | Helper expression | Notes |
|---|--------|------------------|-------|
| 1 | `bmm.cpp` (production) | (inline -- no helper) | Tile-by-tile, 1 call site. Per PR feedback. |
| 2 | `bmm_large_block_zm.cpp` (production) | `matmul_block<in0,in1,out,interm>()` | **Already migrated. Backward compatible.** |
| 3a | `bmm_fused_bias_activation.cpp` -- base path (no defines, no bias) | `matmul_block<in0,in1,out,interm>()` | Same as #2. |
| 3b | `bmm_fused_bias_activation.cpp` -- PACKER_L1_ACC, no bias | `matmul_block<in0,in1,out,interm, xpose, /*l1_acc=*/true>()` | L1 accumulation. |
| 3c | `bmm_fused_bias_activation.cpp` -- FUSE_BIAS, no L1_ACC | `matmul_block<in0,in1,out,interm, xpose, false, /*to_interm=*/true>()` then `add_bias_bcast_rows<interm,bias,out_target>()` | Two-phase compose. |
| 3d | `bmm_fused_bias_activation.cpp` -- FUSE_BIAS + PACKER_L1_ACC | `matmul_block<in0,in1,out,interm, xpose, /*l1_acc=*/true, /*to_interm=*/true>()` then `add_bias_bcast_rows<interm,bias,out_target>()` | Full compose. |
| 3e | `bmm_fused_bias_activation.cpp` -- PACK_RELU, no bias | `matmul_block<..., /*pack_relu=*/true>()` | RELU on last K-block. |
| 3f | `bmm_fused_bias_activation.cpp` -- SFPU, no bias | `matmul_block<..., PostSFPU>()` | SFPU via PostComputeFn. |
| 3g | `bmm_fused_bias_activation.cpp` -- SFPU + FUSE_BIAS | `matmul_block<..., to_interm=true>()` then `add_bias_bcast_rows<..., PostSFPU>()` | SFPU via PostBiasFn. |
| 3h | `bmm_fused_bias_activation.cpp` -- PACK_RELU + FUSE_BIAS | `matmul_block<..., to_interm=true>()` then caller sets RELU, then `add_bias_bcast_rows<>()` | RELU toggled by caller between phases. |
| 3i | `bmm_fused_bias_activation.cpp` -- in0_transpose paths | Inline K-loop + `add_bias_bcast_rows<>()` for bias | in0_transpose is caller-managed (see P4). |
| 3j | `bmm_fused_bias_activation.cpp` -- untilize_out | `matmul_block<..., to_interm=true>()` [+ bias] then inline untilize | Untilize phase is caller-managed. |
| 3k | `bmm_fused_bias_activation.cpp` -- SKIP_COMPUTE | Caller skips matmul_block, runs bias only | Degenerate path. |
| 4 | `bmm_fused_bias_activation_gathered.cpp` (production) | Custom kernel calling `matmul_block<in0,in1, mm_out_cb, mm_partials_cb, xpose, l1_acc>()` within ring loop | Gathered infra is caller-managed. Matmul core from helper. |
| 5 | `conv_bmm_tilize.cpp` | `tilize() --> matmul_block<..., l1_acc, to_interm>() --> add_bias_bcast_rows<>() --> untilize()` | Three-helper compose. |
| 6 | `matmul_block.cpp` (test) | `matmul_block<in0,in1,out,interm>()` | Simple test kernel. |
| 7 | `bmm_large_block_zm.cpp` (test) | `matmul_block<in0,in1,out,interm>()` | Simple test kernel. |
| 8 | `bmm_large_block_zm_fused_bias_activation.cpp` (test) | `matmul_block<..., to_interm>() + add_bias_bcast_rows<..., PostSFPU>()` | Test kernel for bias+SFPU. |
| 9 | `bmm_large_block_zm_mixed_precision.cpp` (test) | `matmul_block<in0,in1,out,interm>()` | Mixed precision via format reconfig (caller-managed). |
| 10 | `bmm_tilize_untilize.cpp` (test) | `tilize() --> matmul_block<..., to_interm>() + add_bias_bcast_rows<..., PostSFPU>() --> untilize()` | Full pipeline test. |

### Kernels NOT covered by helpers (and why)

| Kernel | Reason |
|--------|--------|
| `compute_streaming.hpp` (SDPA) | Matmul is 55 lines in a 1528-line kernel. Custom `blocked_matmul_and_pack` with out-of-order packing, HW semaphores, architecture-specific LLK. (Instance 4, #4) |
| `moreh_matmul.cpp` | Tile-by-tile `matmul_tiles` API (not `matmul_block`). Fused eltwise multiply. 11 `#ifdef` branches. Too specialized. |
| `moe_gate_mm`, `moe_compute`, `topk_router` | Matmul is 15-30 lines in 300-400 line kernels. Complex MoE/topk post-processing. (Instance 4, #8-#10) |
| `matmul_wo` (DeepSeek MLA) | Cross-block DST accumulation without spill/reload (fundamentally different). 105 lines total. |
| `group_attn_matmul` | Uses `matmul_tiles` API (removed per PR feedback). Only 1 call site. |
| DeepSeek V3 unified kernels | Use `custom_mm_block` API (different LLK path). Unified BRISC/NCRISC/TRISC model. |

---

## Section E -- Migration Sequence

### 1. Verify backward compatibility: bmm_large_block_zm.cpp

**Rationale**: This kernel is already migrated. Adding template params with defaults must not break it.

**Validation**: Build + run existing matmul C++ integration tests + Python tests (588 tests). Pass/fail count must match pre-change baseline.

### 2. Production kernel: bmm_large_block_zm_fused_bias_activation.cpp

**Rationale**: PRIMARY TARGET. 500 LOC, 22 `#ifdef` branches, uses 13/14 tracked features. Validates the full design against the most complex production kernel. Covers PACKER_L1_ACC, FUSE_BIAS, SFPU, PACK_RELU, FP32_DEST_ACC_EN, untilize_out paths.

**Approach**: Migrate all `#ifdef` paths except `in0_transpose_tile` (kept inline) and `SKIP_COMPUTE` (caller-managed). The migrated kernel will have `#ifdef` guards that select template params for the helper calls.

**Validation**: Run all matmul Python tests (588) + C++ integration tests. Check PCC on both Wormhole and Blackhole.

### 3. Test kernel: bmm_large_block_zm_fused_bias_activation.cpp (test)

**Rationale**: 163 LOC test kernel with bias + SFPU + mixed precision. Validates helpers in an isolated test context.

### 4. Conv kernel: conv_bmm_tilize.cpp

**Rationale**: 640 LOC, validates three-helper composition (tilize --> matmul --> bias --> untilize). Shares most features with the production kernel.

**Approach**: The matmul core uses the same helper. Tilize (pre) and untilize (post) use existing helpers. The caller manages the complex CB pointer operations and activation reuse logic.

### 5. Gathered variant: bmm_large_block_zm_fused_bias_activation_gathered.cpp

**Rationale**: Validates composability -- the gathered kernel calls the matmul helper within its ring loop with dynamic per-batch CBs.

**Approach**: The gathered kernel file stays separate (substantially different infrastructure). The matmul core calls the helper. Per-batch output CBs, ring sync, and CB pointer management stay caller-managed.

---

## Section F -- Code Sketches

### F1. Production kernel after migration (primary target)

```cpp
// bmm_large_block_zm_fused_bias_activation.cpp (after migration)
// Estimated: ~250 LOC (down from 500)

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#ifdef FUSE_BIAS
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#endif
#include "api/compute/pack_untilize.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

// transpose_tile_block stays as-is (single call site, not worth a helper)
template <uint32_t in0_block_num_tiles, uint32_t block_size = 4>
FORCE_INLINE void transpose_tile_block(uint32_t in0_transpose_cb_id, uint32_t in0_cb_id) {
    // ... unchanged from current production kernel (lines 39-79) ...
}

// SFPU PostComputeFn for !FUSE_BIAS paths
#ifdef SFPU_OP_INIT_ACTIVATION
struct PostMatmulSFPU {
    ALWI void operator()(uint32_t out_subblock_num_tiles) const {
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            SFPU_OP_FUNC_ACTIVATION
        }
    }
};
// SFPU PostBiasFn for FUSE_BIAS paths
struct PostBiasSFPU {
    ALWI void operator()(uint32_t out_subblock_num_tiles) const {
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            SFPU_OP_FUNC_ACTIVATION
        }
    }
};
#endif

void kernel_main() {
    // ── Parse compile-time args (unchanged from current) ──
    // ... lines 148-178 equivalent ...
    constexpr uint32_t in0_cb_id = /* ... */;
    constexpr uint32_t in1_cb_id = /* ... */;
    constexpr uint32_t out_cb_id = /* ... */;
    constexpr uint32_t mm_partials_cb_id = /* ... */;
    // ... etc ...

#ifdef MATMUL_DRAM_SHARDED
    if (/* not worker core */) return;
#endif

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

#ifdef IN1_TRANSPOSE_TILE
    constexpr bool in1_transpose_tile = true;
#else
    constexpr bool in1_transpose_tile = false;
#endif

    // Determine pack target for matmul output
#ifdef FUSE_BIAS
    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;
#else
    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;
#endif

    constexpr bool l1_acc =
#ifdef PACKER_L1_ACC
        true;
#else
        false;
#endif

    constexpr bool do_relu =
#if defined(PACK_RELU) && !defined(FUSE_BIAS)
        true;
#else
        false;
#endif

    mm_block_init(in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile,
                  out_subblock_w, out_subblock_h, in0_block_w);

    for (uint32_t b = 0; b < batch; b++) {
        // [get_batch_from_reader: mailbox check]
        for (uint32_t bh = 0; bh < num_blocks_h_dim; bh++) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; bw++) {

                // ── Pre-matmul setup ──
                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
#ifdef PACK_RELU
                    PACK((llk_pack_relu_config(ReluType::NO_RELU)));
#endif
                    PACK((pack_reconfig_data_format(mm_partials_cb_id)));
                }

                // ── in0 transpose (stays inline -- single call site) ──
                // NOTE: When in0_transpose_tile is true, the K-loop must be
                // inline because the helper can't interleave transpose phases.
                // This path is omitted from the sketch for clarity.
                // See Section H, Open Question 1.

                // ── Matmul K-loop via helper ──
#ifdef FUSE_BIAS
                // Pack last K-block to interm for bias phase
                compute_kernel_lib::matmul_block<
                    in0_cb_id, in1_cb_id, out_cb_id, mm_partials_cb_id,
                    in1_transpose_tile, l1_acc, /*pack_last_to_interm=*/true>(
                    in0_block_w, in0_num_subblocks, in1_num_subblocks,
                    num_blocks_inner_dim, out_subblock_h, out_subblock_w, 1);
#else
                // Pack last K-block directly to output
                compute_kernel_lib::matmul_block<
                    in0_cb_id, in1_cb_id, untilize_mode_out_cb_id, mm_partials_cb_id,
                    in1_transpose_tile, l1_acc, /*pack_last_to_interm=*/false, do_relu,
  #ifdef SFPU_OP_INIT_ACTIVATION
                    PostMatmulSFPU
  #endif
                    >(
                    in0_block_w, in0_num_subblocks, in1_num_subblocks,
                    num_blocks_inner_dim, out_subblock_h, out_subblock_w, 1
  #ifdef SFPU_OP_INIT_ACTIVATION
                    , PostMatmulSFPU{}
  #endif
                    );
#endif

                // ── Bias phase (via helper) ──
#ifdef FUSE_BIAS
  #ifdef PACK_RELU
                PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
  #endif
  #if defined(FP32_DEST_ACC_EN) || defined(PACKER_L1_ACC)
                PACK((pack_reconfig_data_format(untilize_mode_out_cb_id)));
  #endif
  #ifdef PACKER_L1_ACC
                PACK((llk_pack_reconfig_l1_acc(0)));
  #endif

                compute_kernel_lib::add_bias_bcast_rows<
                    mm_partials_cb_id, bias_cb_id, untilize_mode_out_cb_id
  #ifdef SFPU_OP_INIT_ACTIVATION
                    , PostBiasSFPU
  #endif
                    >(
                    in0_num_subblocks, in1_num_subblocks,
                    out_subblock_h, out_subblock_w, in1_block_w
  #ifdef SFPU_OP_INIT_ACTIVATION
                    , PostBiasSFPU{}
  #endif
                    );

                if constexpr (num_blocks_w_dim > 1) {
                    cb_pop_front(bias_cb_id, in1_block_w);
                }
#endif  // FUSE_BIAS

                // ── Untilize phase (stays inline) ──
                if constexpr (untilize_out) {
#ifdef PACK_RELU
                    PACK((llk_pack_relu_config(ReluType::NO_RELU)));
#endif
#ifndef FUSE_BIAS
                    reconfig_data_format_srca(in1_cb_id, mm_partials_cb_id);
  #if defined(FP32_DEST_ACC_EN) || defined(PACKER_L1_ACC)
                    PACK((pack_reconfig_data_format(out_cb_id)));
  #endif
  #ifdef PACKER_L1_ACC
                    PACK((llk_pack_reconfig_l1_acc(0)));
  #endif
#endif
                    pack_untilize_dest_init<out_subblock_w, out_block_w>(out_cb_id);
                    copy_tile_to_dst_init_short(mm_partials_cb_id);
                    for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                        reblock_and_untilize<out_subblock_w, out_block_w>(
                            in1_num_subblocks, out_subblock_num_tiles,
                            out_subblock_h, mm_partials_cb_id, out_cb_id);
                    }
                    pack_untilize_uninit(mm_partials_cb_id);
                }

                // ── Reconfigure for next iteration ──
                if constexpr (batch > 1 || num_blocks_w_dim > 1 || num_blocks_h_dim > 1) {
#ifdef FUSE_BIAS
                    reconfig_data_format(mm_partials_cb_id, in1_cb_id, bias_cb_id, in0_cb_id);
#else
                    reconfig_data_format_srca(mm_partials_cb_id, in1_cb_id);
#endif
                    mm_block_init_short(in0_cb_id, in1_cb_id, in1_transpose_tile,
                                        out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
        }
    }
}
```

### F2. Gathered variant after migration

```cpp
// bmm_large_block_zm_fused_bias_activation_gathered.cpp (sketch)
// Core matmul uses helper; ring infrastructure stays inline.

void kernel_main() {
    // ... parse args, core type check, CB arrays ...

    mm_block_init(in0_cb_id, in1_cb_id, mm_partials_cb_ids[0], in1_transpose_tile,
                  out_subblock_w, out_subblock_h, in0_block_w);

    for (uint32_t b = 0; b < batch; b++) {
        uint32_t mm_out_cb_id = mm_out_cb_ids[b];
        uint32_t mm_partials_cb_id = mm_partials_cb_ids[b];

        // Ring sync
        cb_wait_front(sync_cb2, 1); cb_pop_front(sync_cb2, 1);

        // ENABLE_GLOBAL_CB: setup in1 read pointer for ring position
        // ... CB pointer manipulation (stays inline) ...

        // Use helper for the K-blocking matmul core
        // NOTE: The gathered variant has per-block actions (CB switching,
        // dynamic K dim) that the helper can't express. For blocks with
        // variable in0_block_w or alternating in0 CB, the K-loop stays inline.
        // The helper covers the COMMON case (single in0 CB, fixed block_w).

        // For the simple case (non-gathered or fixed K):
        compute_kernel_lib::matmul_block<
            in0_cb_id, in1_cb_id, mm_out_cb_id, mm_partials_cb_id,
            in1_transpose_tile, l1_acc,
            /*pack_last_to_interm=*/false, /*pack_relu=*/do_relu
#ifdef SFPU_OP_INIT_ACTIVATION
            , PostSFPU
#endif
            >(
            in0_block_w, in0_num_subblocks, in1_num_subblocks,
            num_blocks, out_subblock_h, out_subblock_w, 1
            );

        // Ring sync + CB pointer reset
        cb_reserve_back(sync_cb, 1); cb_push_back(sync_cb, 1);
        // ... ENABLE_GLOBAL_CB: reset in1 read pointer ...
    }
}
```

**Note**: The gathered variant with `ENABLE_GLOBAL_CB` has per-K-block CB pointer manipulation and alternating in0 CB sources that the helper cannot express. For these paths, the K-loop stays inline, calling the LLK `matmul_block` directly. The helper covers the non-global-CB path. This is acceptable per P4 (caller-managed features).

### F3. conv_bmm_tilize after migration (three-helper compose)

```cpp
// conv_bmm_tilize.cpp (sketch of the three-helper pipeline)

void kernel_main() {
    // ... parse many compile-time args ...

    for (uint32_t b = 0; b < num_blocks_act_h; b++) {
        // Phase 1: Tilize input activations (existing helper)
        compute_kernel_lib::tilize<act_block_w_tiles, in_cb, tilized_in_cb>(num_rows);

        // Phase 2: Matmul via helper
        compute_kernel_lib::matmul_block<
            tilized_in_cb, weight_cb, out_cb, interm_cb,
            false, l1_acc, /*pack_last_to_interm=*/fuse_bias>(
            block_w, in0_nsb, in1_nsb, k_blocks, osh, osw, 1);

        // Phase 3: Bias add via helper (if fuse_bias)
        if constexpr (fuse_bias) {
            // Caller: RELU + format reconfig
            compute_kernel_lib::add_bias_bcast_rows<
                interm_cb, bias_cb, untilize_target_cb, PostSFPU>(
                in0_nsb, in1_nsb, osh, osw, bias_w, PostSFPU{});
        }

        // Phase 4: Untilize output (existing helper or inline)
        if constexpr (untilize_out) {
            compute_kernel_lib::untilize<out_block_w, interm_cb, out_cb>(num_rows);
        }
    }
}
```

---

## Section G -- Backward Compatibility

### bmm_large_block_zm.cpp (the one currently-migrated kernel)

Current call:
```cpp
compute_kernel_lib::matmul_block<cb_in0, cb_in1, cb_out, cb_intermed0>(
    in0_block_w, in0_num_subblocks, in1_num_subblocks, num_k_blocks,
    out_subblock_h, out_subblock_w, batch);
```

After the redesign, this call is **unchanged**. The new template params all have defaults:
- `transpose = false` (matches current)
- `packer_l1_acc = false` (matches current -- software spill/reload)
- `pack_last_to_interm = false` (matches current -- pack to out_cb)
- `pack_relu = false` (matches current -- no RELU)
- `PostComputeFn = NoPostCompute` (matches current -- no callback)

### Behavioral change: 4-phase DST

The implementation changes from 2-phase (`acquire_dst/release_dst`) to 4-phase (`tile_regs_acquire/commit/wait/release`). This is functionally equivalent but allows more MATH-PACK pipelining. The change should be transparent to callers.

To verify: run all matmul C++ integration tests and Python tests before and after, confirm identical pass/fail counts and PCC values.

### Migration path if matmul_block signature changes further

If future phases add more template params or change the function signature, the currently-migrated `bmm_large_block_zm.cpp` would need updating. However, since all new params have defaults, additional template params can be added without breaking existing call sites. The only breaking change would be reordering existing template params (which this design does NOT do).

---

## Section H -- Open Questions

### 1. in0_transpose_tile within the K-loop

The production kernel's `in0_transpose_tile` feature requires per-K-block actions (transpose phase + L1_ACC toggle) inside the K-loop. The helper encapsulates the K-loop, so these actions can't be inserted by the caller.

**Current plan**: Keep the in0_transpose path as inline code in the production kernel. The non-transpose paths use the helper.

**Alternative**: Add a `PreKBlockFn` callback to the helper (called at the start of each K-block iteration). The caller passes a functor that does the transpose. This would allow ALL production kernel paths to use the helper. Risk: adds API complexity for a single call site.

**What would resolve it**: Implementation experience in phase 3. If the inline-K-loop code for in0_transpose is short and clean, keep it inline. If it's error-prone or duplicates significant logic, add the callback.

### 2. SKIP_COMPUTE path

The `SKIP_COMPUTE` path skips `matmul_block` LLK calls but still runs the K-blocking loop structure (for reload logic). The current design has the caller skip the matmul helper entirely.

**Risk**: With SKIP_COMPUTE + FUSE_BIAS + PACKER_L1_ACC, the data in interm_cb may have been populated by a previous operation (DRAM sharded path). The bias phase needs to read it. If the caller skips the matmul helper, it needs to ensure interm_cb is properly populated.

**What would resolve it**: Check whether the SKIP_COMPUTE + FUSE_BIAS path actually needs the K-loop's reload logic, or if the data is already in interm_cb from the dataflow kernel.

### 3. Gathered variant with ENABLE_GLOBAL_CB

The gathered variant's `ENABLE_GLOBAL_CB` path has per-K-block CB pointer manipulation and alternating in0 CB sources. The helper's K-loop can't express these per-block actions.

**Options**: (a) Keep the ENABLE_GLOBAL_CB path as inline K-loop code (current plan). (b) Add a `PreKBlockFn` callback (same as Q1). (c) Provide a single-K-block helper that the gathered kernel calls within its own loop.

**What would resolve it**: Phase 3 implementation. If option (a) leads to significant code duplication with the non-ENABLE_GLOBAL_CB path, reconsider (b) or (c).

### 4. pack_tile_block vs pack_tile loop

The production kernel uses `pack_tile_block(start_dst_index, cb, num_tiles)` for packing. The current helper uses a per-tile `pack_tile(i, cb)` loop. `pack_tile_block` may be more efficient (uses hardware unroll / MOP). The redesigned helper should use `pack_tile_block` to match the production kernel. Verify this doesn't break any configurations.

### 5. Untilize integration

The production kernel's untilize phase uses `reblock_and_untilize` (for the base kernel) and `pack_untilize_dest` (for the gathered variant). The existing `untilize_helpers` wrap `pack_untilize_block`. These may be different LLK paths. Verify whether the existing untilize helper can replace the production kernel's inline untilize code, or if a thin wrapper is needed.

### 6. mm_block_init ownership

Currently, the helper calls `mm_block_init` at the start of its implementation. The production kernel also calls it before the outer loops. With the redesigned helper (called per (bh,bw) iteration with batch=1), the helper would call `mm_block_init` on every iteration, which is wasteful. The production kernel calls `mm_block_init` once and `mm_block_init_short` for re-init.

**Plan**: Remove `mm_block_init` from the helper. The caller calls it once at kernel start. The helper assumes matmul is already initialized. For re-init after bias/untilize phases, the caller calls `mm_block_init_short`. This matches the tilize/untilize pattern where `init` is controlled by the `InitUninitMode` enum, but simpler: the helper always assumes initialized.

**Impact on bmm_large_block_zm.cpp**: The migrated kernel would need to add a `mm_block_init` call before the helper. This is a minor change (1 line added to the caller).

---

## Appendix: Implementation Notes for Phase 3

### matmul_block_helpers.inl pseudocode (new implementation)

```
matmul_block<in0, in1, out, interm, xpose, l1_acc, to_interm, relu, PostFn>
    (bw, nsb_m, nsb_n, k_blocks, osh, osw, batch, post_compute):

  // Static asserts (unchanged)
  // Derive quantities (unchanged)
  // Runtime asserts (unchanged)

  // Note: mm_block_init is caller's responsibility (see Open Question 6)

  pack_target = to_interm ? interm : out  // compile-time

  for b in batch:
    spill = k_blocks > 1
    enable_reload = false
    out_num_tiles_to_wait = out_num_tiles

    for block in k_blocks:
      last_out = (block == k_blocks - 1)

      // PACK_RELU: enable on last block when !to_interm
      if constexpr (relu && !to_interm):
        if last_out:
          PACK(llk_pack_relu_config(ZERO_RELU))

      cb_wait_front(in0, in0_block_tiles)
      cb_wait_front(in1, in1_block_tiles)

      for in0_sb in nsb_m:
        for in1_sb in nsb_n:
          tile_regs_acquire()

          if enable_reload:
            copy_tile_to_dst_init_short_with_dt(in1, interm)
            cb_wait_front(interm, out_tiles)
            copy_block_matmul_partials(interm, 0, 0, out_tiles)
            cb_pop_front(interm, out_tiles)
            mm_block_init_short_with_dt(in0, in1, interm, xpose, osw, osh, bw)

          // Matmul inner dim
          dst_idx = 0; in0_idx = ...; in1_idx = ...
          for k in bw:
            ckernel::matmul_block(in0, in1, in0_idx, in1_idx, dst_idx, xpose, osw, osh, bw)
            in0_idx++; in1_idx += in1_per_core_w

          if last_out:
            post_compute(out_tiles)

            tile_regs_commit()
            cb_reserve_back(pack_target, out_tiles)
            tile_regs_wait()

            // Pack format reconfig (handles FP32_DEST_ACC_EN automatically)
            if constexpr (l1_acc || get_fp32_dest_acc_enabled()):
              PACK(pack_reconfig_data_format(pack_target))

            // L1_ACC toggle on last block
            if constexpr (l1_acc):
              if constexpr (to_interm):
                // FUSE_BIAS path: accumulate on all blocks
                PACK(llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1))
              else:
                // No bias: disable L1_ACC for final output
                PACK(llk_pack_reconfig_l1_acc(0))

            pack_tile_block(0, pack_target, out_tiles)
            tile_regs_release()
            cb_push_back(pack_target, out_tiles)

          else:  // not last_out
            tile_regs_commit()
            if block == 0:
              cb_reserve_back(out, out_num_tiles_to_wait)  // shared memory protection
              out_num_tiles_to_wait += out_tiles
            cb_reserve_back(interm, out_tiles)
            tile_regs_wait()

            if constexpr (l1_acc):
              PACK(llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1))

            pack_tile_block(0, interm, out_tiles)
            tile_regs_release()
            cb_push_back(interm, out_tiles)

      // Post-subblock-loop: manage reload state
      if constexpr (l1_acc):
        if constexpr (to_interm):
          // L1_ACC + bias: advance FIFO, never reload
          if block < k_blocks - 1:
            cb_wait_front(interm, out_block_tiles)
            cb_pop_front(interm, out_block_tiles)
          enable_reload = false
        else:
          // L1_ACC + no bias: advance FIFO, reload on K-1
          if block < k_blocks - 2:
            cb_wait_front(interm, out_block_tiles)
            cb_pop_front(interm, out_block_tiles)
          if block == k_blocks - 2:
            enable_reload = true
      else:
        if spill: enable_reload = true

      cb_pop_front(in0, in0_block_tiles)
      cb_pop_front(in1, in1_block_tiles)
```

### bias_add_helpers.inl pseudocode

```
add_bias_bcast_rows<partials, bias, out, PostFn>
    (nsb_m, nsb_n, osh, osw, bias_w, post_bias):

  out_tiles = osh * osw

  // Format reconfig
  reconfig_data_format_srca(partials)
  reconfig_data_format_srcb(bias)
  pack_reconfig_data_format(out)

  // Init bias broadcast
  add_bcast_rows_init_short(partials, bias)

  // Wait for bias tiles (no-op if already available)
  cb_wait_front(bias, bias_w)

  for in0_sb in nsb_m:
    bias_offset = 0
    for in1_sb in nsb_n:
      cb_wait_front(partials, out_tiles)
      tile_regs_acquire()

      // Row-broadcast bias addition
      for j in osh:
        bcast_tile = bias_offset
        for k in osw:
          tile_i = j * osw + k
          add_tiles_bcast_rows(partials, bias, tile_i, bcast_tile, tile_i)
          bcast_tile++

      // PostBiasFn fires BEFORE commit (matches production kernel's SFPU placement)
      post_bias(out_tiles)

      tile_regs_commit()
      cb_pop_front(partials, out_tiles)

      cb_reserve_back(out, out_tiles)
      tile_regs_wait()
      for i in out_tiles:
        pack_tile(i, out)
      tile_regs_release()
      cb_push_back(out, out_tiles)

      bias_offset += osw

  // NOTE: does NOT pop bias. Caller manages bias lifetime.
```
