# MatmulOp Design Specification

## Overview

`MatmulOp` is a header-only class in `tt_metal/hw/inc/api/compute/matmul_op.h` that wraps
the `matmul_tiles` and `matmul_block` LLK calls. It provides three usage modes to cover all
40 active call sites, from deeply embedded single-tile matmul in complex pipelines to fully
automatic blocked matmul with spill/reload.

The class is a compile-time-configured, zero-overhead abstraction. It stores no heap state,
uses no virtual dispatch, and inlines to the same code a hand-written kernel would produce.

---

## File Location

```
tt_metal/hw/inc/api/compute/matmul_op.h
```

This file includes `api/compute/matmul.h`, `api/compute/experimental/matmul_custom.h`,
`api/compute/tile_move_copy.h`, and `api/compute/reg_api.h`.

---

## Configuration Struct

```cpp
namespace ckernel {

struct MatmulOpConfig {
    // --- Required CB IDs ---
    uint32_t in0_cb_id;       // CB for the A matrix (left operand)
    uint32_t in1_cb_id;       // CB for the B matrix (right operand)
    uint32_t out_cb_id;       // CB for the output (or intermediate partials for spill mode)

    // --- Block dimensions (required for block mode, ignored for tile mode) ---
    uint32_t ct_dim = 1;      // Output subblock column dimension in tiles (subblock_w)
    uint32_t rt_dim = 1;      // Output subblock row dimension in tiles (subblock_h)
    uint32_t kt_dim = 1;      // Inner dimension block size in tiles (in0_block_w)

    // --- Transpose ---
    bool transpose = false;   // Transpose B tiles (width-height swap)

    // --- Partials / Spill buffer ---
    // CB for storing partial accumulations when inner dim is blocked (num_blocks_inner > 1).
    // Set to 0 to disable spill/reload.
    uint32_t partials_cb_id = 0;

    // --- Architecture-specific options ---
    // When true, use matmul_block_no_mop (direct replay buffer) instead of matmul_block
    // with MOP. Only effective on Blackhole. Used by SDPA streaming kernels.
    bool use_no_mop = false;
};

} // namespace ckernel
```

### Field Documentation

| Field | Description | Required | Default |
|-------|-------------|----------|---------|
| `in0_cb_id` | Circular buffer for A operand | Always | -- |
| `in1_cb_id` | Circular buffer for B operand | Always | -- |
| `out_cb_id` | Circular buffer for output (or intermediate when using partials) | Always | -- |
| `ct_dim` | Number of tile columns in the output subblock | Block mode only | 1 |
| `rt_dim` | Number of tile rows in the output subblock | Block mode only | 1 |
| `kt_dim` | Number of tiles in the inner dimension per block | Block mode only | 1 |
| `transpose` | Apply W/H transpose to B tiles | Optional | false |
| `partials_cb_id` | CB for intermediate partial sums during spill/reload. 0 = disabled. | Mode 2/3 when `num_blocks_inner > 1` | 0 |
| `use_no_mop` | Use `matmul_block_no_mop` variant (Blackhole SDPA perf optimization) | Optional | false |

---

## Class Definition

```cpp
namespace ckernel {

template <bool IsBlockMode>
class MatmulOp {
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /**
     * Construct a MatmulOp from a configuration struct.
     * Does NOT perform hardware initialization -- call init() or init_short()
     * explicitly after construction.
     *
     * The IsBlockMode template parameter selects between tile-level and block-level
     * matmul. When IsBlockMode=false, calls matmul_tiles(); when true, calls
     * matmul_block() (or matmul_block_no_mop if config.use_no_mop).
     */
    FORCE_INLINE explicit MatmulOp(const MatmulOpConfig& cfg);

    // -------------------------------------------------------------------------
    // Initialization (call before first matmul operation)
    // -------------------------------------------------------------------------

    /**
     * Full initialization: configures unpacker, math engine, and packer for matmul.
     * Call once at kernel startup before any matmul operation.
     *
     * Tile mode:  calls mm_init(in0, in1, out, transpose)
     * Block mode: calls mm_block_init(in0, in1, out, transpose, ct, rt, kt)
     */
    FORCE_INLINE void init() const;

    /**
     * Short re-initialization: reconfigures unpacker and math engine only (no packer).
     * Use when switching back to matmul mode after other compute operations (bias,
     * activation, reduce) within the same kernel.
     *
     * Tile mode:  calls mm_init_short(in0, in1, transpose)
     * Block mode: calls mm_block_init_short(in0, in1, transpose, ct, rt, kt)
     *             or mm_no_mop_init_short() when use_no_mop is set
     */
    FORCE_INLINE void init_short() const;

    /**
     * Short re-initialization with data format reconfig for srcA.
     * Use when the previous operation used a different CB on srcA and we need
     * to reconfigure the data format before resuming matmul.
     *
     * @param old_in1_cb_id  The CB that was previously configured on srcA
     *
     * Tile mode:  calls mm_init_short_with_dt(in0, in1, old_in1, transpose)
     * Block mode: calls mm_block_init_short_with_dt(in0, in1, old_in1, transpose, ct, rt, kt)
     */
    FORCE_INLINE void init_short_with_dt(uint32_t old_in1_cb_id) const;

    /**
     * Short re-initialization with data format reconfig for both srcA and srcB.
     * Block mode only. Use after operations that changed both source CB formats.
     *
     * @param old_in0_cb_id  The CB that was previously configured on srcB (note: srcB maps to in0 operand)
     * @param old_in1_cb_id  The CB that was previously configured on srcA (note: srcA maps to in1 operand)
     *
     * Block mode: calls mm_block_init_short_with_both_dt(in0, in1, old_in0, old_in1, transpose, ct, rt, kt)
     * Tile mode:  static_assert failure (not supported)
     */
    FORCE_INLINE void init_short_with_both_dt(uint32_t old_in0_cb_id, uint32_t old_in1_cb_id) const;

    // -------------------------------------------------------------------------
    // Mode 1: Low-level — single matmul call
    // -------------------------------------------------------------------------

    /**
     * Perform a single matmul_tiles or matmul_block call.
     * DST must already be acquired by the caller.
     * Caller manages all CB wait/pop, DST acquire/release, and pack operations.
     *
     * @param in0_tile_index  Index of tile/block in in0 CB
     * @param in1_tile_index  Index of tile/block in in1 CB
     * @param dst_tile_index  Index in DST register to accumulate into
     *
     * For tile mode (IsBlockMode=false):
     *   calls matmul_tiles(in0, in1, in0_tile_index, in1_tile_index, dst_tile_index)
     *
     * For block mode (IsBlockMode=true):
     *   calls matmul_block(in0, in1, in0_tile_index, in1_tile_index, dst_tile_index,
     *                       transpose, ct_dim, rt_dim, kt_dim)
     *   or matmul_block_no_mop() if use_no_mop is set
     */
    FORCE_INLINE void matmul(uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t dst_tile_index) const;

    // -------------------------------------------------------------------------
    // Mode 2: Semi-automatic — accumulate + end subblock
    // -------------------------------------------------------------------------

    /**
     * Acquire the DST register for MATH.
     * Wraps tile_regs_acquire().
     */
    FORCE_INLINE void begin_subblock() const;

    /**
     * Accumulate across the inner dimension for one subblock.
     * Calls matmul() in a loop for inner_dim iterations, advancing in0/in1 indices
     * according to the block-level striding convention:
     *   - in0 index increments by 1 each inner step
     *   - in1 index increments by in1_stride each inner step
     *
     * DST must already be acquired (via begin_subblock or tile_regs_acquire).
     *
     * @param in0_index_start  Starting tile index in in0 CB for this subblock
     * @param in1_index_start  Starting tile index in in1 CB for this subblock
     * @param dst_index_start  Starting DST index (typically 0 for each subblock)
     * @param inner_dim        Number of inner-dimension steps to accumulate
     * @param in1_stride       Stride to advance in1 index per inner step.
     *                         For block mode: typically N (full output width in tiles)
     *                         or in1_block_w. For tile mode: typically in1_per_core_w.
     */
    FORCE_INLINE void accumulate(
        uint32_t in0_index_start,
        uint32_t in1_index_start,
        uint32_t dst_index_start,
        uint32_t inner_dim,
        uint32_t in1_stride) const;

    /**
     * Commit DST from MATH to PACK, wait for PACK availability, then pack tiles
     * to the output CB.
     *
     * Performs: tile_regs_commit() -> out_cb.reserve_back(num_tiles) ->
     *          tile_regs_wait() -> pack_tile loop -> tile_regs_release() ->
     *          out_cb.push_back(num_tiles)
     *
     * @param dest_cb_id    CB to pack output tiles into
     * @param num_tiles     Number of tiles to pack (typically ct_dim * rt_dim)
     */
    FORCE_INLINE void end_to_output(uint32_t dest_cb_id, uint32_t num_tiles) const;

    /**
     * Commit DST and pack to partials CB for spill (not the final output).
     * Does NOT call tile_regs_release -- the caller controls when to release.
     *
     * Performs: tile_regs_commit() -> partials_cb.reserve_back(num_tiles) ->
     *          tile_regs_wait() -> pack_tile loop -> partials_cb.push_back(num_tiles)
     *          -> tile_regs_release()
     *
     * @param num_tiles  Number of tiles to spill
     */
    FORCE_INLINE void end_to_partials(uint32_t num_tiles) const;

    /**
     * Reload partial accumulations from the partials CB back into DST.
     * Reconfigures data format from matmul source to partials CB, copies tiles,
     * then reconfigures back to matmul mode.
     *
     * Must be called after begin_subblock() and before accumulate() for
     * inner-dim blocks after the first.
     *
     * @param num_tiles  Number of tiles to reload into DST
     */
    FORCE_INLINE void reload_partials(uint32_t num_tiles) const;

    // -------------------------------------------------------------------------
    // Mode 3: Automatic — full blocked matmul
    // -------------------------------------------------------------------------

    /**
     * Execute a complete blocked matmul: C[M,N] = A[M,K] * B[K,N]
     *
     * Manages the full loop nest: batch -> blocks_h -> blocks_w -> blocks_inner,
     * with subblock tiling, DST acquire/release, and spill/reload when
     * num_blocks_inner > 1 and partials_cb_id != 0.
     *
     * The caller must have called init() before this. The caller must ensure
     * that the reader/writer kernels produce/consume tiles in the expected order.
     *
     * For tile mode (IsBlockMode=false):
     *   Iterates batch * Mt * Nt output tiles, each accumulating Kt inner tiles.
     *   Each output tile gets: acquire -> K matmul_tiles -> pack -> release.
     *
     * For block mode (IsBlockMode=true):
     *   Uses subblocking with in0_num_subblocks * in1_num_subblocks per output block.
     *   Each subblock gets: acquire -> [reload] -> K matmul_blocks -> pack -> release.
     *
     * @param batch               Number of batches
     * @param num_blocks_h        Number of output row blocks
     * @param num_blocks_w        Number of output column blocks
     * @param num_blocks_inner    Number of inner-dimension blocks (K accumulation steps)
     * @param in0_num_subblocks   Number of row subblocks within each output block
     * @param in1_num_subblocks   Number of column subblocks within each output block
     * @param in0_block_num_tiles Total tiles per in0 block (rt_dim * kt_dim * in0_num_subblocks)
     * @param in1_block_num_tiles Total tiles per in1 block (ct_dim * kt_dim * in1_num_subblocks)
     * @param in1_block_w         Full width of in1 block in tiles (ct_dim * in1_num_subblocks)
     */
    FORCE_INLINE void run(
        uint32_t batch,
        uint32_t num_blocks_h,
        uint32_t num_blocks_w,
        uint32_t num_blocks_inner,
        uint32_t in0_num_subblocks,
        uint32_t in1_num_subblocks,
        uint32_t in0_block_num_tiles,
        uint32_t in1_block_num_tiles,
        uint32_t in1_block_w) const;

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    /** Return the stored configuration. */
    FORCE_INLINE const MatmulOpConfig& config() const;

    /** Return the in0 CB id. */
    FORCE_INLINE uint32_t in0_cb() const;

    /** Return the in1 CB id. */
    FORCE_INLINE uint32_t in1_cb() const;

    /** Return the output CB id. */
    FORCE_INLINE uint32_t out_cb() const;

private:
    MatmulOpConfig cfg_;
};

// -------------------------------------------------------------------------
// Type aliases for convenience
// -------------------------------------------------------------------------
using TileMatmulOp  = MatmulOp<false>;  // matmul_tiles wrapper
using BlockMatmulOp = MatmulOp<true>;   // matmul_block wrapper

} // namespace ckernel
```

---

## Usage Patterns

### Mode 1: Low-level

The caller manages everything: CB waits, DST ownership, packing, and loop structure.
`MatmulOp` is used only as a clean way to call the underlying LLK with the right arguments.

```cpp
// Example: RoPE matmul (T14) -- tile mode
MatmulOpConfig cfg{
    .in0_cb_id = args.in_cb,
    .in1_cb_id = args.trans_mat_cb,
    .out_cb_id = args.rotated_in_interm_cb,
};
TileMatmulOp mm(cfg);
mm.init_short();

tile_regs_acquire();
for (uint32_t j = 0; j < Wt; ++j) {
    mm.matmul(j, 0, j);        // individual tile matmul
}
tile_regs_commit();
tile_regs_wait();
for (uint32_t j = 0; j < Wt; ++j) {
    pack_tile(j, args.rotated_in_interm_cb, j);
}
tile_regs_release();
```

```cpp
// Example: MOE gate matmul (B11) -- block mode with ct_dim=2
MatmulOpConfig cfg{
    .in0_cb_id = cb_s2c_in,
    .in1_cb_id = cb_r2c_w,
    .out_cb_id = cb_s2c_out,
    .ct_dim = 2,
    .rt_dim = 1,
    .kt_dim = 1,
};
BlockMatmulOp mm(cfg);
mm.init();

tile_regs_acquire();
for (uint32_t block_id = 0; block_id < w_num_blocks; ++block_id) {
    cb_wait_front(cb_r2c_w, w_tiles_per_block);
    for (uint32_t tile_id = 0; tile_id < w_tiles_per_block; tile_id += 2) {
        mm.matmul(tile_index++, tile_id, 0);
    }
    cb_pop_front(cb_r2c_w, w_tiles_per_block);
}
tile_regs_commit();
// ... pack ...
```

### Mode 2: Semi-automatic

The caller provides the loop structure but delegates DST management and spill/reload
to `MatmulOp`. Custom fusion code (bias, activation, mask) can be inserted between
`accumulate()` and `end_to_output()`.

```cpp
// Example: bmm_large_block_zm_fused_bias_activation (B1) -- simplified
MatmulOpConfig cfg{
    .in0_cb_id = in0_cb_id,
    .in1_cb_id = in1_cb_id,
    .out_cb_id = mm_out_cb_id,
    .ct_dim = out_subblock_w,
    .rt_dim = out_subblock_h,
    .kt_dim = in0_block_w,
    .transpose = in1_transpose_tile,
    .partials_cb_id = mm_partials_cb_id,
};
BlockMatmulOp mm(cfg);
mm.init();

for (uint32_t block = 0; block < num_blocks_inner; block++) {
    bool last_out = block == (num_blocks_inner - 1);
    // ... wait for CB data ...

    for (uint32_t in0_sub = 0; in0_sub < in0_num_subblocks; in0_sub++) {
        for (uint32_t in1_sub = 0; in1_sub < in1_num_subblocks; in1_sub++) {
            mm.begin_subblock();

            if (enable_reload) {
                mm.reload_partials(out_subblock_num_tiles);
            }

            mm.accumulate(in0_index, in1_index, 0, in0_block_w, in1_block_w);

            if (last_out) {
                // --- CUSTOM FUSION POINT ---
                // Insert bias, SFPU activation here (between accumulate and end)
                #ifdef FUSE_BIAS
                // ... apply bias in DST ...
                #endif
                #ifdef SFPU_OP_INIT_ACTIVATION
                // ... apply activation in DST ...
                #endif
                mm.end_to_output(mm_out_cb_id, out_subblock_num_tiles);
            } else {
                mm.end_to_partials(out_subblock_num_tiles);
            }
        }
    }
    enable_reload = true;
}
```

```cpp
// Example: SDPA matmul_blocks with mask fusion (B5)
MatmulOpConfig cfg{
    .in0_cb_id = in0_cb,
    .in1_cb_id = in1_cb,
    .out_cb_id = out_cb,
    .ct_dim = subblock_w,
    .rt_dim = subblock_h,
    .kt_dim = in0_block_w,
    .transpose = true,
};
BlockMatmulOp mm(cfg);
mm.init_short();

for (uint32_t in0_sub = 0; in0_sub < in0_num_subblocks; ++in0_sub) {
    for (uint32_t in1_sub = 0; in1_sub < in1_num_subblocks; ++in1_sub) {
        mm.begin_subblock();
        mm.accumulate(in0_index, in1_index, 0, in0_block_w, N);

        // --- MASK FUSION ---
        if (add_mask) {
            cb_wait_front(mask_cb, out_subblock_num_tiles);
            add_tiles_init(zero_cb, mask_cb, true);
            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                add_tiles(zero_cb, mask_cb, 0, i, i);
            }
        }

        mm.end_to_output(out_cb, out_subblock_num_tiles);
    }
}
```

### Mode 3: Automatic

One call does everything. No custom fusion is possible.

```cpp
// Example: simple bmm.cpp (T1)
MatmulOpConfig cfg{
    .in0_cb_id = cb_in0,
    .in1_cb_id = cb_in1,
    .out_cb_id = cb_out,
};
TileMatmulOp mm(cfg);
mm.init();
mm.run(batch, Mt, Nt, Kt, 1, 1, 1, 1, 1);
```

```cpp
// Example: minimal_matmul block mode (B9/B10)
MatmulOpConfig cfg{
    .in0_cb_id = in0_cb,
    .in1_cb_id = in1_cb,
    .out_cb_id = out_cb,
    .ct_dim = subblock_w,
    .rt_dim = subblock_h,
    .kt_dim = K_block_tiles,
};
BlockMatmulOp mm(cfg);
mm.init();
mm.run(1, M_blocks, N_blocks, K_blocks,
       in0_num_subblocks, in1_num_subblocks,
       in0_block_num_tiles, in1_block_num_tiles, in1_block_w);
```

---

## Call Site Mapping (All 40 Sites)

### matmul_tiles sites (T1-T14)

| ID | Mode | MatmulOp Type | Methods Used | Notes |
|----|------|---------------|-------------|-------|
| T1 | 3 | `TileMatmulOp` | `init()`, `run()` | Simplest case. `run(batch, Mt, Nt, Kt, 1, 1, 1, 1, 1)`. Uses `acquire_dst`/`release_dst` pattern internally (old API, equivalent to tile_regs 4-step). |
| T2 | 2 | `TileMatmulOp` | `init()`, `begin_subblock()`, `accumulate()`, `end_to_output()`, `end_to_partials()`, `reload_partials()` | Uses h*w*K loop with tile-level subblocking and explicit spill/reload. `accumulate()` called with per-tile in0/in1 index computation. Alternatively, direct `matmul()` calls (Mode 1) may be simpler here since the striding pattern does not match the standard inner-dim loop: in0 strides by `in0_block_w`, in1 strides by `in1_per_core_w`. **Recommended: Mode 1** with manual DST management, since the h/w/inner loop structure with per-tile index arithmetic does not cleanly map to `accumulate()`. |
| T3 | 1 | `TileMatmulOp` | `init()`, `init_short_with_dt()`, `matmul()` | Per-row matmul with untilize/retilize cycle. Each row: acquire -> K tiles of matmul -> pack -> untilize -> retilize -> reinit matmul. Must use Mode 1 because of interleaved untilize/retilize operations. |
| T4 | 1 | `TileMatmulOp` | `init()`, `matmul()` | Architecture-conditional code (GS vs non-GS), uses `pack_untilize_dest`. Same h/w/inner loop as T2 with `matmul_tiles` but with tile-level indexing. Mode 1 is appropriate. |
| T5 | 1 | `TileMatmulOp` | `init()`, `matmul()` | Width reduction via matmul with scaler tile. Simple: acquire -> Wt matmul_tiles -> pack -> release. Mode 1 correct since it is embedded in a reduce pipeline. |
| T6 | 1 | `TileMatmulOp` | `init_short()`, `matmul()` | Single tile matmul with transpose/mask. Part of complex moreh_matmul kernel with multiple code paths. |
| T7 | 1 | `TileMatmulOp` | `init()`, `matmul()` | Simple K loop: tile_regs_acquire -> K iterations of (wait, matmul, pop) -> commit -> pack. |
| T8 | 1 | `TileMatmulOp` | `init_short()`, `matmul()` | Width reduce + masking. Two call sites: one in accumulation loop, one for single remaining tile. Both use `mm_init_short` before each matmul. |
| T9 | 1 | `TileMatmulOp` | `init_short()`, `matmul()` | Same pattern as T8 but for moreh_sum_w. |
| T10 | 1 | `TileMatmulOp` | `init_short()`, `matmul()` | tt-train SDPA forward: diagonal QxK^T blocked. |
| T11 | 1 | `TileMatmulOp` | `init_short()`, `matmul()` | tt-train SDPA QKxV blocked. |
| T12 | 1 | `TileMatmulOp` | `init_short()`, `matmul()` | tt-train SDPA backward: reduce + reciprocal pipeline. |
| T13 | 1 | `TileMatmulOp` | `init()`, `init_short_with_dt()`, `matmul()` | 6-deep loop nest with interleaved tilize/untilize/bias/SFPU. Complex control flow makes Mode 1 the only viable option. |
| T14 | 1 | `TileMatmulOp` | `init_short()`, `matmul()` | RoPE step 1: matmul is one step in a 4-step pipeline (matmul, mul_bcast, mul_bcast, add). |

### matmul_block sites (B1-B16)

| ID | Mode | MatmulOp Type | Methods Used | Notes |
|----|------|---------------|-------------|-------|
| B1 | 2 | `BlockMatmulOp` | `init()`, `init_short()`, `init_short_with_dt()`, `begin_subblock()`, `accumulate()`, `end_to_output()`, `end_to_partials()`, `reload_partials()` | Most complex production kernel. Has bias fusion, SFPU activation, untilize, PACKER_L1_ACC, transpose, and spill/reload. Custom code between accumulate and pack for bias+activation. |
| B2 | 2 | `BlockMatmulOp` | Same as B1 | Same kernel as B1 but with gathered input CB management. Same matmul core logic. |
| B3 | 2 | `BlockMatmulOp` | `init()`, `init_short_with_dt()`, `begin_subblock()`, `accumulate()`, `end_to_output()`, `end_to_partials()`, `reload_partials()` | Conv2d BMM with tilize preprocessing. Same inner matmul loop as B1. |
| B4 | 2 | `BlockMatmulOp` | `init_short()`, `begin_subblock()`, `accumulate()`, `end_to_output()` | SDPA streaming: uses `matmul_block_no_mop` on Blackhole. Set `use_no_mop = true`. Single accumulation (no spill). |
| B5 | 2 | `BlockMatmulOp` | `init_short()`, `begin_subblock()`, `accumulate()`, `end_to_output()` | SDPA `matmul_blocks` helper: subblock loop with optional mask fusion between accumulate and pack. No spill (single inner block). |
| B6 | 2 | `BlockMatmulOp` | `init_short()`, `begin_subblock()`, `matmul()`, `end_to_output()` | SDPA `matmul_reduce`: Mx1 reduction. Single matmul_block per subblock. Uses Mode 1 matmul() within Mode 2 DST management. |
| B7 | 2 | `BlockMatmulOp` | Via `matmul_blocks` helper | SDPA decode: two calls to the same `matmul_blocks` pattern as B5. |
| B8 | 1 | `BlockMatmulOp` | `init()`, `matmul()` | TopK router: 1x1x1 tile-by-tile with ct_dim=1. Simple accumulation loop with manual DST. |
| B9 | 3 | `BlockMatmulOp` | `init()`, `run()` | minimal_matmul: standard M*N*K blocked matmul. Uses PACKER_L1_ACC for K accumulation. Direct map to `run()`. |
| B10 | 3 | `BlockMatmulOp` | `init_short()`, `run()` | Conv3d: standard subblock + K accumulation. Uses `matmul_blocks` helper function internally. |
| B11 | 1 | `BlockMatmulOp` | `init()`, `matmul()` | MOE gate: ct_dim=2 (send core) or ct_dim=1 (compute core). Ring accumulation with block-by-block weight streaming. Custom tile indexing. |
| B12 | 1 | `BlockMatmulOp` | `init()`, `matmul()` | MLA matmul_wo: ct_dim=7. K tiles streamed in blocks with pack_tile_block at end. |
| B13 | 1 | `BlockMatmulOp` | `init()`, `matmul()` | CCL MOE compute: ct_dim=4. Two matmul_block call sites (W0/W1 and W2), both with custom inter-block signaling and SwiGLU. |
| B14 | 1 | `BlockMatmulOp` | `init()`, `matmul()` | CCL MOE GPT: ct_dim=4, 8 matmul_block calls. Bias via ones tile, SwiGLU activation, untilize. Most complex Mode 1 usage. |
| B15 | 3 | `BlockMatmulOp` | `init_short()`, `run()` | all_gather_minimal_matmul: standard K-accumulation with L1_ACC. Same `matmul_blocks` helper. |
| B16 | 2 | `BlockMatmulOp` | Same as B1 | llama_all_gather variant of B2. Same matmul core logic. |

---

## Method Semantics (Detailed)

### `matmul(in0_tile_index, in1_tile_index, dst_tile_index)`

**Tile mode (`IsBlockMode = false`):**
```
matmul_tiles(cfg_.in0_cb_id, cfg_.in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index);
```

**Block mode (`IsBlockMode = true`, `use_no_mop = false`):**
```
matmul_block(cfg_.in0_cb_id, cfg_.in1_cb_id, in0_tile_index, in1_tile_index,
             dst_tile_index, cfg_.transpose, cfg_.ct_dim, cfg_.rt_dim, cfg_.kt_dim);
```

**Block mode (`IsBlockMode = true`, `use_no_mop = true`):**
```
matmul_block_no_mop(cfg_.in0_cb_id, cfg_.in1_cb_id, in0_tile_index, in1_tile_index,
                     dst_tile_index, cfg_.transpose, cfg_.ct_dim, cfg_.rt_dim, cfg_.kt_dim);
```

### `accumulate(in0_start, in1_start, dst_start, inner_dim, in1_stride)`

```
for (uint32_t k = 0; k < inner_dim; ++k) {
    matmul(in0_start + k, in1_start + k * in1_stride, dst_start);
    // Note: dst_start stays the same -- matmul_block increments internally
    // for block mode. For tile mode, dst accumulates in place.
}
```

The `in1_stride` parameter is critical: it captures the diverse striding patterns across
call sites. For standard blocked matmul, `in1_stride = in1_block_w` (full N width). For
SDPA, it may be a different value like `N`. The caller computes the correct stride.

### `begin_subblock()`

```
tile_regs_acquire();
```

### `end_to_output(dest_cb_id, num_tiles)`

```
tile_regs_commit();
cb_reserve_back(dest_cb_id, num_tiles);
tile_regs_wait();
for (uint32_t i = 0; i < num_tiles; i++) {
    pack_tile(i, dest_cb_id);
}
tile_regs_release();
cb_push_back(dest_cb_id, num_tiles);
```

Note: This uses the simple sequential pack pattern. Call sites that need out-of-order
packing (pack_tile<true> with explicit output_tile_index), untilize packing, or other
custom pack patterns should use Mode 1 and call `tile_regs_commit()` / `tile_regs_wait()`
/ `tile_regs_release()` directly.

### `end_to_partials(num_tiles)`

```
tile_regs_commit();
cb_reserve_back(cfg_.partials_cb_id, num_tiles);
tile_regs_wait();
for (uint32_t i = 0; i < num_tiles; i++) {
    pack_tile(i, cfg_.partials_cb_id);
}
tile_regs_release();
cb_push_back(cfg_.partials_cb_id, num_tiles);
```

### `reload_partials(num_tiles)`

```
copy_tile_to_dst_init_short_with_dt(cfg_.in1_cb_id, cfg_.partials_cb_id);
cb_wait_front(cfg_.partials_cb_id, num_tiles);
copy_block_matmul_partials(cfg_.partials_cb_id, 0, 0, num_tiles);
cb_pop_front(cfg_.partials_cb_id, num_tiles);
// Reconfigure back to matmul mode
if constexpr (IsBlockMode) {
    mm_block_init_short_with_dt(cfg_.in0_cb_id, cfg_.in1_cb_id, cfg_.partials_cb_id,
                                 cfg_.transpose, cfg_.ct_dim, cfg_.rt_dim, cfg_.kt_dim);
} else {
    mm_init_short_with_dt(cfg_.in0_cb_id, cfg_.in1_cb_id, cfg_.partials_cb_id, cfg_.transpose);
}
```

### `run(batch, num_blocks_h, num_blocks_w, num_blocks_inner, ...)`

**Tile mode (`IsBlockMode = false`):**
```
for (batch) {
    for (Mt) {
        for (Nt) {
            tile_regs_acquire();
            for (Kt) {
                cb_wait_front(in0, 1);
                cb_wait_front(in1, 1);
                matmul_tiles(in0, in1, 0, 0, 0);
                cb_pop_front(in0, 1);
                cb_pop_front(in1, 1);
            }
            tile_regs_commit();
            cb_reserve_back(out, 1);
            tile_regs_wait();
            pack_tile(0, out);
            tile_regs_release();
            cb_push_back(out, 1);
        }
    }
}
```

**Block mode (`IsBlockMode = true`):**
```
for (batch) {
    for (blocks_h) {
        for (blocks_w) {
            enable_reload = false;
            for (block_inner) {
                last_out = (block_inner == num_blocks_inner - 1);
                cb_wait_front(in0, in0_block_num_tiles);
                cb_wait_front(in1, in1_block_num_tiles);

                for (in0_sub) {
                    for (in1_sub) {
                        begin_subblock();
                        if (enable_reload) reload_partials(subblock_tiles);
                        accumulate(in0_idx, in1_idx, 0, kt_dim, in1_block_w);
                        if (last_out)  end_to_output(out_cb, subblock_tiles);
                        else           end_to_partials(subblock_tiles);
                    }
                }

                if (num_blocks_inner > 1) enable_reload = true;
                cb_pop_front(in0, in0_block_num_tiles);
                cb_pop_front(in1, in1_block_num_tiles);
            }
        }
    }
}
```

---

## Design Decisions and Rationale

### 1. Template on IsBlockMode rather than runtime dispatch

**Decision:** `MatmulOp<false>` = tile mode, `MatmulOp<true>` = block mode.

**Rationale:** Every call site statically knows whether it uses `matmul_tiles` or
`matmul_block`. A template parameter eliminates a branch on every call and allows the
compiler to dead-strip the unused path entirely. The type aliases `TileMatmulOp` and
`BlockMatmulOp` provide readable names.

### 2. Config struct instead of constructor parameters

**Decision:** Pass a `MatmulOpConfig` struct to the constructor.

**Rationale:** The underlying LLK functions take 4-9 parameters of the same type
(`uint32_t`). Positional constructors are error-prone. A struct with named fields provides
self-documenting construction via designated initializers (C++20, supported by the
TT-Metal toolchain). Fields have sensible defaults so only relevant ones need to be set.

### 3. Separate init() and init_short() from constructor

**Decision:** Construction does NOT call hardware init. The caller must call `init()` or
`init_short()` explicitly.

**Rationale:** Many call sites need to reinitialize matmul mode multiple times during
kernel execution (after bias addition, after untilize, etc.). Separating construction from
initialization matches the existing codebase pattern where `mm_init` and
`mm_block_init_short` are called at different points. This also allows creating the
`MatmulOp` object early and deferring init to the right point.

### 4. use_no_mop as a config flag rather than a separate class

**Decision:** A boolean flag in the config, checked at compile time via `if constexpr`
where possible, or runtime branch in `matmul()`.

**Rationale:** `matmul_block_no_mop` has the exact same signature as `matmul_block`. Only
the SDPA streaming kernel uses it, and only on Blackhole. Making it a separate class would
create a third template dimension with minimal benefit. A flag keeps the API simple. The
implementation can use `#ifdef ARCH_BLACKHOLE` to compile-time eliminate the branch on
other architectures.

### 5. accumulate() takes in1_stride rather than computing it

**Decision:** The caller provides `in1_stride` explicitly.

**Rationale:** Different call sites use different stride patterns:
- Standard BMM: `in1_stride = in1_block_w` (the full width of the in1 block)
- SDPA: `in1_stride = N` (the full output width)
- Tile mode: `in1_stride = in1_per_core_w`
- Conv3d: `in1_stride = N`

Rather than trying to infer or store all possible stride conventions, we let the caller
pass the stride. This keeps the API general without adding fields to the config that only
apply to specific call patterns.

### 6. end_to_output() uses simple sequential pack

**Decision:** `end_to_output()` packs tiles sequentially (pack_tile(0), pack_tile(1), ...).

**Rationale:** This covers the majority of call sites (T1, T2, T5-T12, B1-B3, B5-B7,
B9-B10, B15-B16). Call sites that need out-of-order packing (`pack_tile<true>` with
explicit output_tile_index), pack_untilize_dest, or custom pack patterns are already
Mode 1 by necessity (B11-B14, T3, T4, T13). They use `matmul()` directly and manage
packing themselves. Adding template parameters for pack style would complicate the API
for little gain.

### 7. Mode 2 methods are available to Mode 1 users

**Decision:** `begin_subblock()`, `end_to_output()`, etc. are public and can be called
even when the caller is primarily using `matmul()` directly.

**Rationale:** Some call sites (B5, B6) use a hybrid pattern: they call `begin_subblock()`
and `end_to_output()` from Mode 2, but call `matmul()` from Mode 1 within the subblock
(because they insert custom mask/reduce operations between matmul calls). Making all methods
available avoids artificial restrictions.

### 8. No CB wait/pop in accumulate()

**Decision:** `accumulate()` does NOT call `cb_wait_front` or `cb_pop_front`.

**Rationale:** CB management patterns vary across call sites:
- Some wait for the full block before the inner loop (B1, B5)
- Some wait tile-by-tile inside the loop (T1, T5, T7)
- Some wait for subsets of tiles (B11, B14)
- Some never pop (reusing the same data across subblocks)

Putting CB management inside `accumulate()` would break the majority of call sites. The
caller handles CB management and `accumulate()` only does the matmul computation.

### 9. No PACKER_L1_ACC support in the class

**Decision:** The class does not manage `llk_pack_reconfig_l1_acc()` or
`pack_reconfig_data_format()` calls.

**Rationale:** PACKER_L1_ACC is controlled by compile-time defines (`#ifdef PACKER_L1_ACC`)
and requires careful sequencing with spill/reload and output block boundaries. Different
kernels use it differently (B1 has different L1_ACC logic with vs without FUSE_BIAS). It
is orthogonal to the matmul operation itself. Callers using PACKER_L1_ACC can call the
L1_ACC APIs directly alongside Mode 2 MatmulOp methods. If the `run()` method
(Mode 3) needs L1_ACC, it can be added as a future `bool use_l1_acc` config field.

### 10. No fusion callback mechanism

**Decision:** No std::function, function pointer, or template callback for fusion.

**Rationale:** RISC-V kernel environment prohibits std::function. Template callbacks would
work but explosion of template parameters for bias CB, activation type, mask CB, etc.
makes the API harder to use than Mode 2's explicit fusion points. Mode 2's
begin_subblock/accumulate/end pattern naturally provides fusion points without any callback
mechanism.

---

## Architecture Compatibility

The class works on all supported architectures:

- **Wormhole B0**: All methods map directly to existing LLK calls.
- **Blackhole**: `use_no_mop = true` routes to `matmul_block_no_mop`. Dynamic throttling
  in `matmul_block` is handled by the underlying `matmul.h` implementation (transparent).
- **Quasar**: The underlying `matmul_tiles` and `matmul_block` in `matmul.h` already have
  Quasar-specific `#ifdef` branches. `MatmulOp` inherits this support transparently.
  `use_no_mop` has no effect on Quasar (no `matmul_block_no_mop` implementation exists).

---

## Include Dependencies

```cpp
#pragma once

#include "api/compute/matmul.h"
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
```

The caller does NOT need to include `matmul.h` separately -- `matmul_op.h` re-exports it.

---

## Non-Goals

The following are explicitly out of scope for this design:

1. **Fused bias/activation in Mode 3** -- Mode 3 `run()` does pure matmul with
   spill/reload. Fused operations require Mode 2.
2. **Out-of-order packing** -- Call sites needing `pack_tile<true>` with explicit indices
   use Mode 1.
3. **Untilize packing** -- Call sites needing `pack_untilize_dest` use Mode 1.
4. **Data format reconfig inside accumulate()** -- Call sites that change data formats
   mid-accumulation (FP32_DEST_ACC_EN patterns) handle this themselves.
5. **Multi-MatmulOp orchestration** -- Kernels with multiple matmul operations (B14 has
   W0/W1 matmul then W2 matmul) create separate MatmulOp instances.
