// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

#ifdef ARCH_BLACKHOLE
#include "api/compute/experimental/matmul_custom.h"
#endif

namespace ckernel {

// -------------------------------------------------------------------------
// MoE DM1 state for W2 accumulate helpers
// -------------------------------------------------------------------------

struct MoeDm1State {
    uint32_t step;
    uint32_t tiles_remaining;
    uint32_t buf;     // current buffer index (for cycling)
    uint32_t offset;  // tile offset for current buffer
    uint32_t index;   // current in2 tile index
};

// -------------------------------------------------------------------------
// Configuration struct
// -------------------------------------------------------------------------

struct MatmulOpConfig {
    // --- Required CB IDs ---
    uint32_t in0_cb_id;  // CB for the A matrix (left operand)
    uint32_t in1_cb_id;  // CB for the B matrix (right operand)
    uint32_t out_cb_id;  // CB for the output (or intermediate partials for spill mode)

    // --- Block dimensions (required for block mode, ignored for tile mode) ---
    uint32_t ct_dim = 1;  // Output subblock column dimension in tiles (subblock_w)
    uint32_t rt_dim = 1;  // Output subblock row dimension in tiles (subblock_h)
    uint32_t kt_dim = 1;  // Inner dimension block size in tiles (in0_block_w)

    // --- Transpose ---
    bool transpose = false;  // Transpose B tiles (width-height swap)

    // --- Partials / Spill buffer ---
    // CB for storing partial accumulations when inner dim is blocked (num_blocks_inner > 1).
    // Set to 0 to disable spill/reload.
    uint32_t partials_cb_id = 0;
};

// -------------------------------------------------------------------------
// MatmulOp class template
// -------------------------------------------------------------------------

template <bool IsBlockMode>
class MatmulOp {
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    FORCE_INLINE explicit MatmulOp(const MatmulOpConfig& cfg) : cfg_(cfg) {}

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------

    FORCE_INLINE void init() const {
        if constexpr (IsBlockMode) {
            mm_block_init(
                cfg_.in0_cb_id, cfg_.in1_cb_id, cfg_.out_cb_id, cfg_.transpose, cfg_.ct_dim, cfg_.rt_dim, cfg_.kt_dim);
        } else {
            mm_init(cfg_.in0_cb_id, cfg_.in1_cb_id, cfg_.out_cb_id, cfg_.transpose);
        }
    }

    FORCE_INLINE void init_short() const {
        if constexpr (IsBlockMode) {
            mm_block_init_short(cfg_.in0_cb_id, cfg_.in1_cb_id, cfg_.transpose, cfg_.ct_dim, cfg_.rt_dim, cfg_.kt_dim);
        } else {
            mm_init_short(cfg_.in0_cb_id, cfg_.in1_cb_id, cfg_.transpose);
        }
    }

    FORCE_INLINE void init_short_with_dt(uint32_t old_in1_cb_id) const {
        if constexpr (IsBlockMode) {
            mm_block_init_short_with_dt(
                cfg_.in0_cb_id, cfg_.in1_cb_id, old_in1_cb_id, cfg_.transpose, cfg_.ct_dim, cfg_.rt_dim, cfg_.kt_dim);
        } else {
            mm_init_short_with_dt(cfg_.in0_cb_id, cfg_.in1_cb_id, old_in1_cb_id, cfg_.transpose);
        }
    }

    FORCE_INLINE void init_short_with_both_dt(uint32_t old_in0_cb_id, uint32_t old_in1_cb_id) const {
        static_assert(IsBlockMode, "init_short_with_both_dt is only supported in block mode");
        mm_block_init_short_with_both_dt(
            cfg_.in0_cb_id,
            cfg_.in1_cb_id,
            old_in0_cb_id,
            old_in1_cb_id,
            cfg_.transpose,
            cfg_.ct_dim,
            cfg_.rt_dim,
            cfg_.kt_dim);
    }

    // -------------------------------------------------------------------------
    // Accumulate: the core matmul loop primitive
    // -------------------------------------------------------------------------

    FORCE_INLINE void begin_subblock() const { tile_regs_acquire(); }

    // Generalized accumulate: iterates count times, advancing each index by its stride.
    //   for k in [0, count): matmul(in0_start + k*in0_stride, in1_start + k*in1_stride, dst_start + k*dst_stride)
    // Common patterns:
    //   Inner-dim reduction:  accumulate(a, b, dst, K, 1, stride, 0)  -- dst fixed, in0 walks, in1 strides
    //   Broadcast-in0:        accumulate(0, b, 0, N, 0, 1, 1)        -- in0 fixed, in1/dst walk
    //   Broadcast-in1:        accumulate(0, 0, 0, N, 1, 0, 1)        -- in1 fixed, in0/dst walk
    FORCE_INLINE void accumulate(
        uint32_t in0_start,
        uint32_t in1_start,
        uint32_t dst_start,
        uint32_t count,
        uint32_t in0_stride,
        uint32_t in1_stride,
        uint32_t dst_stride) const {
        for (uint32_t k = 0; k < count; ++k) {
            matmul_single(in0_start + k * in0_stride, in1_start + k * in1_stride, dst_start + k * dst_stride);
        }
    }

    // Tile-mode subblock accumulate: computes an (out_h x out_w) subblock of output tiles.
    // Each output tile at (h, w) accumulates over inner_dim with the standard tile indexing:
    //   in0 = in0_offset + h * inner_dim + k
    //   in1 = in1_offset + k * in1_stride + w
    //   dst = sequential (0, 1, 2, ...)
    FORCE_INLINE void accumulate_tile_subblock(
        uint32_t in0_subblock_offset,
        uint32_t in1_subblock_offset,
        uint32_t out_h,
        uint32_t out_w,
        uint32_t inner_dim,
        uint32_t in1_stride) const {
        uint32_t dst_index = 0;
        for (uint32_t h = 0; h < out_h; ++h) {
            for (uint32_t w = 0; w < out_w; ++w) {
                accumulate(
                    in0_subblock_offset + h * inner_dim,
                    in1_subblock_offset + w,
                    dst_index,
                    inner_dim,
                    1,
                    in1_stride,
                    0);
                ++dst_index;
            }
        }
    }

    FORCE_INLINE void end_to_output(uint32_t dest_cb_id, uint32_t num_tiles) const {
        tile_regs_commit();
        cb_reserve_back(dest_cb_id, num_tiles);
        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles; i++) {
            pack_tile(i, dest_cb_id);
        }
        tile_regs_release();
        cb_push_back(dest_cb_id, num_tiles);
    }

    FORCE_INLINE void end_to_partials(uint32_t num_tiles) const {
        tile_regs_commit();
        cb_reserve_back(cfg_.partials_cb_id, num_tiles);
        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles; i++) {
            pack_tile(i, cfg_.partials_cb_id);
        }
        tile_regs_release();
        cb_push_back(cfg_.partials_cb_id, num_tiles);
    }

    FORCE_INLINE void reload_partials(uint32_t num_tiles) const {
        copy_tile_to_dst_init_short_with_dt(cfg_.in1_cb_id, cfg_.partials_cb_id);
        cb_wait_front(cfg_.partials_cb_id, num_tiles);
        copy_block_matmul_partials(cfg_.partials_cb_id, 0, 0, num_tiles);
        cb_pop_front(cfg_.partials_cb_id, num_tiles);
        // Reconfigure back to matmul mode
        if constexpr (IsBlockMode) {
            mm_block_init_short_with_dt(
                cfg_.in0_cb_id,
                cfg_.in1_cb_id,
                cfg_.partials_cb_id,
                cfg_.transpose,
                cfg_.ct_dim,
                cfg_.rt_dim,
                cfg_.kt_dim);
        } else {
            mm_init_short_with_dt(cfg_.in0_cb_id, cfg_.in1_cb_id, cfg_.partials_cb_id, cfg_.transpose);
        }
    }

    // Combines begin_subblock + optional reload + accumulate + end_to_output/partials.
    // Replaces the standard subblock compute-and-pack pattern found in many kernels.
    FORCE_INLINE void accumulate_and_pack(
        uint32_t in0_index_start,
        uint32_t in1_index_start,
        uint32_t inner_dim,
        uint32_t in1_stride,
        uint32_t dest_cb_id,
        uint32_t num_tiles,
        bool reload = false) const {
        begin_subblock();
        if (reload) {
            reload_partials(num_tiles);
        }
        accumulate(in0_index_start, in1_index_start, 0, inner_dim, 1, in1_stride, 0);
        end_to_output(dest_cb_id, num_tiles);
    }

    // -------------------------------------------------------------------------
    // Single-tile convenience: replaces accumulate(x, y, z, 1, 0, 0, 0)
    // -------------------------------------------------------------------------

    FORCE_INLINE void matmul_one_tile(uint32_t in0_idx, uint32_t in1_idx, uint32_t dst_idx) const {
        matmul_single(in0_idx, in1_idx, dst_idx);
    }

    // -------------------------------------------------------------------------
    // Blackhole no-MOP accumulate (block mode only)
    // Uses matmul_block_no_mop for better performance on Blackhole SDPA.
    // -------------------------------------------------------------------------

#ifdef ARCH_BLACKHOLE
    FORCE_INLINE void accumulate_no_mop(
        uint32_t in0_start,
        uint32_t in1_start,
        uint32_t dst_start,
        uint32_t count,
        uint32_t in0_stride,
        uint32_t in1_stride,
        uint32_t dst_stride) const {
        static_assert(IsBlockMode, "accumulate_no_mop is only supported in block mode");
        for (uint32_t k = 0; k < count; ++k) {
            matmul_block_no_mop(
                cfg_.in0_cb_id,
                cfg_.in1_cb_id,
                in0_start + k * in0_stride,
                in1_start + k * in1_stride,
                dst_start + k * dst_stride,
                cfg_.transpose,
                cfg_.ct_dim,
                cfg_.rt_dim,
                cfg_.kt_dim);
        }
    }
#endif

    // -------------------------------------------------------------------------
    // Reduce-W tiles: per-tile CB wait/pop on in0 with matmul accumulation.
    // Absorbs: for(w){cb_wait(in0,1); matmul(0,0,dst); cb_pop(in0,1);}
    // -------------------------------------------------------------------------

    FORCE_INLINE void reduce_w_tiles(uint32_t count, uint32_t dst_idx) const {
        for (uint32_t w = 0; w < count; ++w) {
            cb_wait_front(cfg_.in0_cb_id, 1);
            matmul_single(0, 0, dst_idx);
            cb_pop_front(cfg_.in0_cb_id, 1);
        }
    }

    // Same as reduce_w_tiles but calls init_short() per tile.
    // For kernels that reconfigure data formats between matmul iterations
    // (e.g., moreh_mean_w, moreh_sum_w). Includes FP32_DEST_ACC_EN reconfig.
    FORCE_INLINE void reduce_w_tiles_with_init(uint32_t count, uint32_t dst_idx) const {
        for (uint32_t w = 0; w < count; ++w) {
            cb_wait_front(cfg_.in0_cb_id, 1);
#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cfg_.in0_cb_id, cfg_.in1_cb_id);
#endif
            init_short();
            matmul_single(0, 0, dst_idx);
            cb_pop_front(cfg_.in0_cb_id, 1);
        }
    }

    // -------------------------------------------------------------------------
    // Transformer attention accumulate: progressive in0 reveal + per-tile in1.
    // Absorbs the inner Kt loop from transformer_attn_matmul.
    // On first call (progressive_in0=true), progressively waits for in0 tiles.
    // Each iteration: wait(in1,1), matmul(kt, 0, 0), pop(in1,1).
    // -------------------------------------------------------------------------

    FORCE_INLINE void accumulate_attn(uint32_t inner_dim, bool progressive_in0) const {
        for (uint32_t kt = 0; kt < inner_dim; ++kt) {
            if (progressive_in0) {
                cb_wait_front(cfg_.in0_cb_id, kt + 1);
            }
            cb_wait_front(cfg_.in1_cb_id, 1);
            matmul_single(kt, 0, 0);
            cb_pop_front(cfg_.in1_cb_id, 1);
        }
    }

    // -------------------------------------------------------------------------
    // Reduce subblock inplace: per-subblock acquire/matmul/commit/pop/pack/release.
    // Absorbs the reduction loop from SDPA compute_common matmul_reduce.
    // Each subblock: acquire → matmul(0,0,0,1) → commit → pop(out,n) → pack(n) → release → push(n).
    // -------------------------------------------------------------------------

    FORCE_INLINE void reduce_subblock_inplace(uint32_t num_subblocks, uint32_t subblock_tiles) const {
        for (uint32_t sub = 0; sub < num_subblocks; ++sub) {
            tile_regs_acquire();
            matmul_single(0, 0, 0);
            tile_regs_commit();
            cb_pop_front(cfg_.out_cb_id, subblock_tiles);
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_tiles; i++) {
                pack_tile(i, cfg_.out_cb_id);
            }
            tile_regs_release();
            cb_push_back(cfg_.out_cb_id, subblock_tiles);
        }
    }

    // -------------------------------------------------------------------------
    // MoE blocked accumulate with bias: iterates over weight blocks,
    // accumulating data tiles with stride, stopping when k_tracker reaches
    // limit, then adding bias via bias_op. Weight CB wait/pop handled internally.
    // Returns updated in0_index.
    // -------------------------------------------------------------------------

    FORCE_INLINE uint32_t moe_accumulate_with_bias(
        MatmulOp& bias_op,
        uint32_t in0_start,
        uint32_t num_blocks,
        uint32_t tiles_per_block,
        uint32_t tile_stride,
        uint32_t limit) const {
        uint32_t k_tracker = 0;
        uint32_t in0_index = in0_start;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_wait_front(cfg_.in1_cb_id, tiles_per_block);
            uint32_t last_k_index = 0;
            for (uint32_t k = 0; k < tiles_per_block; k += tile_stride) {
                if (k_tracker == limit) {
                    last_k_index = k;
                    break;
                }
                matmul_single(in0_index, k, 0);
                in0_index++;
                k_tracker++;
            }
            if (k_tracker == limit) {
                bias_op.matmul_one_tile(0, last_k_index, 0);
            }
            cb_pop_front(cfg_.in1_cb_id, tiles_per_block);
        }
        return in0_index;
    }

    // -------------------------------------------------------------------------
    // MoE W2 blocked accumulate with DM1 buffer cycling and bias.
    // For moe_gpt W2 pattern: weight blocks with dm1 buffer tracking,
    // 6-buffer cycling, and bias addition at k_tracker limit.
    // -------------------------------------------------------------------------

    FORCE_INLINE void moe_w2_accumulate_with_dm1_cycling(
        MatmulOp& bias_op,
        MoeDm1State& dm1,
        uint32_t num_blocks,
        uint32_t tiles_per_block,
        uint32_t tile_stride,
        uint32_t limit,
        uint32_t dm1_rdy_cb,
        uint32_t tiles_per_step,
        uint32_t num_buffers,
        const uint32_t* dm1_table) const {
        uint32_t k_tracker = 0;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_wait_front(cfg_.in1_cb_id, tiles_per_block);
            uint32_t last_k_index = 0;
            for (uint32_t k = 0; k < tiles_per_block; k += tile_stride) {
                if (k_tracker == limit) {
                    last_k_index = k;
                    break;
                }
                if (dm1.tiles_remaining == 0) {
                    cb_pop_front(dm1_rdy_cb, 1);
                    cb_wait_front(dm1_rdy_cb, 1);
                    dm1.tiles_remaining = dm1_table[++dm1.step];
                    dm1.buf = (dm1.buf >= num_buffers - 1) ? 0 : dm1.buf + 1;
                    dm1.offset = dm1.buf * tiles_per_step;
                    dm1.index = dm1.offset;
                }
                dm1.tiles_remaining--;
                matmul_single(dm1.index, k, 0);
                dm1.index++;
                k_tracker++;
            }
            if (k_tracker == limit) {
                bias_op.matmul_one_tile(0, last_k_index, 0);
            }
            cb_pop_front(cfg_.in1_cb_id, tiles_per_block);
        }
    }

    // -------------------------------------------------------------------------
    // MoE W2 blocked accumulate with DM1 linear advance (no bias).
    // For moe_compute W2 pattern: weight blocks with dm1 buffer tracking,
    // linear offset advance, and early exit on last block.
    // -------------------------------------------------------------------------

    FORCE_INLINE void moe_w2_accumulate_with_dm1_linear(
        MoeDm1State& dm1,
        uint32_t num_blocks,
        uint32_t tiles_per_block,
        uint32_t tile_stride,
        uint32_t dm1_rdy_cb,
        uint32_t tiles_per_step,
        const uint32_t* dm1_table,
        uint32_t last_block_early_exit_k) const {
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_wait_front(cfg_.in1_cb_id, tiles_per_block);
            for (uint32_t k = 0; k < tiles_per_block; k += tile_stride) {
                if ((block == (num_blocks - 1)) && (k == last_block_early_exit_k)) {
                    cb_pop_front(dm1_rdy_cb, 1);
                    break;
                }
                if (dm1.tiles_remaining == 0) {
                    cb_pop_front(dm1_rdy_cb, 1);
                    cb_wait_front(dm1_rdy_cb, 1);
                    dm1.tiles_remaining = dm1_table[++dm1.step];
                    dm1.offset += tiles_per_step;
                    dm1.index = dm1.offset;
                }
                dm1.tiles_remaining--;
                matmul_single(dm1.index, k, 0);
                dm1.index++;
            }
            cb_pop_front(cfg_.in1_cb_id, tiles_per_block);
        }
    }

    // Computes one output tile by accumulating over inner_dim single tiles.
    // Per-tile CB wait/pop on both inputs. Tile mode only.
    // Replaces: acquire + for(K){wait/matmul/pop} + commit + reserve/pack/release/push
    FORCE_INLINE void compute_one_tile(uint32_t inner_dim) const {
        tile_regs_acquire();
        for (uint32_t k = 0; k < inner_dim; ++k) {
            cb_wait_front(cfg_.in0_cb_id, 1);
            cb_wait_front(cfg_.in1_cb_id, 1);
            matmul_single(0, 0, 0);
            cb_pop_front(cfg_.in0_cb_id, 1);
            cb_pop_front(cfg_.in1_cb_id, 1);
        }
        tile_regs_commit();
        cb_reserve_back(cfg_.out_cb_id, 1);
        tile_regs_wait();
        pack_tile(0, cfg_.out_cb_id);
        tile_regs_release();
        cb_push_back(cfg_.out_cb_id, 1);
    }

    // Computes a full inner block across subblocks with optional spill/reload.
    // Handles the double subblock loop (in0_num_subblocks x in1_num_subblocks),
    // CB wait/pop for the input block, and spill/reload between inner blocks.
    // Replaces the standard blocked matmul inner loop found in many kernels.
    FORCE_INLINE void compute_inner_block(
        uint32_t in0_num_subblocks,
        uint32_t in1_num_subblocks,
        uint32_t in0_block_num_tiles,
        uint32_t in1_block_num_tiles,
        uint32_t in1_block_w,
        bool enable_reload,
        bool last_out) const {
        static_assert(IsBlockMode, "compute_inner_block is only supported in block mode");
        uint32_t out_subblock_num_tiles = cfg_.ct_dim * cfg_.rt_dim;
        uint32_t in0_subblock_num_tiles = cfg_.rt_dim * cfg_.kt_dim;

        cb_wait_front(cfg_.in0_cb_id, in0_block_num_tiles);
        cb_wait_front(cfg_.in1_cb_id, in1_block_num_tiles);

        uint32_t in0_index_subblock_offset = 0;
        for (uint32_t in0_sub = 0; in0_sub < in0_num_subblocks; in0_sub++) {
            uint32_t in1_index_subblock_offset = 0;
            for (uint32_t in1_sub = 0; in1_sub < in1_num_subblocks; in1_sub++) {
                begin_subblock();
                if (enable_reload) {
                    reload_partials(out_subblock_num_tiles);
                }
                accumulate(in0_index_subblock_offset, in1_index_subblock_offset, 0, cfg_.kt_dim, 1, in1_block_w, 0);
                if (last_out) {
                    end_to_output(cfg_.out_cb_id, out_subblock_num_tiles);
                } else {
                    end_to_partials(out_subblock_num_tiles);
                }
                in1_index_subblock_offset += cfg_.ct_dim;
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

        cb_pop_front(cfg_.in0_cb_id, in0_block_num_tiles);
        cb_pop_front(cfg_.in1_cb_id, in1_block_num_tiles);
    }

    // -------------------------------------------------------------------------
    // Mode 3: Automatic -- full blocked matmul
    // -------------------------------------------------------------------------

    FORCE_INLINE void run(
        uint32_t batch,
        uint32_t num_blocks_h,
        uint32_t num_blocks_w,
        uint32_t num_blocks_inner,
        uint32_t in0_num_subblocks,
        uint32_t in1_num_subblocks,
        uint32_t in0_block_num_tiles,
        uint32_t in1_block_num_tiles,
        uint32_t in1_block_w) const {
        if constexpr (!IsBlockMode) {
            for (uint32_t b = 0; b < batch; b++) {
                for (uint32_t m = 0; m < num_blocks_h; m++) {
                    for (uint32_t n = 0; n < num_blocks_w; n++) {
                        compute_one_tile(num_blocks_inner);
                    }
                }
            }
        } else {
            for (uint32_t b = 0; b < batch; b++) {
                for (uint32_t bh = 0; bh < num_blocks_h; bh++) {
                    for (uint32_t bw = 0; bw < num_blocks_w; bw++) {
                        bool enable_reload = false;
                        for (uint32_t block_inner = 0; block_inner < num_blocks_inner; block_inner++) {
                            bool last_out = (block_inner == num_blocks_inner - 1);
                            compute_inner_block(
                                in0_num_subblocks,
                                in1_num_subblocks,
                                in0_block_num_tiles,
                                in1_block_num_tiles,
                                in1_block_w,
                                enable_reload,
                                last_out);
                            if (num_blocks_inner > 1) {
                                enable_reload = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    FORCE_INLINE const MatmulOpConfig& config() const { return cfg_; }
    FORCE_INLINE uint32_t in0_cb() const { return cfg_.in0_cb_id; }
    FORCE_INLINE uint32_t in1_cb() const { return cfg_.in1_cb_id; }
    FORCE_INLINE uint32_t out_cb() const { return cfg_.out_cb_id; }

private:
    FORCE_INLINE void matmul_single(uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t dst_tile_index) const {
        if constexpr (IsBlockMode) {
            matmul_block(
                cfg_.in0_cb_id,
                cfg_.in1_cb_id,
                in0_tile_index,
                in1_tile_index,
                dst_tile_index,
                cfg_.transpose,
                cfg_.ct_dim,
                cfg_.rt_dim,
                cfg_.kt_dim);
        } else {
            matmul_tiles(cfg_.in0_cb_id, cfg_.in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index);
        }
    }

    MatmulOpConfig cfg_;
};

// -------------------------------------------------------------------------
// Type aliases
// -------------------------------------------------------------------------
using TileMatmulOp = MatmulOp<false>;  // matmul_tiles wrapper
using BlockMatmulOp = MatmulOp<true>;  // matmul_block wrapper

}  // namespace ckernel
