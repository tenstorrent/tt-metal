// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn pre-pass, route step, compute kernel.
//
// Decides which K blocks each local Q block attends exactly. Per unit of work:
//   1. Mean and variance of the pooled K blocks (phases A..D).
//   2. A threshold per Q block from those moments and tau (E..G).
//   3. Score every (Q block, K block) pair and compare with the threshold (H).
//   4. OR in the always-keep blocks, AND the valid-block tile (I).
// The mask has one row per 64-token Q block and one column per 64-token K block, which is what the ring op's
// resident mode reads.
//
// A unit is (head, Q-block tile) when UNITS_PER_SHARD = 0, or (head, K shard) when UNITS_PER_SHARD = 1. A Q-block
// tile is 32 consecutive Q blocks: one 32-row tile row of the mask. With (head, shard) units a core only has its own
// shard's K, so the moments are built from partial sums that the reader exchanges between the head's cores (phases
// CL and C), and the mask leaves as row-major uint32 rows (phase I') instead of tiles.
//
// Naming: a "block" is 64 tokens, a "tile" is 32 rows or columns, "padded" means rounded up to a multiple of 32.
// SHARD_* counts are per K/V shard (one chip of the sequence-parallel ring), ALL_* counts cover every shard.
//
// Compile-time args (shared with the reader and writer):
//    0 NUM_HEADS               heads on this device
//    1 NUM_SHARDS              sequence shards in the ring
//    2 SHARD_BLOCK_TILES       blocks per shard, padded, / 32
//    3 HEAD_DIM_TILES          head dim / 32
//    4 TAU                     selection threshold in standard deviations (fp32 bits)
//    5 VARIANCE_EPSILON        added to the variance term before the sqrt (fp32 bits)
//    6 INV_NUM_VALID_BLOCKS    1 / number of non-padding K blocks (fp32 bits)
//    7 GRID_X, 8 GRID_Y        core grid
//    9 NUM_UNITS               total units of work
//   10 UNITS_PER_SHARD         unit kind, see above

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_k_pooled = 0, cb_q_pooled = 1, cb_ones = 2, cb_const_or = 6, cb_valid_blocks = 7,
                   cb_partial_moments = 8, cb_moments = 9, cb_q_squared = 10, cb_threshold_col = 11, cb_threshold = 12,
                   cb_block_hits = 13, cb_block_mask = 14, cb_mask_out = 16, cb_variance = 20, cb_gathered_moments = 23,
                   cb_local_moments = 24, cb_shard_select = 29;
// Moment tiles: HEAD_DIM_TILES sums (or means) in tiles 0..3, HEAD_DIM_TILES sums of squares (or E[x^2]) in 4..7.
constexpr uint32_t PARTIAL_MOMENT_TILES = 8;
constexpr uint32_t MOMENT_TILES = 8;
constexpr uint32_t SHARD_SELECT_TILES = 8;
constexpr uint32_t MINUS_HALF_F32 = 0xBF000000u;  // -0.5f
// Unary SFPU ops only work on DST tiles 0..3, so they run on slabs of SLAB_TILES tiles with tile SLAB_TILES as scratch.
constexpr uint32_t SLAB_TILES = 3;       // DST tiles 0..2: a column slab, 3: scratch
constexpr uint32_t DST_BATCH_TILES = 4;  // tiles per DST batch
inline uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }
}  // namespace

void kernel_main() {
    constexpr uint32_t NUM_HEADS = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_SHARDS = get_compile_time_arg_val(1);
    constexpr uint32_t SHARD_BLOCK_TILES = get_compile_time_arg_val(2);
    constexpr uint32_t HEAD_DIM_TILES = get_compile_time_arg_val(3);
    constexpr uint32_t TAU = get_compile_time_arg_val(4);
    constexpr uint32_t VARIANCE_EPSILON = get_compile_time_arg_val(5);
    constexpr uint32_t INV_NUM_VALID_BLOCKS = get_compile_time_arg_val(6);
    constexpr uint32_t GRID_X = get_compile_time_arg_val(7);
    constexpr uint32_t GRID_Y = get_compile_time_arg_val(8);
    constexpr uint32_t NUM_UNITS = get_compile_time_arg_val(9);
    constexpr uint32_t UNITS_PER_SHARD = get_compile_time_arg_val(10);  // 1: (head, shard); 0: (head, Q-block tile)

    constexpr uint32_t ALL_BLOCK_TILES = NUM_SHARDS * SHARD_BLOCK_TILES;
    // Pooled-K block tiles a unit reads: its own shard's, or every shard's.
    constexpr uint32_t K_BLOCK_TILES_PER_UNIT = UNITS_PER_SHARD ? SHARD_BLOCK_TILES : ALL_BLOCK_TILES;
    constexpr uint32_t K_TILES_PER_UNIT = K_BLOCK_TILES_PER_UNIT * HEAD_DIM_TILES;
    constexpr uint32_t Q_TILES_PER_HEAD = SHARD_BLOCK_TILES * HEAD_DIM_TILES;
    // DST tiles 0..7 are used: the program needs dst_full_sync_en.
    static_assert(HEAD_DIM_TILES <= 4, "moments use DST 0..3 (sums) and 4..7 (sums of squares)");
    static_assert(NUM_UNITS == (UNITS_PER_SHARD ? NUM_HEADS * NUM_SHARDS : NUM_HEADS * SHARD_BLOCK_TILES));
    // This core's units, [core_index * NUM_UNITS / NUM_CORES, (core_index + 1) * NUM_UNITS / NUM_CORES).
    constexpr uint32_t NUM_CORES = GRID_X * GRID_Y;
    const uint32_t core_index = static_cast<uint32_t>(get_absolute_logical_y()) * GRID_X + get_absolute_logical_x();
    const uint32_t unit_start = core_index * NUM_UNITS / NUM_CORES;
    const uint32_t unit_count = (core_index + 1) * NUM_UNITS / NUM_CORES - unit_start;
    // The output CB sets the packer format for the whole kernel: it must be a bf16 tile CB.
    compute_kernel_hw_startup(cb_q_pooled, cb_k_pooled, cb_partial_moments);
    if (unit_count == 0) {
        return;
    }
    CircularBuffer k_pooled(cb_k_pooled), q_pooled(cb_q_pooled), ones(cb_ones), const_or(cb_const_or),
        valid_blocks(cb_valid_blocks), partial_moments(cb_partial_moments), moments(cb_moments),
        q_squared(cb_q_squared), threshold_col(cb_threshold_col), threshold(cb_threshold), block_hits(cb_block_hits),
        block_mask(cb_block_mask), mask_out(cb_mask_out), variance(cb_variance), gathered_moments(cb_gathered_moments),
        local_moments(cb_local_moments), shard_select(cb_shard_select);
    constexpr uint32_t GATHER_TILES = (NUM_SHARDS + 3) / 4;  // 4 shards x 8 moment rows per tile

    for (uint32_t unit = unit_start; unit < unit_start + unit_count; ++unit) {
        // One of the two ranges is a single value; the loops are the same either way.
        const uint32_t q_block_tile_begin = UNITS_PER_SHARD ? 0 : unit % SHARD_BLOCK_TILES;
        const uint32_t q_block_tile_end = UNITS_PER_SHARD ? SHARD_BLOCK_TILES : q_block_tile_begin + 1;
        const uint32_t shard_begin = UNITS_PER_SHARD ? unit % NUM_SHARDS : 0;
        const uint32_t shard_end = UNITS_PER_SHARD ? shard_begin + 1 : NUM_SHARDS;
        // ---- A/B: per-tile sums and sums of squares of the pooled K.
        partial_moments.reserve_back(PARTIAL_MOMENT_TILES);
        tile_regs_acquire();
        copy_init(cb_k_pooled);
        add_binary_tile_init();
        for (uint32_t k_block_tile = 0; k_block_tile < K_BLOCK_TILES_PER_UNIT; ++k_block_tile) {
            k_pooled.wait_front((k_block_tile + 1) * HEAD_DIM_TILES);
            for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                if (k_block_tile == 0) {
                    copy_tile(cb_k_pooled, dim_tile, dim_tile);
                } else {
                    copy_tile(cb_k_pooled, k_block_tile * HEAD_DIM_TILES + dim_tile, 4);
                    add_binary_tile(dim_tile, 4, dim_tile);
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
            pack_tile<true>(dim_tile, cb_partial_moments, dim_tile);
        }
        tile_regs_release();

        // Squares: copy + SFPU multiply.
        tile_regs_acquire();
        copy_init(cb_k_pooled);
        for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
            copy_tile(cb_k_pooled, dim_tile, dim_tile);
        }
        mul_binary_tile_init();
        for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
            mul_binary_tile(dim_tile, dim_tile, dim_tile);
        }
        for (uint32_t k_block_tile = 1; k_block_tile < K_BLOCK_TILES_PER_UNIT; ++k_block_tile) {
            for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                copy_tile(cb_k_pooled, k_block_tile * HEAD_DIM_TILES + dim_tile, 4);
                mul_binary_tile_init();
                mul_binary_tile(4, 4, 4);
                add_binary_tile_init();
                add_binary_tile(dim_tile, 4, dim_tile);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
            pack_tile<true>(dim_tile, cb_partial_moments, 4 + dim_tile);
        }
        tile_regs_release();
        partial_moments.push_back(PARTIAL_MOMENT_TILES);
        partial_moments.wait_front(PARTIAL_MOMENT_TILES);

        // Per-core constants, read once by the reader.
        ones.wait_front(1);
        if constexpr (UNITS_PER_SHARD) {
            shard_select.wait_front(SHARD_SELECT_TILES);
        }

        // ---- CL: (head, shard) units: column sums of this shard's partial tiles, for the reader to exchange.
        if constexpr (UNITS_PER_SHARD) {
            local_moments.reserve_back(PARTIAL_MOMENT_TILES);
            for (uint32_t batch_first = 0; batch_first < PARTIAL_MOMENT_TILES; batch_first += DST_BATCH_TILES) {
                tile_regs_acquire();
                matmul_init(cb_ones, cb_partial_moments, 0);
                for (uint32_t i = 0; i < DST_BATCH_TILES; ++i) {
                    matmul_tiles(cb_ones, cb_partial_moments, 0, batch_first + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < DST_BATCH_TILES; ++i) {
                    pack_tile<true>(i, cb_local_moments, batch_first + i);
                }
                tile_regs_release();
            }
            local_moments.push_back(PARTIAL_MOMENT_TILES);
            partial_moments.pop_front(PARTIAL_MOMENT_TILES);
            gathered_moments.wait_front(GATHER_TILES);  // every shard's column sums
        }

        // ---- C: column sums of the partial tiles, scaled by 1/num_valid_blocks -> mean and E[x^2].
        moments.reserve_back(MOMENT_TILES);
        for (uint32_t batch_first = 0; batch_first < MOMENT_TILES; batch_first += DST_BATCH_TILES) {
            tile_regs_acquire();
            if constexpr (UNITS_PER_SHARD) {  // sum over the gather tiles with the shard selectors
                matmul_init(cb_shard_select, cb_gathered_moments, 0);
                for (uint32_t i = 0; i < DST_BATCH_TILES; ++i) {
                    for (uint32_t gather_tile = 0; gather_tile < GATHER_TILES; ++gather_tile) {
                        matmul_tiles(cb_shard_select, cb_gathered_moments, batch_first + i, gather_tile, i);
                    }
                }
            } else {
                matmul_init(cb_ones, cb_partial_moments, 0);
                for (uint32_t i = 0; i < DST_BATCH_TILES; ++i) {
                    matmul_tiles(cb_ones, cb_partial_moments, 0, batch_first + i, i);
                }
            }
            binop_with_scalar_tile_init();
            for (uint32_t i = 0; i < DST_BATCH_TILES; ++i) {
                mul_unary_tile(i, INV_NUM_VALID_BLOCKS);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < DST_BATCH_TILES; ++i) {
                pack_tile<true>(i, cb_moments, batch_first + i);
            }
            tile_regs_release();
        }
        moments.push_back(MOMENT_TILES);
        moments.wait_front(MOMENT_TILES);
        if constexpr (UNITS_PER_SHARD) {
            gathered_moments.pop_front(GATHER_TILES);
        } else {
            partial_moments.pop_front(PARTIAL_MOMENT_TILES);
        }

        // ---- D: variance = relu(E[x^2] - mean^2), two head-dim tiles at a time.
        variance.reserve_back(HEAD_DIM_TILES);
        for (uint32_t dim_first = 0; dim_first < HEAD_DIM_TILES; dim_first += 2) {
            const uint32_t dims_in_pair = min_u(2, HEAD_DIM_TILES - dim_first);
            tile_regs_acquire();
            copy_init(cb_moments);
            for (uint32_t k = 0; k < dims_in_pair; ++k) {
                copy_tile(cb_moments, dim_first + k, 2 * k);          // mean
                copy_tile(cb_moments, 4 + dim_first + k, 2 * k + 1);  // E[x^2]
            }
            mul_binary_tile_init();
            for (uint32_t k = 0; k < dims_in_pair; ++k) {
                mul_binary_tile(2 * k, 2 * k, 2 * k);
            }
            sub_binary_tile_init();
            for (uint32_t k = 0; k < dims_in_pair; ++k) {
                sub_binary_tile(2 * k + 1, 2 * k, 2 * k + 1);
            }
            relu_tile_init();
            for (uint32_t k = 0; k < dims_in_pair; ++k) {
                relu_tile(2 * k + 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < dims_in_pair; ++k) {
                pack_tile<true>(2 * k + 1, cb_variance, dim_first + k);
            }
            tile_regs_release();
        }
        variance.push_back(HEAD_DIM_TILES);
        variance.wait_front(HEAD_DIM_TILES);
        q_pooled.wait_front(Q_TILES_PER_HEAD);  // Q is first needed in E

        for (uint32_t q_block_tile = q_block_tile_begin; q_block_tile < q_block_tile_end; ++q_block_tile) {
            const uint32_t first_q_tile = q_block_tile * HEAD_DIM_TILES;  // this tile row's pooled Q tiles

            // ---- E: pooled Q squared.
            q_squared.reserve_back(HEAD_DIM_TILES);
            tile_regs_acquire();
            copy_init(cb_q_pooled);
            for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                copy_tile(cb_q_pooled, first_q_tile + dim_tile, dim_tile);
            }
            mul_binary_tile_init();
            for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                mul_binary_tile(dim_tile, dim_tile, dim_tile);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                pack_tile<true>(dim_tile, cb_q_squared, dim_tile);
            }
            tile_regs_release();
            q_squared.push_back(HEAD_DIM_TILES);
            q_squared.wait_front(HEAD_DIM_TILES);

            // ---- F: threshold column per Q block: q.mean + tau * sqrt(q^2.variance + epsilon).
            threshold_col.reserve_back(1);
            tile_regs_acquire();
            matmul_init(cb_q_pooled, cb_moments, 1);
            for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                matmul_tiles(cb_q_pooled, cb_moments, first_q_tile + dim_tile, dim_tile, 0);
            }
            matmul_init(cb_q_squared, cb_variance, 1);
            for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                matmul_tiles(cb_q_squared, cb_variance, dim_tile, dim_tile, 1);
            }
            binop_with_scalar_tile_init();
            add_unary_tile(1, VARIANCE_EPSILON);
            sqrt_tile_init();
            sqrt_tile(1);
            binop_with_scalar_tile_init();
            mul_unary_tile(1, TAU);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_threshold_col, 0);
            tile_regs_release();
            threshold_col.push_back(1);
            threshold_col.wait_front(1);

            // ---- G: copy the threshold column across all 32 columns.
            threshold.reserve_back(1);
            tile_regs_acquire();
            matmul_init(cb_threshold_col, cb_ones, 0);
            matmul_tiles(cb_threshold_col, cb_ones, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_threshold, 0);
            tile_regs_release();
            threshold.push_back(1);
            threshold.wait_front(1);

            for (uint32_t shard = shard_begin; shard < shard_end; ++shard) {
                const_or.wait_front(SHARD_BLOCK_TILES);
                valid_blocks.wait_front(SHARD_BLOCK_TILES);

                // ---- H: scores against the shard's K blocks, 1 where score > threshold.
                block_hits.reserve_back(SHARD_BLOCK_TILES);
                for (uint32_t slab_first = 0; slab_first < SHARD_BLOCK_TILES; slab_first += SLAB_TILES) {
                    const uint32_t slab_size = min_u(SLAB_TILES, SHARD_BLOCK_TILES - slab_first);
                    tile_regs_acquire();
                    matmul_init(cb_q_pooled, cb_k_pooled, 1);
                    for (uint32_t col = 0; col < slab_size; ++col) {
                        for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                            const uint32_t k_tile =
                                ((UNITS_PER_SHARD ? 0 : shard * SHARD_BLOCK_TILES) + slab_first + col) *
                                    HEAD_DIM_TILES +
                                dim_tile;
                            matmul_tiles(cb_q_pooled, cb_k_pooled, first_q_tile + dim_tile, k_tile, col);
                        }
                    }
                    copy_init(cb_threshold);
                    copy_tile(cb_threshold, 0, SLAB_TILES);
                    for (uint32_t col = 0; col < slab_size; ++col) {
                        sub_binary_tile_init();
                        sub_binary_tile(col, SLAB_TILES, col);
                        gtz_tile_init();
                        gtz_tile(col);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t col = 0; col < slab_size; ++col) {
                        pack_tile<true>(col, cb_block_hits, slab_first + col);
                    }
                    tile_regs_release();
                }
                block_hits.push_back(SHARD_BLOCK_TILES);
                block_hits.wait_front(SHARD_BLOCK_TILES);

                // ---- I: OR the always-keep tile, AND the valid-block tile (full K blocks, real Q blocks).
                // (head, shard) units go on to phase I' through cb_block_mask; the others send tiles to the writer.
                constexpr uint32_t cb_result = UNITS_PER_SHARD ? cb_block_mask : cb_mask_out;
                CircularBuffer result(cb_result);
                result.reserve_back(SHARD_BLOCK_TILES);
                for (uint32_t slab_first = 0; slab_first < SHARD_BLOCK_TILES; slab_first += SLAB_TILES) {
                    const uint32_t slab_size = min_u(SLAB_TILES, SHARD_BLOCK_TILES - slab_first);
                    tile_regs_acquire();
                    for (uint32_t col = 0; col < slab_size; ++col) {
                        const uint32_t k_block_tile = slab_first + col;
                        copy_init(cb_block_hits);
                        copy_tile(cb_block_hits, k_block_tile, col);
                        copy_init(cb_const_or);
                        copy_tile(cb_const_or, k_block_tile, SLAB_TILES);
                        add_binary_tile_init();
                        add_binary_tile(col, SLAB_TILES, col);
                        binop_with_scalar_tile_init();
                        add_unary_tile(col, MINUS_HALF_F32);
                        gtz_tile_init();
                        gtz_tile(col);
                        copy_init(cb_valid_blocks);
                        copy_tile(cb_valid_blocks, k_block_tile, SLAB_TILES);
                        mul_binary_tile_init();
                        mul_binary_tile(col, SLAB_TILES, col);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t col = 0; col < slab_size; ++col) {
                        pack_tile<true>(col, cb_result, slab_first + col);
                    }
                    tile_regs_release();
                }
                result.push_back(SHARD_BLOCK_TILES);

                // ---- I': (head, shard) units: typecast the block mask to uint32 and untilize it into one row-major
                // page.
                if constexpr (UNITS_PER_SHARD) {
                    constexpr uint32_t UNTILIZE_BLOCK_TILES = (SHARD_BLOCK_TILES <= SLAB_TILES) ? SHARD_BLOCK_TILES : 1;
                    static_assert(
                        SHARD_BLOCK_TILES % UNTILIZE_BLOCK_TILES == 0,
                        "untilize block width must divide the row width");
                    block_mask.wait_front(SHARD_BLOCK_TILES);
                    mask_out.reserve_back(1);
                    pack_untilize_dest_init<UNTILIZE_BLOCK_TILES, SHARD_BLOCK_TILES>(cb_mask_out);
                    for (uint32_t block_first = 0; block_first < SHARD_BLOCK_TILES;
                         block_first += UNTILIZE_BLOCK_TILES) {
                        tile_regs_acquire();
                        copy_init(cb_block_mask);
                        for (uint32_t col = 0; col < UNTILIZE_BLOCK_TILES; ++col) {
                            copy_tile(cb_block_mask, block_first + col, col);
                        }
                        typecast_tile_init<
                            static_cast<uint32_t>(DataFormat::Float32),
                            static_cast<uint32_t>(DataFormat::UInt32)>();
                        for (uint32_t col = 0; col < UNTILIZE_BLOCK_TILES; ++col) {
                            typecast_tile<
                                static_cast<uint32_t>(DataFormat::Float32),
                                static_cast<uint32_t>(DataFormat::UInt32)>(col);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_untilize_dest<UNTILIZE_BLOCK_TILES, SHARD_BLOCK_TILES>(
                            cb_mask_out, 1, block_first / UNTILIZE_BLOCK_TILES);
                        tile_regs_release();
                    }
                    pack_untilize_uninit(cb_block_mask);  // back to tilized bf16 packing
                    mask_out.push_back(1);
                    block_mask.pop_front(SHARD_BLOCK_TILES);
                }

                block_hits.pop_front(SHARD_BLOCK_TILES);
                const_or.pop_front(SHARD_BLOCK_TILES);
                valid_blocks.pop_front(SHARD_BLOCK_TILES);
            }

            q_squared.pop_front(HEAD_DIM_TILES);
            threshold_col.pop_front(1);
            threshold.pop_front(1);
        }

        k_pooled.pop_front(K_TILES_PER_UNIT);
        q_pooled.pop_front(Q_TILES_PER_HEAD);
        moments.pop_front(MOMENT_TILES);
        variance.pop_front(HEAD_DIM_TILES);
    }
}
