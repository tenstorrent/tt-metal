// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn pre-pass, route step, compute kernel.
//
// Decides which pooled K blocks each Q chunk keeps. Per unit of work:
//   1. Mean and variance of the pooled K blocks (phases A..D).
//   2. A threshold per Q block from those moments and tau (E..G).
//   3. Score every (Q block, K block) pair and compare with the threshold (H).
//   4. OR the scores over each Q chunk, OR in the constant mask, drop padding blocks (I).
//   5. Chunk mode only: OR over each K chunk to get the final mask (J).
//
// A unit is (head, Q-chunk tile) when UNITS_PER_SHARD = 0, or (head, K shard) when UNITS_PER_SHARD = 1 (gather
// mode only). A Q-chunk tile is a group of 32 consecutive Q chunks (one 32-row tile row of the mask). With
// (head, shard) units a core only has its own shard's K, so the moments are built from partial sums that the reader
// exchanges between the head's cores (phases CL and C), and the mask leaves as row-major uint32 rows (phase I')
// instead of tiles.
//
// Naming: a "block" is 64 tokens, a "tile" is 32 rows or columns, "padded" means rounded up to a multiple of 32.
// SHARD_* counts are per K/V shard (one chip of the sequence-parallel ring), ALL_* counts cover every shard.
//
// Compile-time args (shared with the reader and writer):
//    0 NUM_HEADS               heads on this device
//    1 NUM_SHARDS              sequence shards in the ring
//    2 SHARD_BLOCK_TILES       blocks per shard, padded, / 32
//    3 HEAD_DIM_TILES          head dim / 32
//    4 SHARD_Q_CHUNK_TILES     Q chunks per shard, padded, / 32
//    5 SHARD_K_CHUNK_TILES     K chunks per shard, padded, / 32
//    6 SHARD_Q_CHUNKS          Q chunks per shard
//    7 SHARD_K_CHUNKS          K chunks per shard
//    8 ALL_K_CHUNKS_PADDED     K chunks of all shards, padded
//    9 MODE                    0 chunk, 1 gather, 2 compact
//   10 TAU                     selection threshold in standard deviations (fp32 bits)
//   11 VARIANCE_EPSILON        added to the variance term before the sqrt (fp32 bits)
//   12 NEG_INF                 finite minus infinity (fp32 bits)
//   13 INV_NUM_VALID_BLOCKS    1 / number of non-padding K blocks (fp32 bits)
//   14 BLOCKS_PER_Q_CHUNK      Q-block tile rows per Q-chunk tile = q_chunk / 64
//   15 GRID_X, 16 GRID_Y       core grid
//   17 NUM_UNITS               total units of work
//   18 UNITS_PER_SHARD         unit kind, see above

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
constexpr uint32_t cb_k_pooled = 0, cb_q_pooled = 1, cb_ones = 2, cb_q_select = 3, cb_k_select = 4, cb_const_or = 6,
                   cb_valid_blocks = 7, cb_partial_moments = 8, cb_moments = 9, cb_q_squared = 10,
                   cb_threshold_col = 11, cb_threshold = 12, cb_block_hits = 13, cb_block_mask = 14, cb_mask_out = 16,
                   cb_variance = 20, cb_union_out = 21, cb_gathered_moments = 23, cb_local_moments = 24,
                   cb_shard_select = 29;
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
    constexpr uint32_t SHARD_Q_CHUNK_TILES = get_compile_time_arg_val(4);
    constexpr uint32_t SHARD_K_CHUNK_TILES = get_compile_time_arg_val(5);
    constexpr uint32_t MODE = get_compile_time_arg_val(9);
    constexpr uint32_t TAU = get_compile_time_arg_val(10);
    constexpr uint32_t VARIANCE_EPSILON = get_compile_time_arg_val(11);
    constexpr uint32_t INV_NUM_VALID_BLOCKS = get_compile_time_arg_val(13);
    constexpr uint32_t BLOCKS_PER_Q_CHUNK = get_compile_time_arg_val(14);
    constexpr uint32_t GRID_X = get_compile_time_arg_val(15);
    constexpr uint32_t GRID_Y = get_compile_time_arg_val(16);
    constexpr uint32_t NUM_UNITS = get_compile_time_arg_val(17);
    constexpr uint32_t UNITS_PER_SHARD = get_compile_time_arg_val(18);  // 1: (head, shard); 0: (head, Q-chunk tile)

    constexpr uint32_t ALL_BLOCK_TILES = NUM_SHARDS * SHARD_BLOCK_TILES;
    // Pooled-K block tiles a unit reads: its own shard's, or every shard's.
    constexpr uint32_t K_BLOCK_TILES_PER_UNIT = UNITS_PER_SHARD ? SHARD_BLOCK_TILES : ALL_BLOCK_TILES;
    constexpr uint32_t K_TILES_PER_UNIT = K_BLOCK_TILES_PER_UNIT * HEAD_DIM_TILES;
    constexpr uint32_t Q_TILES_PER_HEAD = SHARD_BLOCK_TILES * HEAD_DIM_TILES;
    constexpr uint32_t MASK_TILES_PER_SHARD = MODE == 0 ? SHARD_K_CHUNK_TILES : SHARD_BLOCK_TILES;
    // DST tiles 0..7 are used: the program needs dst_full_sync_en.
    static_assert(HEAD_DIM_TILES <= 4, "moments use DST 0..3 (sums) and 4..7 (sums of squares)");
    static_assert(SHARD_K_CHUNK_TILES <= 8, "chunk tiles are processed in DST batches of DST_BATCH_TILES");
    static_assert(BLOCKS_PER_Q_CHUNK <= 8, "Q-block tile rows per Q-chunk tile must fit one DST batch");
    static_assert(UNITS_PER_SHARD == 0 || MODE == 1, "(head, shard) units emit only gather-form rows (MODE 1)");
    static_assert(NUM_UNITS == (UNITS_PER_SHARD ? NUM_HEADS * NUM_SHARDS : NUM_HEADS * SHARD_Q_CHUNK_TILES));
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
    CircularBuffer k_pooled(cb_k_pooled), q_pooled(cb_q_pooled), ones(cb_ones), q_select(cb_q_select),
        k_select(cb_k_select), const_or(cb_const_or), valid_blocks(cb_valid_blocks),
        partial_moments(cb_partial_moments), moments(cb_moments), q_squared(cb_q_squared),
        threshold_col(cb_threshold_col), threshold(cb_threshold), block_hits(cb_block_hits), block_mask(cb_block_mask),
        mask_out(cb_mask_out), variance(cb_variance), union_out(cb_union_out), gathered_moments(cb_gathered_moments),
        local_moments(cb_local_moments), shard_select(cb_shard_select);
    constexpr uint32_t GATHER_TILES = (NUM_SHARDS + 3) / 4;  // 4 shards x 8 moment rows per tile

    for (uint32_t unit = unit_start; unit < unit_start + unit_count; ++unit) {
        // One of the two ranges is a single value; the loops are the same either way.
        const uint32_t q_chunk_tile_begin = UNITS_PER_SHARD ? 0 : unit % SHARD_Q_CHUNK_TILES;
        const uint32_t q_chunk_tile_end = UNITS_PER_SHARD ? SHARD_Q_CHUNK_TILES : q_chunk_tile_begin + 1;
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
        q_select.wait_front(SHARD_Q_CHUNK_TILES * SHARD_BLOCK_TILES);
        k_select.wait_front(SHARD_BLOCK_TILES * SHARD_K_CHUNK_TILES);
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

        for (uint32_t q_chunk_tile = q_chunk_tile_begin; q_chunk_tile < q_chunk_tile_end; ++q_chunk_tile) {
            // The Q-block tile rows of this Q-chunk tile, clipped to the shard (0 rows if all padding).
            const uint32_t q_block_tile_begin = min_u(q_chunk_tile * BLOCKS_PER_Q_CHUNK, SHARD_BLOCK_TILES);
            const uint32_t q_block_tile_end = min_u(q_block_tile_begin + BLOCKS_PER_Q_CHUNK, SHARD_BLOCK_TILES);
            const uint32_t num_q_block_tiles = q_block_tile_end - q_block_tile_begin;

            // ---- E: pooled Q squared.
            q_squared.reserve_back(BLOCKS_PER_Q_CHUNK * HEAD_DIM_TILES);
            for (uint32_t batch_first = 0; batch_first < num_q_block_tiles * HEAD_DIM_TILES; batch_first += 8) {
                const uint32_t batch_size = min_u(8, num_q_block_tiles * HEAD_DIM_TILES - batch_first);
                tile_regs_acquire();
                copy_init(cb_q_pooled);
                for (uint32_t i = 0; i < batch_size; ++i) {
                    copy_tile(cb_q_pooled, q_block_tile_begin * HEAD_DIM_TILES + batch_first + i, i);
                }
                mul_binary_tile_init();
                for (uint32_t i = 0; i < batch_size; ++i) {
                    mul_binary_tile(i, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < batch_size; ++i) {
                    pack_tile<true>(i, cb_q_squared, batch_first + i);
                }
                tile_regs_release();
            }
            q_squared.push_back(BLOCKS_PER_Q_CHUNK * HEAD_DIM_TILES);
            q_squared.wait_front(BLOCKS_PER_Q_CHUNK * HEAD_DIM_TILES);

            // ---- F: threshold column per Q block: q.mean + tau * sqrt(q^2.variance + epsilon), two Q tiles at a time.
            threshold_col.reserve_back(BLOCKS_PER_Q_CHUNK);
            for (uint32_t pair_first = 0; pair_first < num_q_block_tiles; pair_first += 2) {
                const uint32_t tiles_in_pair = min_u(2, num_q_block_tiles - pair_first);
                tile_regs_acquire();
                matmul_init(cb_q_pooled, cb_moments, 1);
                for (uint32_t k = 0; k < tiles_in_pair; ++k) {
                    for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                        matmul_tiles(
                            cb_q_pooled,
                            cb_moments,
                            (q_block_tile_begin + pair_first + k) * HEAD_DIM_TILES + dim_tile,
                            dim_tile,
                            2 * k);
                    }
                }
                matmul_init(cb_q_squared, cb_variance, 1);
                for (uint32_t k = 0; k < tiles_in_pair; ++k) {
                    for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                        matmul_tiles(
                            cb_q_squared,
                            cb_variance,
                            (pair_first + k) * HEAD_DIM_TILES + dim_tile,
                            dim_tile,
                            2 * k + 1);
                    }
                }
                binop_with_scalar_tile_init();
                for (uint32_t k = 0; k < tiles_in_pair; ++k) {
                    add_unary_tile(2 * k + 1, VARIANCE_EPSILON);
                }
                sqrt_tile_init();
                for (uint32_t k = 0; k < tiles_in_pair; ++k) {
                    sqrt_tile(2 * k + 1);
                }
                binop_with_scalar_tile_init();
                for (uint32_t k = 0; k < tiles_in_pair; ++k) {
                    mul_unary_tile(2 * k + 1, TAU);
                }
                add_binary_tile_init();
                for (uint32_t k = 0; k < tiles_in_pair; ++k) {
                    add_binary_tile(2 * k, 2 * k + 1, 2 * k);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < tiles_in_pair; ++k) {
                    pack_tile<true>(2 * k, cb_threshold_col, pair_first + k);
                }
                tile_regs_release();
            }
            threshold_col.push_back(BLOCKS_PER_Q_CHUNK);
            threshold_col.wait_front(BLOCKS_PER_Q_CHUNK);

            // ---- G: broadcast the threshold column across all 32 columns.
            threshold.reserve_back(BLOCKS_PER_Q_CHUNK);
            for (uint32_t batch_first = 0; batch_first < num_q_block_tiles; batch_first += DST_BATCH_TILES) {
                const uint32_t batch_size = min_u(DST_BATCH_TILES, num_q_block_tiles - batch_first);
                tile_regs_acquire();
                matmul_init(cb_threshold_col, cb_ones, 0);
                for (uint32_t i = 0; i < batch_size; ++i) {
                    matmul_tiles(cb_threshold_col, cb_ones, batch_first + i, 0, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < batch_size; ++i) {
                    pack_tile<true>(i, cb_threshold, batch_first + i);
                }
                tile_regs_release();
            }
            threshold.push_back(BLOCKS_PER_Q_CHUNK);
            threshold.wait_front(BLOCKS_PER_Q_CHUNK);

            for (uint32_t shard = shard_begin; shard < shard_end; ++shard) {
                const_or.wait_front(SHARD_BLOCK_TILES);
                valid_blocks.wait_front(SHARD_BLOCK_TILES);

                // ---- H: scores against the shard's K blocks, 1 where score > threshold.
                block_hits.reserve_back(BLOCKS_PER_Q_CHUNK * SHARD_BLOCK_TILES);
                for (uint32_t q_block_tile = 0; q_block_tile < num_q_block_tiles; ++q_block_tile) {
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
                                matmul_tiles(
                                    cb_q_pooled,
                                    cb_k_pooled,
                                    (q_block_tile_begin + q_block_tile) * HEAD_DIM_TILES + dim_tile,
                                    k_tile,
                                    col);
                            }
                        }
                        copy_init(cb_threshold);
                        copy_tile(cb_threshold, q_block_tile, SLAB_TILES);
                        for (uint32_t col = 0; col < slab_size; ++col) {
                            sub_binary_tile_init();
                            sub_binary_tile(col, SLAB_TILES, col);
                            gtz_tile_init();
                            gtz_tile(col);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t col = 0; col < slab_size; ++col) {
                            pack_tile<true>(col, cb_block_hits, q_block_tile * SHARD_BLOCK_TILES + slab_first + col);
                        }
                        tile_regs_release();
                    }
                }
                block_hits.push_back(BLOCKS_PER_Q_CHUNK * SHARD_BLOCK_TILES);
                block_hits.wait_front(BLOCKS_PER_Q_CHUNK * SHARD_BLOCK_TILES);

                // ---- I: OR over the Q blocks of each chunk, OR the constant mask, AND the valid-block row.
                block_mask.reserve_back(SHARD_BLOCK_TILES);
                if constexpr (MODE != 0 && !UNITS_PER_SHARD) {
                    mask_out.reserve_back(MASK_TILES_PER_SHARD);
                }
                for (uint32_t slab_first = 0; slab_first < SHARD_BLOCK_TILES; slab_first += SLAB_TILES) {
                    const uint32_t slab_size = min_u(SLAB_TILES, SHARD_BLOCK_TILES - slab_first);
                    tile_regs_acquire();
                    if (num_q_block_tiles > 0) {
                        matmul_init(cb_q_select, cb_block_hits, 0);
                        for (uint32_t col = 0; col < slab_size; ++col) {
                            for (uint32_t q_block_tile = 0; q_block_tile < num_q_block_tiles; ++q_block_tile) {
                                matmul_tiles(
                                    cb_q_select,
                                    cb_block_hits,
                                    q_chunk_tile * SHARD_BLOCK_TILES + q_block_tile_begin + q_block_tile,
                                    q_block_tile * SHARD_BLOCK_TILES + slab_first + col,
                                    col);
                            }
                        }
                    }
                    for (uint32_t col = 0; col < slab_size; ++col) {
                        const uint32_t k_block_tile = slab_first + col;
                        copy_init(cb_const_or);
                        if (num_q_block_tiles > 0) {
                            copy_tile(cb_const_or, k_block_tile, SLAB_TILES);
                            add_binary_tile_init();
                            add_binary_tile(col, SLAB_TILES, col);
                        } else {
                            copy_tile(cb_const_or, k_block_tile, col);
                        }
                        binop_with_scalar_tile_init();
                        add_unary_tile(col, MINUS_HALF_F32);
                        gtz_tile_init();
                        gtz_tile(col);
                        copy_init(cb_valid_blocks);
                        copy_tile(cb_valid_blocks, k_block_tile, SLAB_TILES);  // valid-block row
                        mul_binary_tile_init();
                        mul_binary_tile(col, SLAB_TILES, col);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t col = 0; col < slab_size; ++col) {
                        pack_tile<true>(col, cb_block_mask, slab_first + col);
                        if constexpr (MODE != 0 && !UNITS_PER_SHARD) {
                            pack_tile<true>(col, cb_mask_out, slab_first + col);
                        }
                    }
                    tile_regs_release();
                }
                block_mask.push_back(SHARD_BLOCK_TILES);
                block_mask.wait_front(SHARD_BLOCK_TILES);
                if constexpr (MODE != 0 && !UNITS_PER_SHARD) {
                    mask_out.push_back(MASK_TILES_PER_SHARD);
                }

                // ---- I': (head, shard) units: typecast the block mask to uint32 and untilize it into one row-major
                // page.
                if constexpr (UNITS_PER_SHARD) {
                    constexpr uint32_t UNTILIZE_BLOCK_TILES = (SHARD_BLOCK_TILES <= SLAB_TILES) ? SHARD_BLOCK_TILES : 1;
                    static_assert(
                        SHARD_BLOCK_TILES % UNTILIZE_BLOCK_TILES == 0,
                        "untilize block width must divide the row width");
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
                }

                // ---- J: OR over each K chunk (MODE 0: the mask, MODE 2: the union row).
                if constexpr (MODE != 1) {
                    if constexpr (MODE == 0) {
                        mask_out.reserve_back(MASK_TILES_PER_SHARD);
                    } else {
                        union_out.reserve_back(SHARD_K_CHUNK_TILES);
                    }
                    for (uint32_t batch_first = 0; batch_first < SHARD_K_CHUNK_TILES; batch_first += DST_BATCH_TILES) {
                        const uint32_t batch_size = min_u(DST_BATCH_TILES, SHARD_K_CHUNK_TILES - batch_first);
                        tile_regs_acquire();
                        matmul_init(cb_block_mask, cb_k_select, 0);
                        for (uint32_t i = 0; i < batch_size; ++i) {
                            for (uint32_t k_block_tile = 0; k_block_tile < SHARD_BLOCK_TILES; ++k_block_tile) {
                                matmul_tiles(
                                    cb_block_mask,
                                    cb_k_select,
                                    k_block_tile,
                                    k_block_tile * SHARD_K_CHUNK_TILES + batch_first + i,
                                    i);
                            }
                        }
                        for (uint32_t i = 0; i < batch_size; ++i) {
                            binop_with_scalar_tile_init();
                            add_unary_tile(i, MINUS_HALF_F32);
                            gtz_tile_init();
                            gtz_tile(i);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t i = 0; i < batch_size; ++i) {
                            if constexpr (MODE == 0) {
                                pack_tile<true>(i, cb_mask_out, batch_first + i);
                            } else {
                                pack_tile<true>(i, cb_union_out, batch_first + i);
                            }
                        }
                        tile_regs_release();
                    }
                    if constexpr (MODE == 0) {
                        mask_out.push_back(MASK_TILES_PER_SHARD);
                    } else {
                        union_out.push_back(SHARD_K_CHUNK_TILES);
                    }
                }

                block_hits.pop_front(BLOCKS_PER_Q_CHUNK * SHARD_BLOCK_TILES);
                block_mask.pop_front(SHARD_BLOCK_TILES);
                const_or.pop_front(SHARD_BLOCK_TILES);
                valid_blocks.pop_front(SHARD_BLOCK_TILES);
            }

            q_squared.pop_front(BLOCKS_PER_Q_CHUNK * HEAD_DIM_TILES);
            threshold_col.pop_front(BLOCKS_PER_Q_CHUNK);
            threshold.pop_front(BLOCKS_PER_Q_CHUNK);
        }

        k_pooled.pop_front(K_TILES_PER_UNIT);
        q_pooled.pop_front(Q_TILES_PER_HEAD);
        moments.pop_front(MOMENT_TILES);
        variance.pop_front(HEAD_DIM_TILES);
    }
}
