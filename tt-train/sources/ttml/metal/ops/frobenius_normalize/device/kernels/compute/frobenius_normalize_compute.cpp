// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Frobenius normalize compute kernel — all-to-origin reduction
//
// Phase 1:   square + accumulate all input tiles → cb_sq_acc (FP32 in DST)
// Phase 2:   sfpu_reduce → cb_sq_partial (reader writes partial to origin's L1 using 4-byte stride)
// Phase 3:   origin only: sfpu_reduce all partials from cb_recv, sqrt + eps + recip → cb_norm
// Phase 4:   all cores: multiply each tile by 1/norm
//
// BF16 I/O, all intermediates are FP32

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr auto cb_input = tt::CBIndex::c_0;
constexpr auto cb_sq_acc = tt::CBIndex::c_1;
constexpr auto cb_recv = tt::CBIndex::c_3;
constexpr auto cb_norm = tt::CBIndex::c_4;
constexpr auto cb_output = tt::CBIndex::c_5;
constexpr auto cb_sq_partial = tt::CBIndex::c_7;

constexpr uint32_t num_tiles_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);

inline void drain_padding_tiles() {
    constexpr uint32_t padding = (block_size - num_tiles_per_core % block_size) % block_size;
    if constexpr (padding > 0) {
        cb_wait_front(cb_input, padding);
        cb_pop_front(cb_input, padding);
    }
}

inline void sfpu_reduce_sum_fp32(uint32_t dst_idx) {
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(dst_idx);
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(dst_idx);
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t eps_u32 = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t accum_reg = 0;
    constexpr uint32_t work_reg = 1;

    init_sfpu(cb_input, cb_output);
    binary_op_init_common(cb_input, cb_input, cb_output);

    // =========================================================================
    // Phase 1: Square and accumulate all input tiles into one FP32 tile
    // =========================================================================
    {
        if (num_tiles_per_core > 0) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < num_tiles_per_core; ++i) {
                cb_wait_front(cb_input, 1);
                const auto reg = (i == 0) ? accum_reg : work_reg;
                copy_tile_init(cb_input);
                copy_tile(cb_input, 0, reg);
                mul_binary_tile_init();
                mul_binary_tile(reg, reg, reg);
                if (i > 0) {
                    add_binary_tile_init();
                    add_binary_tile(accum_reg, work_reg, accum_reg);
                }
                cb_pop_front(cb_input, 1);
            }
            tile_regs_commit();
            pack_and_push(accum_reg, cb_sq_acc);
        }
    }

    // Drain Pass 1 padding tiles
    drain_padding_tiles();

    // =========================================================================
    // Phase 2: sfpu_reduce → scalar. Pack to cb_sq_partial for reader to write to origin.
    // =========================================================================
    {
        cb_wait_front(cb_sq_acc, 1);

        init_sfpu(cb_sq_acc, cb_sq_partial);

        tile_regs_acquire();
        copy_tile_init(cb_sq_acc);
        copy_tile(cb_sq_acc, 0, 0);
        cb_pop_front(cb_sq_acc, 1);

        sfpu_reduce_sum_fp32(0);

        tile_regs_commit();
        pack_and_push(0, cb_sq_partial);
    }

    // =========================================================================
    // Phase 3 (origin only): sfpu_reduce the reduction tile (all partials summed
    // into one FP32 tile by reader), then sqrt + eps + recip → cb_norm
    // =========================================================================
    {
#ifdef IS_ORIGIN
        {
            // cb_recv has one tile with all partials at positions 0, 4, 8, ...
            // sfpu_reduce sums all the partials
            cb_wait_front(cb_recv, 1);

            tile_regs_acquire();
            copy_tile_init(cb_recv);
            copy_tile(cb_recv, 0, 0);
            cb_pop_front(cb_recv, 1);

            sfpu_reduce_sum_fp32(0);

            sqrt_tile_init();
            sqrt_tile(0);

            binop_with_scalar_tile_init();
            add_unary_tile(0, eps_u32);

            recip_tile_init<false>();
            recip_tile<false>(0);

            tile_regs_commit();
            pack_and_push(0, cb_norm);
        }
#endif
    }

    // =========================================================================
    // Phase 4: Multiply each tile by 1/norm
    // =========================================================================
    {
        cb_wait_front(cb_norm, 1);
        const uint32_t norm_u32 = read_tile_value(cb_norm, 0, 0);
        cb_pop_front(cb_norm, 1);

        init_sfpu(cb_input, cb_output);

        copy_tile_to_dst_init_short_with_dt(cb_output, cb_input);
        binop_with_scalar_tile_init();

        for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_core; tile_idx += block_size) {
            const uint32_t current = std::min(block_size, num_tiles_per_core - tile_idx);
            cb_wait_front(cb_input, current);
            tile_regs_acquire();
            for (uint32_t j = 0; j < current; ++j) {
                copy_tile(cb_input, j, j);
                mul_unary_tile(j, norm_u32);
            }
            tile_regs_commit();
            cb_pop_front(cb_input, current);
            pack_and_push_block(cb_output, current);
        }
        // Drain Pass 2 padding tiles
        drain_padding_tiles();
    }
}
