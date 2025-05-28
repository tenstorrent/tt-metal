// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api/cb_api.h>
#include <compute_kernel_api/pack.h>
#include <compute_kernel_api/reconfig_data_format.h>
#include <compute_kernel_api/reg_api.h>
#include <debug/dprint.h>
#include <hostdevcommon/kernel_structs.h>
#include <tensix.h>

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);  // rows to process in this kernel
constexpr uint32_t block_size = get_compile_time_arg_val(1);         // size of block
constexpr uint32_t Wt = get_compile_time_arg_val(2);
constexpr uint32_t scaler_bits = get_compile_time_arg_val(3);

constexpr auto cb_input = tt::CBIndex::c_0;
constexpr auto cb_mask = tt::CBIndex::c_2;
constexpr auto cb_max_mask = tt::CBIndex::c_3;
constexpr auto cb_reduction_scaler = tt::CBIndex::c_4;
constexpr auto cb_max_value_before_reduction = tt::CBIndex::c_5;
constexpr auto cb_max_value_after_reduction = tt::CBIndex::c_6;
constexpr auto cb_exp_sum_before_reduction = tt::CBIndex::c_7;
constexpr auto cb_exp_sum_after_reduction = tt::CBIndex::c_8;
constexpr auto cb_mat_mul_reduce = tt::CBIndex::c_9;
constexpr auto cb_output = tt::CBIndex::c_10;

constexpr uint32_t onetile = 1U;

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

#ifdef EVERYTHING_FITS_IN_L1

void find_max_value_in_row() {
    const uint32_t max_value_register = 0;
    const uint32_t tile_register = 1U;
    cb_reserve_back(cb_max_value_before_reduction, onetile);
    tile_regs_acquire();
    reconfig_data_format(cb_input, cb_input);
    for (uint32_t col = 0; col < Wt; col += block_size) {
        // try to procces data by blocks to improve performance
        cb_wait_front(cb_input, col + block_size);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            auto working_register = col == 0 ? max_value_register : tile_register;
            copy_tile_init(cb_input);
            copy_tile(cb_input, /* tile_idx */ col + block_idx, /* register_idx */ working_register);

            if constexpr (do_mask_w) {
                if (col + 1 == Wt) {
                    // this is limitation of the function mask_tile
                    // mask tile currently does not work for mask register that is not next to data register
                    const uint32_t mask_register =
                        working_register + 1U;  // mask register should be next to data register

                    // the next 4 lines are important because we overwrite what's in the trash padding.
                    // it's possible that the padding contains a NaN, and operations like NaN + (-inf) = NaN,
                    // instead of the expected -inf. similarly, -inf * 0 = NaN.
                    copy_tile_init(cb_mask);
                    copy_tile(cb_mask, /* tile_idx */ 0, /* register idx */ mask_register);

                    mask_tile_init();
                    mask_tile(working_register, mask_register);  // mask should be next to tile register.

                    copy_tile_init(cb_max_mask);
                    copy_tile(cb_max_mask, /* tile_idx */ 0, /* register idx */ mask_register);

                    add_binary_tile_init();
                    add_binary_tile(working_register, mask_register);
                }
            }

            if (col > 0) {
                max_tile_init();
                max_tile(max_value_register, tile_register);
            }
        }
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_max_value_before_reduction);
    pack_tile(max_value_register, cb_max_value_before_reduction);
    tile_regs_release();
    cb_push_back(cb_max_value_before_reduction, onetile);
}

#else

void find_max_value_in_row() {
    cb_reserve_back(cb_max_value_before_reduction, onetile);
    tile_regs_acquire();
    const uint32_t max_value_register = 0;
    const uint32_t tile_register = 1U;
    for (uint32_t col = 0; col < Wt;) {
        cb_wait_front(cb_input, block_size);  // wait until reader kernel has written block_size tiles to input buffer
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx, ++col) {
            auto working_register = col == 0 ? max_value_register : tile_register;
            copy_tile_init(cb_input);
            copy_tile(cb_input, /* tile_idx */ block_idx, /* register_idx */ working_register);

            if constexpr (do_mask_w) {
                if (col + 1 == Wt) {
                    // this is limitation of the function mask_tile
                    // mask tile currently does not work for mask register that is not next to data register
                    const uint32_t mask_register =
                        working_register + 1U;  // mask register should be next to data register

                    // the next 4 lines are important because we overwrite what's in the trash padding.
                    // it's possible that the padding contains a NaN, and operations like NaN + (-inf) = NaN,
                    // instead of the expected -inf. similarly, -inf * 0 = NaN.
                    copy_tile_init(cb_mask);
                    copy_tile(cb_mask, /* tile_idx */ 0, /* register idx */ mask_register);

                    mask_tile_init();
                    mask_tile(working_register, mask_register);  // mask should be next to tile register.

                    copy_tile_init(cb_max_mask);
                    copy_tile(cb_max_mask, /* tile_idx */ 0, /* register idx */ mask_register);

                    add_binary_tile_init();
                    add_binary_tile(working_register, mask_register);
                }
            }

            if (col > 0) {
                max_tile_init();
                max_tile(max_value_register, tile_register);
            }
        }
        cb_pop_front(cb_input, block_size);  // delete block_size tiles from input buffer
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_max_value_before_reduction);
    pack_tile(max_value_register, cb_max_value_before_reduction);
    tile_regs_release();
    cb_push_back(cb_max_value_before_reduction, onetile);
}

#endif

void reduce_max_value() {
    // reduce max value and push it to cb_max_value_after_reduction
    cb_wait_front(cb_max_value_before_reduction, onetile);
    cb_reserve_back(cb_max_value_after_reduction, onetile);

    const uint32_t reduction_register = 0;
    tile_regs_acquire();
    reconfig_data_format(cb_max_value_before_reduction, cb_reduction_scaler);
    reduce_init_delta<false, PoolType::MAX, ReduceDim::REDUCE_ROW>(
        cb_max_value_before_reduction, cb_reduction_scaler, cb_max_value_after_reduction);
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
        cb_max_value_before_reduction,
        cb_reduction_scaler,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        reduction_register);
    reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_max_value_after_reduction);
    tile_regs_commit();

    tile_regs_wait();

    pack_reconfig_data_format(cb_max_value_after_reduction);
    pack_tile(reduction_register, cb_max_value_after_reduction);
    tile_regs_release();

    cb_push_back(cb_max_value_after_reduction, onetile);   // push tile after reduction
    cb_pop_front(cb_max_value_before_reduction, onetile);  // pop tile before reduction
}

#ifdef EVERYTHING_FITS_IN_L1

// calculate sum of exp(x - max(x))
void calculate_sum_exp_x() {
    cb_wait_front(cb_max_value_after_reduction, onetile);  // wait until we get max value in each row

    // run through all tiles in row
    const uint32_t accum_register = 0;
    const uint32_t tile_register = 1U;

    cb_reserve_back(cb_exp_sum_before_reduction, onetile);
    tile_regs_acquire();

    const uint32_t max_value_register = 3U;
    reconfig_data_format(cb_max_value_after_reduction, cb_input);
    unary_bcast_init<BroadcastType::COL>(cb_max_value_after_reduction, cb_max_value_after_reduction);
    unary_bcast<BroadcastType::COL>(
        cb_max_value_after_reduction, /* tile idx */ 0, /* reg tile idx */ max_value_register);
    for (uint32_t col = 0; col < Wt; ++col) {
        auto working_register = col == 0 ? accum_register : tile_register;

        copy_tile_init(cb_input);
        copy_tile(cb_input, /* tile_idx */ col, /* register_idx */ working_register);

        sub_binary_tile_init();
        sub_binary_tile(working_register, max_value_register);  // subtract max value from each tile

        exp_tile_init<false>();
        exp_tile</* approx */ false>(working_register);  // calculate exp for each tile in tile register

        if constexpr (do_mask_w) {
            if (col + 1 == Wt) {
                // this is limitation of the function mask_tile
                // mask tile currently does not work for mask register that is not next to data register
                const uint32_t mask_register = working_register + 1U;  // mask register should be next to data register
                copy_tile_init(cb_mask);
                copy_tile(cb_mask, /* tile_idx */ 0, /* register idx */ mask_register);

                mask_tile_init();
                mask_tile(working_register, mask_register);  // mask should be next to tile register
            }
        }

        if (col > 0) {
            add_binary_tile_init();
            add_binary_tile(accum_register, working_register);
        }
    }
    tile_regs_commit();

    // packing to cb_exp_sum_before_reduction
    tile_regs_wait();
    pack_reconfig_data_format(cb_exp_sum_before_reduction);
    pack_tile(accum_register, cb_exp_sum_before_reduction);
    tile_regs_release();
    cb_push_back(cb_exp_sum_before_reduction, onetile);
}

#else

// calculate sum of exp(x - max(x))
void calculate_sum_exp_x() {
    // wait until we get max value in each row
    cb_wait_front(cb_max_value_after_reduction, onetile);  // wait until we get max value in each row
    cb_reserve_back(cb_exp_sum_before_reduction, onetile);
    // run through all tiles in row
    const uint32_t accum_register = 0;
    const uint32_t tile_register = 1U;

    tile_regs_acquire();

    const uint32_t max_value_register = 3U;
    reconfig_data_format(cb_max_value_after_reduction, cb_input);
    unary_bcast_init<BroadcastType::COL>(cb_max_value_after_reduction, cb_max_value_after_reduction);
    unary_bcast<BroadcastType::COL>(
        cb_max_value_after_reduction, /* tile idx */ 0, /* reg tile idx */ max_value_register);
    for (uint32_t col = 0; col < Wt;) {
        cb_wait_front(cb_input, block_size);  // wait until reader kernel has written block_size tiles to input buffer
        reconfig_data_format(cb_max_value_after_reduction, cb_input);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx, ++col) {
            auto working_register = col == 0 ? accum_register : tile_register;

            copy_tile_init(cb_input);
            copy_tile(cb_input, /* tile_idx */ block_idx, /* register_idx */ working_register);

            sub_binary_tile_init();
            sub_binary_tile(working_register, max_value_register);  // subtract max value from each tile

            exp_tile_init<false>();
            exp_tile</* approx */ false>(working_register);  // calculate exp for each tile in tile register

            if constexpr (do_mask_w) {
                if (col + 1 == Wt) {
                    // this is limitation of the function mask_tile
                    // mask tile currently does not work for mask register that is not next to data register
                    const uint32_t mask_register =
                        working_register + 1U;  // mask register should be next to data register
                    copy_tile_init(cb_mask);
                    copy_tile(cb_mask, /* tile_idx */ 0, /* register idx */ mask_register);

                    mask_tile_init();
                    mask_tile(working_register, mask_register);  // mask should be next to tile register
                }
            }

            if (col > 0) {
                add_binary_tile_init();
                add_binary_tile(accum_register, working_register);
            }
        }
        cb_pop_front(cb_input, block_size);
    }
    tile_regs_commit();

    // packing to cb_exp_sum_before_reduction
    tile_regs_wait();
    pack_reconfig_data_format(cb_exp_sum_before_reduction);
    pack_tile(accum_register, cb_exp_sum_before_reduction);
    tile_regs_release();
    cb_push_back(cb_exp_sum_before_reduction, onetile);
}

#endif

void reduce_sum_exp_x() {
    // reduce exp sum to cb_exp_sum_after_reduction
    cb_wait_front(cb_exp_sum_before_reduction, onetile);
    cb_reserve_back(cb_exp_sum_after_reduction, onetile);

    // use only for testing
    cb_wait_front(cb_mat_mul_reduce, onetile);

    tile_regs_acquire();
    const uint32_t reduction_register = 0;

    // reconfig_data_format(cb_exp_sum_before_reduction, cb_reduction_scaler);
    // reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(
    //     cb_exp_sum_before_reduction, cb_reduction_scaler, cb_exp_sum_after_reduction);
    // reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
    //     cb_exp_sum_before_reduction,
    //     cb_reduction_scaler,
    //     /* tile_idx */ 0,
    //     /* tile_idx */ 0,
    //     /* reduction_register */ reduction_register);
    // reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_exp_sum_after_reduction);

    // We used matmul_tiles instead of reduce_tile, because reduce_tile causes a loss of precision. The same issue has
    // been observed in moreh’s ops.
    mm_init(cb_exp_sum_before_reduction, cb_mat_mul_reduce, cb_exp_sum_after_reduction, 0);
    matmul_tiles(
        cb_exp_sum_before_reduction, cb_mat_mul_reduce, /* tile_idx */ 0, /* tile_idx */ 0, reduction_register, 0);

    recip_tile_init();
    recip_tile(reduction_register);  // DST[0] = 1/sum(exp(x))

    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_exp_sum_after_reduction);
    pack_tile(/* tile_idx */ reduction_register, cb_exp_sum_after_reduction);
    tile_regs_release();
    cb_push_back(cb_exp_sum_after_reduction, onetile);
    cb_pop_front(cb_exp_sum_before_reduction, onetile);
}

void MAIN {
    if constexpr (do_mask_w) {
        cb_wait_front(cb_mask, onetile);
        cb_wait_front(cb_max_mask, onetile);
    }
    cb_wait_front(cb_reduction_scaler, onetile);

    init_sfpu(cb_input, cb_output);
    binary_op_init_common(cb_input, cb_input, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        find_max_value_in_row();  // find max value in each row
        reduce_max_value();       // reduce max value to cb_max_value_after_reduction

        calculate_sum_exp_x();  // calculate sum of exp(x - max(x))
        reduce_sum_exp_x();     // reduce sum(exp(x - max(x))), take log and push to cb_exp_sum_after_reduction

        cb_wait_front(cb_exp_sum_after_reduction, onetile);  //  wait log(sum(exp(x - max(x)))

        const uint32_t working_register = 0;
        const uint32_t sum_exp_register = block_size;

        for (uint32_t col = 0; col < Wt; col += block_size) {
#ifndef EVERYTHING_FITS_IN_L1
            cb_wait_front(cb_input, block_size);
#endif
            cb_reserve_back(cb_output, block_size);

            tile_regs_acquire();
            reconfig_data_format(cb_exp_sum_after_reduction, cb_exp_sum_after_reduction);
            unary_bcast_init<BroadcastType::COL>(cb_exp_sum_after_reduction, cb_exp_sum_after_reduction);
            unary_bcast<BroadcastType::COL>(
                cb_exp_sum_after_reduction, /* tile idx */ 0, /* reg tile idx */ sum_exp_register);

            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
#ifdef EVERYTHING_FITS_IN_L1
                const uint32_t input_tile_idx = col + block_idx;
#else
                const uint32_t input_tile_idx = block_idx;
#endif

                reconfig_data_format(cb_input, cb_max_value_after_reduction);
                sub_bcast_cols_init_short(cb_input, cb_max_value_after_reduction);
                sub_tiles_bcast<BroadcastType::COL>(
                    cb_input,
                    cb_max_value_after_reduction,
                    /* tile_idx */ input_tile_idx,
                    /* tile_idx */ 0,
                    /* register idx */ block_idx);

                exp_tile_init<false>();
                exp_tile</* approx */ false>(block_idx);  // calculate exp for each tile in tile register

                mul_binary_tile_init();
                mul_binary_tile(block_idx, sum_exp_register);  // multiply by scaler

                binop_with_scalar_tile_init();
                mul_unary_tile(block_idx, scaler_bits);  // multiply by scaler
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_reconfig_data_format(cb_output);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                pack_tile(block_idx, cb_output);
            }
            tile_regs_release();
            cb_push_back(cb_output, block_size);

#ifndef EVERYTHING_FITS_IN_L1
            cb_pop_front(cb_input, block_size);
#endif
        }

        cb_pop_front(cb_max_value_after_reduction, onetile);  // pop tile after reduction
        cb_pop_front(cb_exp_sum_after_reduction, onetile);    // pop tile after reduction

#ifdef EVERYTHING_FITS_IN_L1
        cb_pop_front(cb_input, Wt);  // pop Wt tiles from input buffer
#endif
    }

    // pop scaler and masks
    if constexpr (do_mask_w) {
        cb_pop_front(cb_mask, onetile);
        cb_pop_front(cb_max_mask, onetile);
    }
    cb_pop_front(cb_reduction_scaler, onetile);
}

}  // namespace NAMESPACE
