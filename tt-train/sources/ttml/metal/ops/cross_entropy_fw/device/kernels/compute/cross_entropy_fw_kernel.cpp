// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api/cb_api.h>
#include <compute_kernel_api/pack.h>
#include <compute_kernel_api/reg_api.h>
#include <debug/dprint.h>
#include <tensix.h>

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);  // rows to process in this kernel
constexpr uint32_t block_size = get_compile_time_arg_val(1);         // size of block
constexpr uint32_t Wt = get_compile_time_arg_val(2);                 // number of tiles in inner dimension

constexpr auto cb_input = tt::CBIndex::c_0;
constexpr auto cb_target = tt::CBIndex::c_1;
constexpr auto cb_mask = tt::CBIndex::c_2;
constexpr auto cb_max_mask = tt::CBIndex::c_3;
constexpr auto cb_scaler = tt::CBIndex::c_4;  // used to reduction
constexpr auto cb_max_value_before_reduction = tt::CBIndex::c_5;
constexpr auto cb_max_value_after_reduction = tt::CBIndex::c_6;
constexpr auto cb_exp_sum_before_reduction = tt::CBIndex::c_7;
constexpr auto cb_exp_sum_after_reduction = tt::CBIndex::c_8;
constexpr auto cb_output_before_reduction = tt::CBIndex::c_9;
constexpr auto cb_output = tt::CBIndex::c_10;

constexpr uint32_t onetile = 1;
#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

void find_max_value_in_row() {
    const uint32_t max_value_register = 0;
    const uint32_t tile_register = 1U;
    cb_wait_front(cb_input, Wt);  // wait until reader kernel has written Wt tiles to input buffer
    cb_reserve_back(cb_max_value_before_reduction, onetile);
    tile_regs_acquire();
    for (uint32_t col = 0; col < Wt; ++col) {
        auto working_register = col == 0 ? max_value_register : tile_register;
        copy_tile_init(cb_input);
        copy_tile(cb_input, /* tile_idx */ col, /* register_idx */ working_register);

        if constexpr (do_mask_w) {
            if (col + 1 == Wt) {
                // this is limitation of the function mask_tile
                // mask tile currently does not work for mask register that is not next to data register
                const uint32_t mask_register = working_register + 1U;  // mask register should be next to data register
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
    // cb_pop_front(cb_input, Wt);
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_max_value_before_reduction);
    pack_tile(max_value_register, cb_max_value_before_reduction);
    tile_regs_release();
    cb_push_back(cb_max_value_before_reduction, onetile);

    // reduce max value and push it to cb_max_value_after_reduction
    cb_wait_front(cb_max_value_before_reduction, onetile);
    cb_reserve_back(cb_max_value_after_reduction, onetile);

    const uint32_t reduction_register = 0;
    tile_regs_acquire();
    reconfig_data_format(cb_max_value_before_reduction, cb_scaler);
    reduce_init_delta<false, PoolType::MAX, ReduceDim::REDUCE_ROW>(
        cb_max_value_before_reduction, cb_scaler, cb_max_value_after_reduction);
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
        cb_max_value_before_reduction,
        cb_scaler,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        reduction_register);
    reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_max_value_before_reduction);
    tile_regs_commit();

    tile_regs_wait();

    pack_reconfig_data_format(cb_max_value_after_reduction);
    pack_tile(reduction_register, cb_max_value_after_reduction);
    tile_regs_release();

    cb_push_back(cb_max_value_after_reduction, onetile);   // push tile after reduction
    cb_pop_front(cb_max_value_before_reduction, onetile);  // pop tile before reduction
}

// calculate sum of exp(x - max(x))
void calculate_sum_exp_x() {
    // need to read from input second time
    // cb_wait_front(cb_input, Wt);  // wait until reader kernel has written Wt tiles to input buffer
    cb_wait_front(cb_max_value_after_reduction, onetile);  // wait until we get max value in each row

    // run through all tiles in row
    const uint32_t accum_register = 0;
    const uint32_t tile_register = 1U;

    cb_reserve_back(cb_exp_sum_before_reduction, onetile);  // create extra buffer
    tile_regs_acquire();
    for (uint32_t col = 0; col < Wt; ++col) {
        auto working_register = col == 0 ? accum_register : tile_register;

        // subtract max value from each tile
        sub_bcast_cols_init_short(cb_input, cb_max_value_after_reduction);
        sub_tiles_bcast_cols(
            cb_input,
            cb_max_value_after_reduction,
            /* tile idx */ col,
            /* tile idx */ 0,
            /* reg tile idx */ working_register);

        exp_tile_init();
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

    // cb_pop_front(cb_input, Wt);  // delete Wt tiles from input buffer

    // exp(x - max(x))
    // reduce exp sum to cb_exp_sum_after_reduction
    cb_wait_front(cb_exp_sum_before_reduction, onetile);
    cb_reserve_back(cb_exp_sum_after_reduction, onetile);

    tile_regs_acquire();
    const uint32_t reduction_register = 0;
    reconfig_data_format(cb_exp_sum_before_reduction, cb_scaler);
    reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(
        cb_exp_sum_before_reduction, cb_scaler, cb_exp_sum_after_reduction);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
        cb_exp_sum_before_reduction,
        cb_scaler,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* reduction_register */ reduction_register);
    reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_exp_sum_before_reduction);

    log_tile_init();
    log_tile(reduction_register);

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
    cb_wait_front(cb_scaler, onetile);

    init_sfpu(cb_input, cb_output);
    binary_op_init_common(cb_input, cb_target, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        find_max_value_in_row();  // find max value in each row
        calculate_sum_exp_x();    // calculate sum of exp(x - max(x))
        //  set (input +  sum of exp(x - max())) to output
        {
            // need to read input third time
            // cb_wait_front(cb_input, Wt);   // wait until reader kernel has written Wt tiles to input buffer
            cb_wait_front(cb_target, Wt);  // wait until reader kernel has written Wt tiles to target buffer
            cb_wait_front(cb_exp_sum_after_reduction, onetile);  // <- get here log(sum(exp(x - max(x)))

            cb_reserve_back(
                cb_output, onetile);  // reserve Wt tiles in output buffer == wait until cb will has Wt tiles
            // for (uint32_t col = 0; col < Wt; col += block_size) {
            const uint32_t accum_register = 0;
            tile_regs_acquire();
            reconfig_data_format(cb_input, cb_max_value_after_reduction);
            // accumulate log(sum(exp(x - max(x))) over row
            for (uint32_t col = 0; col < Wt; col++) {
                auto working_register = col == 0 ? 0 : 1U;
                auto target_register = working_register + 1U;
                auto log_sum_exp_register = working_register + 2U;
                // cb_reserve_back(cb_output, block_size);  // wait until output cb will have block_size free tiles
                sub_bcast_cols_init_short(cb_input, cb_max_value_after_reduction);
                sub_tiles_bcast_cols(
                    cb_input,
                    cb_max_value_after_reduction,
                    /* tile idx */ col,
                    /* tile idx */ 0,
                    /* reg tile idx */ working_register);

                mul_bcast_cols_init_short(cb_target, cb_exp_sum_after_reduction);
                mul_tiles_bcast_cols(
                    cb_target,
                    cb_exp_sum_after_reduction,
                    /* tile idx */ col,
                    /* tile idx */ 0,
                    /* reg tile idx */ log_sum_exp_register);

                copy_tile_init(cb_target);
                copy_tile(cb_target, /* tile_idx */ col, /* register_idx */ target_register);
                mul_binary_tile_init();
                mul_binary_tile(working_register, target_register);  // choose (x - max(x)) by index
                negative_tile_init();
                negative_tile(working_register);
                add_binary_tile_init();
                add_binary_tile(working_register, log_sum_exp_register);

                if (col > 0) {
                    add_binary_tile_init();
                    add_binary_tile(accum_register, working_register);
                }
            }
            tile_regs_commit();

            tile_regs_wait();
            cb_reserve_back(cb_output_before_reduction, onetile);
            pack_reconfig_data_format(cb_output_before_reduction);
            pack_tile(accum_register, cb_output_before_reduction);
            tile_regs_release();
            cb_push_back(cb_output_before_reduction, onetile);

            cb_wait_front(cb_output_before_reduction, onetile);
            const uint32_t reduction_register = 0;
            tile_regs_acquire();
            reconfig_data_format(cb_output_before_reduction, cb_scaler);
            reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(
                cb_output_before_reduction, cb_scaler, cb_output);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                cb_output_before_reduction,
                cb_scaler,
                /* tile_idx */ 0,
                /* tile_idx */ 0,
                /* reduction_register */ reduction_register);
            reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_output_before_reduction);
            tile_regs_commit();

            tile_regs_wait();
            pack_reconfig_data_format(cb_output);
            pack_tile(reduction_register, cb_output);
            tile_regs_release();
            cb_push_back(cb_output, onetile);

            cb_pop_front(cb_max_value_after_reduction, onetile);  // pop tile after reduction
            cb_pop_front(cb_exp_sum_after_reduction, onetile);    // pop tile after reduction
            cb_pop_front(cb_output_before_reduction, onetile);    // pop tile before reduction
            cb_pop_front(cb_input, Wt);                           // pop Wt tiles from input buffer
            cb_pop_front(cb_target, Wt);                          // pop Wt tiles from target buffer
        }
    }
}

}  // namespace NAMESPACE
