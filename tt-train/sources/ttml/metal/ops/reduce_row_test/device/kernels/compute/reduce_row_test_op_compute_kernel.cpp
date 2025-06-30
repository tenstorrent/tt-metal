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

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);  // rows to process in this kernel
constexpr uint32_t Wt = get_compile_time_arg_val(1);

constexpr auto cb_input = tt::CBIndex::c_0;
constexpr auto cb_reduction_scaler = tt::CBIndex::c_1;
constexpr auto cb_matmul_reduce = tt::CBIndex::c_2;
constexpr auto cb_before_reduction = tt::CBIndex::c_3;  // using cb_reduction_scaler as second input
constexpr auto cb_output = tt::CBIndex::c_4;

constexpr uint32_t onetile = 1U;

#ifdef USE_MATMUL

void calculate_and_reduce_sum() {
    const uint32_t accum_register = 0;
    const uint32_t tile_register = 1U;
    cb_reserve_back(cb_before_reduction, onetile);

    tile_regs_acquire();
    for (uint32_t col = 0; col < Wt; ++col) {
        const uint32_t working_register = (col == 0) ? accum_register : tile_register;
        copy_tile_init(cb_input);
        copy_tile(cb_input, /* tile_idx */ col, /* register_idx */ working_register);

        if (col > 0) {
            add_binary_tile_init();
            add_binary_tile(accum_register, tile_register);
        }
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_before_reduction);
    pack_tile(accum_register, cb_before_reduction);
    tile_regs_release();
    cb_push_back(cb_before_reduction, onetile);

    cb_wait_front(cb_before_reduction, onetile);
    cb_reserve_back(cb_output, onetile);

    tile_regs_acquire();
    const uint32_t reduction_register = 0;
    // We used matmul_tiles instead of reduce_tile, because reduce_tile causes a loss of precision. The same issue has
    // been observed in moreh’s ops.
    mm_init(cb_before_reduction, cb_matmul_reduce, cb_output, 0);
    matmul_tiles(cb_before_reduction, cb_matmul_reduce, /* tile_idx */ 0, /* tile_idx */ 0, reduction_register, 0);

    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_output);
    pack_tile(/* tile_idx */ reduction_register, cb_output);
    tile_regs_release();
    cb_push_back(cb_output, onetile);
}

#else

void calculate_and_reduce_sum() {
    const uint32_t reduction_register = 0;
    cb_reserve_back(cb_output, onetile);

    tile_regs_acquire();
    reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_input, cb_reduction_scaler, cb_output);
    for (uint32_t col = 0; col < Wt; ++col) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_input,
            cb_reduction_scaler,
            /* tile_idx */ col,
            /* tile_idx */ 0,
            /* reduction_idx */ reduction_register);
    }
    reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_output);
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_output);
    pack_tile(/* tile_idx */ reduction_register, cb_output);
    tile_regs_release();
    cb_push_back(cb_output, onetile);
}

#endif

void MAIN {
    init_sfpu(cb_input, cb_output);
    binary_op_init_common(cb_input, cb_reduction_scaler, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_input, Wt);                  //  wait Wt tiles
        cb_wait_front(cb_matmul_reduce, onetile);     // wait for matmul tile
        cb_wait_front(cb_reduction_scaler, onetile);  // wait for scaler tile

        calculate_and_reduce_sum();

        cb_pop_front(cb_input, Wt);
    }

    cb_pop_front(cb_matmul_reduce, onetile);
    cb_pop_front(cb_reduction_scaler, onetile);
}

}  // namespace NAMESPACE
