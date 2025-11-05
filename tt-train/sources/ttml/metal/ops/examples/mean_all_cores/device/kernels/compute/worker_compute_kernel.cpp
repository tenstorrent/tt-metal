// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

// Compile time arguments
constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

// Circular buffer indices
constexpr auto cb_input = tt::CBIndex::c_0;
constexpr auto cb_reduction_scaler = tt::CBIndex::c_1;
constexpr auto cb_interm_output = tt::CBIndex::c_2;

constexpr uint32_t onetile = 1;

void MAIN {
    cb_wait_front(cb_reduction_scaler, onetile);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_input, Wt);

        reconfig_data_format(cb_input, cb_reduction_scaler);
        const uint32_t reduction_register = 0;
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR, /*enforce_fp32_accumulation*/ true>(
            cb_input, cb_reduction_scaler, cb_interm_output);
        tile_regs_acquire();
        for (uint32_t col = 0; col < Wt; ++col) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR, /*enforce_fp32_accumulation*/ true>(
                cb_input,
                cb_reduction_scaler,
                /* tile_idx */ col,
                /* tile_idx */ 0,
                /* reduction_register */ reduction_register);
        }
        reduce_uninit</*enforce_fp32_accumulation*/ true>();
        tile_regs_commit();

        cb_reserve_back(cb_interm_output, onetile);
        tile_regs_wait();
        pack_reconfig_data_format(cb_interm_output);
        pack_tile(/* tile_idx */ reduction_register, cb_interm_output);
        tile_regs_release();
        cb_push_back(cb_interm_output, onetile);
    }
}

}  // namespace NAMESPACE
