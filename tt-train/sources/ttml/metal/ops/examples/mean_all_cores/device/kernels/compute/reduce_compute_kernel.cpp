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
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

// Compile time arguments
constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);

// Circular buffer indices
constexpr uint32_t cb_transfer_input_01 = tt::CBIndex::c_3;
constexpr uint32_t cb_transfer_input_02 = tt::CBIndex::c_4;
constexpr uint32_t cb_output = tt::CBIndex::c_5;

constexpr uint32_t onetile = 1;

void MAIN {
    init_sfpu(cb_transfer_input_01, cb_output);
    binary_op_init_common(cb_transfer_input_01, cb_transfer_input_02, cb_output);
    cb_wait_front(cb_transfer_input_01, onetile);  // wait for 2 tiles from transfer buffer
    cb_wait_front(cb_transfer_input_02, onetile);

    const uint32_t reduction_register = 0;
    tile_regs_acquire();
    add_tiles_init(cb_transfer_input_01, cb_transfer_input_02);
    add_tiles(cb_transfer_input_01, cb_transfer_input_02, /*tile_idx*/ 0, /*tile_idx*/ 0, reduction_register);
    tile_regs_commit();

    cb_reserve_back(cb_output, onetile);
    tile_regs_wait();
    pack_reconfig_data_format(cb_output);
    pack_tile(/* tile_idx */ reduction_register, cb_output);
    tile_regs_release();
    cb_push_back(cb_output, onetile);
}

}  // namespace NAMESPACE
