// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#define EXP_APPROX_MODE false

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h"

using namespace ckernel;

void kernel_main() {
    // CB indices passed as compile-time args
    constexpr uint32_t cb_l1 = get_compile_time_arg_val(0);       // l1 input
    constexpr uint32_t cb_l2 = get_compile_time_arg_val(1);       // l2 input
    constexpr uint32_t cb_ms1 = get_compile_time_arg_val(2);      // ms1 input (worker max)
    constexpr uint32_t cb_ms2 = get_compile_time_arg_val(3);      // ms2 input (prev max)
    constexpr uint32_t cb_l_out = get_compile_time_arg_val(4);    // l output
    constexpr uint32_t cb_ms_out = get_compile_time_arg_val(5);   // ms output (cur max)
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(6);  // scale as fp32 bits
    constexpr uint32_t block_size = get_compile_time_arg_val(7);  // number of row tiles
    constexpr uint32_t num_blocks = get_compile_time_arg_val(8);  // number of column tiles
    constexpr bool final_reduction = get_compile_time_arg_val(9);

    constexpr int vector_mode = VectorMode::RC_custom;

    binary_op_init_common(cb_l1, cb_l1, cb_l_out);
    exp_tile_init<EXP_APPROX_MODE, false>();

    sdpa_tail<EXP_APPROX_MODE, final_reduction, block_size, num_blocks, scale_fp32, vector_mode>(
        cb_ms1,     // worker max (ms1)
        cb_ms2,     // prev max (m2)
        cb_ms_out,  // cur max output (m = max(m1, m2))
        cb_l1,      // l1 input
        cb_l2,      // l2 input
        cb_l_out    // l output
    );
}
