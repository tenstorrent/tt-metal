// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/deepseek_moe_gate.h"

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t bias_cb = get_compile_time_arg_val(1);
    constexpr uint32_t input_indices_cb = get_compile_time_arg_val(2);
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);
    constexpr uint32_t output_indices_cb = get_compile_time_arg_val(4);
    constexpr uint32_t eps = get_compile_time_arg_val(5);
    constexpr uint32_t scaling_factor = get_compile_time_arg_val(6);
    constexpr uint32_t enable_sigmoid = get_compile_time_arg_val(7);

    // Init portion, only done once
    binary_op_init_common(input_cb, bias_cb, output_cb);
    cb_wait_front(input_indices_cb, 1);
    cb_wait_front(bias_cb, 1);

    // Compute portion, to be looped in fused op
    copy_tile_to_dst_init_short(input_indices_cb);
    reconfig_data_format_srca(input_indices_cb);

    tile_regs_acquire();

    // Copy indices (already transposed to cols)
    copy_tile(input_indices_cb, 0, 1);

    reconfig_data_format_srca(input_cb);
    deepseek_moe_gate_init<enable_sigmoid>(input_cb, bias_cb);
    cb_wait_front(input_cb, 1);
    deepseek_moe_gate<enable_sigmoid>(input_cb, bias_cb, eps, scaling_factor);
    // Pop input tile
    cb_pop_front(input_cb, 1);

    tile_regs_commit();

    pack_reconfig_data_format(output_cb);
    cb_reserve_back(output_cb, 1);
    cb_reserve_back(output_indices_cb, 1);

    tile_regs_wait();

    pack_tile(0, output_cb);
    cb_push_back(output_cb, 1);

    pack_reconfig_data_format(output_indices_cb);
    pack_tile(1, output_indices_cb);
    cb_push_back(output_indices_cb, 1);

    tile_regs_release();
}
}  // namespace NAMESPACE
