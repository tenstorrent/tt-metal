// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the used-rows untilize (tt-blaze G53F-0203): ttnn.untilize's compute (untilize_variable_num_blocks.cpp,
// the kernel_lib untilize over full-width Wt-tile blocks) with the block count taken from the reader at run time
// (the device-side used row count) instead of a host runtime argument.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t cb_in_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_out_id = get_compile_time_arg_val(2);
constexpr uint32_t cb_cnt_id = get_compile_time_arg_val(3);

void kernel_main() {
    cb_wait_front(cb_cnt_id, 1);
    const uint32_t n = read_tile_value(cb_cnt_id, 0, 0);
    cb_pop_front(cb_cnt_id, 1);
    if (n == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_in_id, cb_out_id);
    compute_kernel_lib::untilize<
        Wt,
        cb_in_id,
        cb_out_id,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(n);
}
