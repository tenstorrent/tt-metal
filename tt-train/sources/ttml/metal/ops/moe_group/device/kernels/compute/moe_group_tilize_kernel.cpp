// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

// Compile-time args:
//   0: cb_id_in0          (= kCbSrc0)
//   1: cb_id_out0         (= kCbOut)
//   2: total_tiles_per_core   (CT worst-case bound, kept for safety/diag only)
//   3: tiles_per_chunk    (Wt / num_chunks — tiles wide per chunk)
//   4: num_chunks
//   5: cb_id_ctrl         (NCRISC publishes per-core active block count here)

void kernel_main() {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t total_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_ctrl = get_compile_time_arg_val(5);
    (void)total_tiles;
    (void)num_chunks;

    compute_kernel_hw_startup(cb_id_in0, cb_id_out0);

    cb_wait_front(cb_id_ctrl, 1U);
    uint32_t active_blocks = read_tile_value(cb_id_ctrl, 0U, 0U);
    cb_pop_front(cb_id_ctrl, 1U);

    compute_kernel_lib::tilize<
        tiles_per_chunk,
        cb_id_in0,
        cb_id_out0,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(active_blocks);
}
