// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

// Compile-time args:
//   0: cb_id_in0          (= kCbSrc0)
//   1: cb_id_out0         (= kCbOut)
//   2: total_tiles_per_core
//   3: tiles_per_chunk    (Wt / num_chunks — tiles wide per chunk)
//   4: num_chunks

void kernel_main() {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t total_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(4);

    // === DMA-ONLY PASSTHROUGH (tilize disabled for perf measurement) ===
    // Keeps the CB handshake alive so reader/writer stay in sync, but skips
    // tilize math entirely. Output will be uninitialized L1 garbage — correctness
    // tests will fail; only use for measuring pure data-movement cost.
    // for (uint32_t block = 0; block < total_tiles * num_chunks; ++block) {
    //     cb_wait_front(cb_id_in0, tiles_per_chunk);
    //     cb_reserve_back(cb_id_out0, tiles_per_chunk);
    //     cb_push_back(cb_id_out0, tiles_per_chunk);
    //     cb_pop_front(cb_id_in0, tiles_per_chunk);
    // }

    // === ORIGINAL TILIZE ===
    compute_kernel_hw_startup(cb_id_in0, cb_id_out0);
    compute_kernel_lib::tilize<
        tiles_per_chunk,
        cb_id_in0,
        cb_id_out0,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(total_tiles * num_chunks);
}
