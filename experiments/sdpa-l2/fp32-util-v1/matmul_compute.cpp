// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/matmul_custom.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t kt = get_compile_time_arg_val(0);
    constexpr uint32_t repetitions = get_compile_time_arg_val(1);
    constexpr bool transpose = get_compile_time_arg_val(2) != 0;
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    mm_no_mop_init_short(0, 1, transpose, 4, 1, kt);
    cb_wait_front(0, kt);
    cb_wait_front(1, kt * 4);
    cb_reserve_back(16, 4);
    {
        DeviceZoneScopedN("UTIL_MATCHING_MM");
        for (uint32_t rep = 0; rep < repetitions; ++rep) {
            tile_regs_acquire();
            for (uint32_t inner = 0; inner < kt; ++inner) {
                matmul_block_no_mop(0, 1, inner, inner * 4, 0, transpose, 4, 1, kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < 4; ++j) {
                pack_tile<true>(j, 16, j);
            }
            tile_regs_release();
        }
    }
    cb_push_back(16, 4);
    cb_pop_front(0, kt);
    cb_pop_front(1, kt * 4);
}
