// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// tests/bfp8_tiny_probe.py: CB 0 (bf16 batch x 32) -> DST (fp32) -> CB 1 (bfp8 batch x 32), as the streamers pack.
// Compile-time arg 0: n.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(0, 1);
    cb_wait_front(0, n);
    cb_reserve_back(1, n);
    pack_reconfig_data_format<true>(1);
    copy_init(0);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(0, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, 1, i);
        tile_regs_release();
    }
    cb_push_back(1, n);
    cb_pop_front(0, n);
}
