// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute tilize microbench: a resident row-major block (CB 0, 32 rows x W tiles, bf16) tilized REPS times to CB 16
// with fast_tilize (fp32 DEST, as the expert's compute runs). SWITCH 1: every rep also pays the mode switch the
// expert would (fast_tilize_init / uninit + matmul_block_init back), else init once.
// Stamps at RT 0: [0] start, [1] end (UNPACK / PACK). CT: 0 W, 1 REPS, 2 SWITCH
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t REPS = get_compile_time_arg_val(1);
    constexpr uint32_t SWITCH = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(0, 16);
    cb_wait_front(0, W);
    if constexpr (!SWITCH) {
        fast_tilize_init(0, W, 16);
    }
    UNPACK(st[0] = wall());
    for (uint32_t r = 0; r < REPS; ++r) {
        if constexpr (SWITCH) {
            fast_tilize_init(0, W, 16);
        }
        cb_reserve_back(16, W);
        fast_tilize_block(0, W, 16);
        cb_push_back(16, W);
        if constexpr (SWITCH) {
            fast_tilize_uninit(0, 16, W);
            matmul_block_init(0, 1, false, 4, 1, 120);
        }
    }
    PACK(st[1] = wall());
}
