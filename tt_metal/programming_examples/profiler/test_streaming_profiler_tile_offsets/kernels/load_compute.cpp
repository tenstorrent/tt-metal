// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// FPU load: `bursts` of `iters` HiFi4 matmul blocks of 8 tiles each, on whatever the input CBs hold, with `idle`
// cycles between bursts. Every TRISC runs the same loop, so the three stay in step.
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"

inline void spin(uint32_t cycles) {
    volatile uint32_t* const wall = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t t0 = *wall;
    while (*wall - t0 < cycles) {
    }
}

void kernel_main() {
    const uint32_t bursts = get_arg_val<uint32_t>(0);
    const uint32_t iters = get_arg_val<uint32_t>(1);
    const uint32_t idle = get_arg_val<uint32_t>(2);
    constexpr auto in0 = tt::CBIndex::c_0;
    constexpr auto in1 = tt::CBIndex::c_1;
    constexpr auto out = tt::CBIndex::c_16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out);
    matmul_init(in0, in1);
    for (uint32_t b = 0; b < bursts; b++) {
        for (uint32_t i = 0; i < iters; i++) {
            tile_regs_acquire();
            for (uint32_t t = 0; t < 8; t++) {
                matmul_tiles(in0, in1, 0, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();
        }
        spin(idle);
    }
}
