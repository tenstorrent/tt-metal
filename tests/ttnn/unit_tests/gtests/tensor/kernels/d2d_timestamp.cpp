// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Lightweight on-device Tensix cycle timestamp recorder.
// Appends a 64-bit wall-clock sample (lo, hi) to an L1 buffer whose word[0] holds
// the sample count, so one cached MeshWorkload can be enqueued repeatedly around
// Recv / Wait / Compute with zero host runtime-arg updates.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t buf_l1_addr = get_compile_time_arg_val(0);
constexpr uint32_t max_samples = get_compile_time_arg_val(1);

void kernel_main() {
    volatile tt_reg_ptr uint32_t* clk_lo =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile tt_reg_ptr uint32_t* clk_hi =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    const uint32_t lo = *clk_lo;  // latches WALL_CLOCK_H
    const uint32_t hi = *clk_hi;

    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_l1_addr);
    const uint32_t idx = buf[0];
    if (idx < max_samples) {
        buf[2u + (2u * idx)] = lo;
        buf[2u + (2u * idx) + 1u] = hi;
        buf[0] = idx + 1u;
    }
}
