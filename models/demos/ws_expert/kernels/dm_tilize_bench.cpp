// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Data-movement tilize microbench (one RISC): a row-major block of 32 rows x W tiles (bf16, row pitch W * 64 B) in L1
// at SRC -> W tiles at DST (faces 16 x 16, face order 0 1 2 3). Each tile is 64 face rows of 32 B:
//   MODE 0: local NoC reads (32 B each); MODE 1: RISC copies (4 x uint32 a 16 B quarter); MODE 2: NoC reads of
//   whole 16-row face columns via 16 reads of 32 B batched with a single barrier per block (same as 0, barrier once).
// Repeats REPS times. Stamps at ST: [0] start, [1] end.
// CT: 0 W, 1 REPS, 2 SRC, 3 DST, 4 ST, 5 MODE, 6 my noc x, 7 my noc y
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t REPS = get_compile_time_arg_val(1);
    constexpr uint32_t SRC = get_compile_time_arg_val(2);
    constexpr uint32_t DST = get_compile_time_arg_val(3);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_compile_time_arg_val(4));
    constexpr uint32_t MODE = get_compile_time_arg_val(5);
    constexpr uint32_t mx = get_compile_time_arg_val(6), my = get_compile_time_arg_val(7);
    constexpr uint32_t pitch = W * 64;
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    const uint64_t self = get_noc_addr(mx, my, 0);
    st[0] = wall();
    for (uint32_t rep = 0; rep < REPS; ++rep) {
        for (uint32_t t = 0; t < W; ++t) {
            const uint32_t dst_t = DST + t * 2048;
            for (uint32_t f = 0; f < 4; ++f) {
                const uint32_t r0 = (f >> 1) * 16, c0 = (f & 1) * 16;
                for (uint32_t r = 0; r < 16; ++r) {
                    const uint32_t src = SRC + (r0 + r) * pitch + t * 64 + c0 * 2;
                    const uint32_t dst = dst_t + f * 512 + r * 32;
                    if constexpr (MODE == 1) {
                        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
                        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
#pragma GCC unroll 8
                        for (uint32_t w = 0; w < 8; ++w) {
                            d[w] = s[w];
                        }
                    } else {
                        noc_async_read(self | src, dst, 32);
                    }
                }
            }
            if constexpr (MODE == 0) {
                noc_async_read_barrier();
            }
        }
        if constexpr (MODE == 2) {
            noc_async_read_barrier();
        }
    }
    noc_async_read_barrier();
    st[1] = wall();
}
