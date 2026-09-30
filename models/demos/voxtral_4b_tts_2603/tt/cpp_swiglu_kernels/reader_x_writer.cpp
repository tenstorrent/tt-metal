// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The activation stream and the output writer of the fused C++ SwiGLU (see tt/cpp_swiglu.py).
// Runs on the other RISC / NoC from the weight stream: every core reads the whole [MT x KT]
// activation (resident in L1) one K block at a time, then writes its MT x PN gated output tiles,
// RT rows of one column at a time, in the order the compute kernel packs them.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t n0 = get_arg_val<uint32_t>(2);

    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t PN = get_compile_time_arg_val(3);
    constexpr uint32_t RT = get_compile_time_arg_val(4);
    constexpr uint32_t KT = get_compile_time_arg_val(5);
    constexpr uint32_t NT = get_compile_time_arg_val(6);
    constexpr auto ax = TensorAccessorArgs<7>();
    constexpr auto ay = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(ax, x_addr);
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_y = 16;
    const uint32_t x_bytes = get_tile_size(cb_x);
    const uint32_t y_bytes = get_tile_size(cb_y);

    for (uint32_t b = 0; b < NB; ++b) {
        cb_reserve_back(cb_x, MT * KB);
        uint32_t l1 = get_write_ptr(cb_x);
        for (uint32_t r = 0; r < MT; ++r) {
            uint32_t t = r * KT + b * KB;
            for (uint32_t k = 0; k < KB; ++k) {
                noc_async_read_page(t + k, sx, l1);
                l1 += x_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, MT * KB);
    }

    for (uint32_t rs = 0; rs < MT / RT; ++rs) {
        for (uint32_t p = 0; p < PN; ++p) {
            cb_wait_front(cb_y, RT);
            uint32_t l1 = get_read_ptr(cb_y);
            for (uint32_t j = 0; j < RT; ++j) {
                noc_async_write_page((rs * RT + j) * NT + n0 + p, sy, l1);
                l1 += y_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_y, RT);
        }
    }
}
