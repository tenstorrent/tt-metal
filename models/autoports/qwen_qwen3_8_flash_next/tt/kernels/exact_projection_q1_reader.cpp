// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t xb = get_compile_time_arg_val(1), wb = get_compile_time_arg_val(2);
    constexpr bool zero_once = get_compile_time_arg_val(3);
    constexpr auto xa = TensorAccessorArgs<4>();
    constexpr auto wa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, get_arg_val<uint32_t>(0), 1024 * xb);
    const auto w = TensorAccessor(wa, get_arg_val<uint32_t>(1), 1024 * wb);
    const uint32_t first = get_arg_val<uint32_t>(2), stride = get_arg_val<uint32_t>(3);
    const uint32_t groups = get_arg_val<uint32_t>(4), kt = get_arg_val<uint32_t>(5);
    if constexpr (zero_once) {
        // Both two-page CB allocations are free before any push. Reserve waits
        // for capacity but does not advance the write pointer. Initialize every
        // byte of both pages before compute can observe the first live page.
        cb_reserve_back(2, 2);
        cb_reserve_back(3, 2);
        volatile tt_l1_ptr uint32_t* xx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));
        volatile tt_l1_ptr uint32_t* ww = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3));
        for (uint32_t i = 0; i < 512 * xb; ++i) {
            xx[i] = 0;
        }
        for (uint32_t i = 0; i < 512 * wb; ++i) {
            ww[i] = 0;
        }
        // Do not push initialization pages. Every later read fully overwrites
        // exactly the same row0/R-row footprint in a free page. Compute only
        // unpacks CB2/3; result packing and scratch use distinct CB4/5.
    }
    for (uint32_t g = first; g < groups; g += stride) {
        const uint32_t row = g * rows;
        const uint32_t offset = ((row % 32) / 16) * 512 + (row % 16) * 16;
        for (uint32_t k = 0; k < kt; ++k) {
            cb_reserve_back(2, 1);
            cb_reserve_back(3, 1);
            const uint32_t xp = get_write_ptr(2), wp = get_write_ptr(3);
            if constexpr (!zero_once) {
                // Conservative default: initialize each page on every use.
                volatile tt_l1_ptr uint32_t* xx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xp);
                volatile tt_l1_ptr uint32_t* ww = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wp);
                for (uint32_t i = 0; i < 256 * xb; ++i) {
                    xx[i] = 0;
                }
                for (uint32_t i = 0; i < 256 * wb; ++i) {
                    ww[i] = 0;
                }
            }
            const uint64_t xn = x.get_noc_addr(k), wn = w.get_noc_addr((row / 32) * kt + k);
            noc_async_read(xn, xp, 16 * xb);
            noc_async_read(xn + 256 * xb, xp + 256 * xb, 16 * xb);
            noc_async_read(wn + offset * wb, wp, rows * 16 * wb);
            noc_async_read(wn + (offset + 256) * wb, wp + 256 * wb, rows * 16 * wb);
            noc_async_read_barrier();
            cb_push_back(2, 1);
            cb_push_back(3, 1);
        }
    }
}
