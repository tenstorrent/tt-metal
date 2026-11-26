// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill cross-pass messages, reader. Unit = (tile row R, column tiles j0..j1). Per unit: the w tile (R, c0 / 32)
// of the [T, 64] fp32 weights, spread into 4 column-broadcast tiles (message k: every column of row r holds
// w[r, c0 + k]); per j: n1[R, j] and diff[R, j] (fp32).
// Runtime args: n1, diff, w, c0, n_units, (R, j0, j1) per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t WT = get_named_compile_time_arg_val("wt");
    constexpr uint32_t cb_n = 0, cb_d = 1, cb_w = 2, cb_ws = 3;
    const InterleavedAddrGen<true> na = {.bank_base_address = get_arg_val<uint32_t>(0), .page_size = 4096};
    const InterleavedAddrGen<true> da = {.bank_base_address = get_arg_val<uint32_t>(1), .page_size = 4096};
    const InterleavedAddrGen<true> wa = {.bank_base_address = get_arg_val<uint32_t>(2), .page_size = 4096};
    const uint32_t c0 = get_arg_val<uint32_t>(3), n_units = get_arg_val<uint32_t>(4);
    const uint32_t wtiles = 2;  // [T, 64] fp32: 2 column tiles
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t R = get_arg_val<uint32_t>(5 + 3 * u), j0 = get_arg_val<uint32_t>(6 + 3 * u);
        const uint32_t j1 = get_arg_val<uint32_t>(7 + 3 * u);
        // w tile -> scratch, then 4 broadcast tiles
        cb_reserve_back(cb_ws, 1);
        noc_async_read(wa.get_noc_addr(R * wtiles + c0 / 32), get_write_ptr(cb_ws), 4096);
        noc_async_read_barrier();
        const volatile uint32_t* src = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_ws));
        cb_reserve_back(cb_w, 4);
        for (uint32_t k = 0; k < 4; ++k) {
            const uint32_t c = (c0 + k) % 32;
            volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_w) + k * 4096);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t v = src[((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)];
                volatile uint32_t* row0 = dst + (r >> 4) * 512 + (r & 15) * 16;  // face (r/16, 0) row
                for (uint32_t i = 0; i < 16; ++i) {
                    row0[i] = v;
                    row0[256 + i] = v;  // face (r/16, 1)
                }
            }
        }
        cb_push_back(cb_w, 4);
        for (uint32_t j = j0; j < j1; ++j) {
            cb_reserve_back(cb_n, 1);
            cb_reserve_back(cb_d, 1);
            noc_async_read(na.get_noc_addr(R * WT + j), get_write_ptr(cb_n), 4096);
            noc_async_read(da.get_noc_addr(R * WT + j), get_write_ptr(cb_d), 4096);
            noc_async_read_barrier();
            cb_push_back(cb_n, 1);
            cb_push_back(cb_d, 1);
        }
    }
}
