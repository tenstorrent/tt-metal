// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill attention sink, reader: the H selection tiles once, then per tile row R the qgrs tile (R, col).
// Runtime args: qgrs, sel, n_rows, R per row.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_named_compile_time_arg_val("h");
    constexpr uint32_t QT = get_named_compile_time_arg_val("qt");
    constexpr uint32_t COL = get_named_compile_time_arg_val("col");
    constexpr uint32_t cb_q = 0, cb_sel = 1;
    const uint32_t q = get_arg_val<uint32_t>(0), sel = get_arg_val<uint32_t>(1), n = get_arg_val<uint32_t>(2);
    const InterleavedAddrGen<true> qa = {.bank_base_address = q, .page_size = 2048};
    const InterleavedAddrGen<true> sa = {.bank_base_address = sel, .page_size = 2048};
    cb_reserve_back(cb_sel, H);
    for (uint32_t h = 0; h < H; ++h) {
        noc_async_read(sa.get_noc_addr(h), get_write_ptr(cb_sel) + h * 2048, 2048);
    }
    noc_async_read_barrier();
    cb_push_back(cb_sel, H);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t R = get_arg_val<uint32_t>(3 + i);
        cb_reserve_back(cb_q, 1);
        noc_async_read(qa.get_noc_addr(R * QT + COL), get_write_ptr(cb_q), 2048);
        noc_async_read_barrier();
        cb_push_back(cb_q, 1);
    }
}
