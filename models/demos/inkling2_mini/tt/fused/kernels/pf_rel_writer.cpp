// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill skewed rel bias, writer: row m of each untilized 32 x (32 NB) block is the output row (page) of query
// q = 32 (32 i + m) + r of head h (rows past the chunk are dropped).
// Runtime args: out, n_units, h / r per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T = get_named_compile_time_arg_val("t");
    constexpr uint32_t NB = get_named_compile_time_arg_val("nb");
    constexpr uint32_t QT = T / 32, NA = (QT + 31) / 32, ROW = NB * 64;
    constexpr uint32_t cb_out = 16;
    const uint32_t out = get_arg_val<uint32_t>(0), n_units = get_arg_val<uint32_t>(1);
    const InterleavedAddrGen<true> oa = {.bank_base_address = out, .page_size = ROW};
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t h = get_arg_val<uint32_t>(2 + 2 * u), r = get_arg_val<uint32_t>(3 + 2 * u);
        for (uint32_t i = 0; i < NA; ++i) {
            cb_wait_front(cb_out, NB);
            const uint32_t l = get_read_ptr(cb_out);
            for (uint32_t m = 0; m < 32; ++m) {
                const uint32_t qt = 32 * i + m;
                if (qt < QT) {
                    noc_async_write(l + m * ROW, oa.get_noc_addr(h * T + 32 * qt + r), ROW);
                }
            }
            noc_async_writes_flushed();  // the block's L1 can be reused once the writes left; acks at the end
            cb_pop_front(cb_out, NB);
        }
    }
    noc_async_write_barrier();
}
