// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill causal conv, writer: output tile (R, j) of every unit row to out (page bytes per the output dtype).
// Runtime args: out, page bytes, n_units, (j, R0, R1) per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t WT = get_named_compile_time_arg_val("wt");
    constexpr uint32_t cb_out = 16;
    const uint32_t out = get_arg_val<uint32_t>(0), page = get_arg_val<uint32_t>(1);
    const uint32_t n_units = get_arg_val<uint32_t>(2);
    const InterleavedAddrGen<true> oa = {.bank_base_address = out, .page_size = page};
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t j = get_arg_val<uint32_t>(3 + 3 * u), r0 = get_arg_val<uint32_t>(4 + 3 * u);
        const uint32_t r1 = get_arg_val<uint32_t>(5 + 3 * u);
        for (uint32_t R = r0; R < r1; ++R) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), oa.get_noc_addr(R * WT + j), page);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
