// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill cross-pass messages, writer: per unit tile (R, j), message tiles k = 0..3 to their 4 outputs (fp32).
// Runtime args: out0..out3, n_units, (R, j0, j1) per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t WT = get_named_compile_time_arg_val("wt");
    constexpr uint32_t cb_out = 16;
    const InterleavedAddrGen<true> oa[4] = {
        {.bank_base_address = get_arg_val<uint32_t>(0), .page_size = 4096},
        {.bank_base_address = get_arg_val<uint32_t>(1), .page_size = 4096},
        {.bank_base_address = get_arg_val<uint32_t>(2), .page_size = 4096},
        {.bank_base_address = get_arg_val<uint32_t>(3), .page_size = 4096}};
    const uint32_t n_units = get_arg_val<uint32_t>(4);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t R = get_arg_val<uint32_t>(5 + 3 * u), j0 = get_arg_val<uint32_t>(6 + 3 * u);
        const uint32_t j1 = get_arg_val<uint32_t>(7 + 3 * u);
        for (uint32_t j = j0; j < j1; ++j) {
            for (uint32_t k = 0; k < 4; ++k) {
                cb_wait_front(cb_out, 1);
                noc_async_write(get_read_ptr(cb_out), oa[k].get_noc_addr(R * WT + j), 4096);
                noc_async_writes_flushed();
                cb_pop_front(cb_out, 1);
            }
        }
    }
    noc_async_write_barrier();
}
