// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill softcapped SwiGLU, writer: tile (R, j) of the [rows, HT tiles] output for R in [r0, r1), j < HT.
// Runtime args: out, r0, r1.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t HT = get_named_compile_time_arg_val("ht");
    constexpr uint32_t cb_out = 16;
    const uint32_t out = get_arg_val<uint32_t>(0), r0 = get_arg_val<uint32_t>(1), r1 = get_arg_val<uint32_t>(2);
    const InterleavedAddrGen<true> oa = {.bank_base_address = out, .page_size = 2048};
    for (uint32_t R = r0; R < r1; ++R) {
        for (uint32_t j = 0; j < HT; ++j) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), oa.get_noc_addr(R * HT + j), 2048);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
