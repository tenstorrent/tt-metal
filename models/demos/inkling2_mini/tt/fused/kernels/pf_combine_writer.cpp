// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill MoE combine, writer: the CW output tiles of unit (R, j) to tiles (R, CW j + c) of [T, HIDDEN].
// Runtime args: out, n_units, (R, j) per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CW = get_named_compile_time_arg_val("cw");
    constexpr uint32_t OT = get_named_compile_time_arg_val("ot");  // output tile columns
    constexpr uint32_t cb_out = 16;
    const uint32_t out = get_arg_val<uint32_t>(0), n_units = get_arg_val<uint32_t>(1);
    const InterleavedAddrGen<true> oa = {.bank_base_address = out, .page_size = 2048};
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t R = get_arg_val<uint32_t>(2 + 2 * u), j = get_arg_val<uint32_t>(3 + 2 * u);
        cb_wait_front(cb_out, CW);
        for (uint32_t c = 0; c < CW; ++c) {
            noc_async_write(get_read_ptr(cb_out) + c * 2048, oa.get_noc_addr(R * OT + CW * j + c), 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, CW);
    }
}
