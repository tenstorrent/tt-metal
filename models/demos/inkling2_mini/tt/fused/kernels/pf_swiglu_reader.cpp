// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill softcapped SwiGLU, reader: for tile rows R in [r0, r1) and column tiles j < HT, the gate tile (R, j) and the
// up tile (R, HT + j) of the gate/up output (bf16 TILE, 2 HT column tiles), read in place.
// Runtime args: gu, r0, r1.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t HT = get_named_compile_time_arg_val("ht");
    constexpr uint32_t cb_g = 0, cb_u = 1;
    const uint32_t gu = get_arg_val<uint32_t>(0), r0 = get_arg_val<uint32_t>(1), r1 = get_arg_val<uint32_t>(2);
    const InterleavedAddrGen<true> ga = {.bank_base_address = gu, .page_size = 2048};
    for (uint32_t R = r0; R < r1; ++R) {
        for (uint32_t j = 0; j < HT; ++j) {
            cb_reserve_back(cb_g, 1);
            cb_reserve_back(cb_u, 1);
            noc_async_read(ga.get_noc_addr(R * 2 * HT + j), get_write_ptr(cb_g), 2048);
            noc_async_read(ga.get_noc_addr(R * 2 * HT + HT + j), get_write_ptr(cb_u), 2048);
            noc_async_read_barrier();
            cb_push_back(cb_g, 1);
            cb_push_back(cb_u, 1);
        }
    }
}
