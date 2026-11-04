// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 EP combine — writer (moe_combine_common.hpp): y tile (r, c) of [C, H] bf16 TILE. Compile-time: CTC.
// Runtime: y address, first unit, unit count.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "moe_combine_common.hpp"

using namespace mcomb;

void kernel_main() {
    constexpr uint32_t CTC = get_compile_time_arg_val(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    if (n == 0) {
        return;
    }
    const InterleavedAddrGen<true> yg{.bank_base_address = y_addr, .page_size = BF16_TILE};
    for (uint32_t u = u0; u < u0 + n; ++u) {
        cb_wait_front(CB_Y, 1);
        noc_async_write(get_read_ptr(CB_Y), yg.get_noc_addr(u), BF16_TILE);  // unit u = tile r CTC + c
        noc_async_write_barrier();
        cb_pop_front(CB_Y, 1);
    }
}
