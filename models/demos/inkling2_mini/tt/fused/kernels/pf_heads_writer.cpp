// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill per-head norm, writer: the TPH output tiles of unit (R, h) to [1, n, T, 32 TPH] (tile (h RT + R) TPH + c).
// Runtime args: out, n_units, (R, h) per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t RT = get_named_compile_time_arg_val("rt");  // tile rows (T / 32)
    constexpr uint32_t TPH = get_named_compile_time_arg_val("tph");
    constexpr uint32_t cb_out = 19;
    const uint32_t out = get_arg_val<uint32_t>(0), n_units = get_arg_val<uint32_t>(1);
    const InterleavedAddrGen<true> oa = {.bank_base_address = out, .page_size = 2048};
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t R = get_arg_val<uint32_t>(2 + 2 * u), h = get_arg_val<uint32_t>(3 + 2 * u);
        const uint32_t base = (h * RT + R) * TPH;
        cb_wait_front(cb_out, TPH);
        for (uint32_t c = 0; c < TPH; ++c) {
            noc_async_write(get_read_ptr(cb_out) + c * 2048, oa.get_noc_addr(base + c), 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, TPH);
    }
}
