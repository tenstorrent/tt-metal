// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill attention output gate, reader. Unit = (tile row R, head h): attn tiles (h, R, c) of the [1, n, T, 128]
// SDPA output and gate tiles (R, g0 + 4 h + c) of the [T, W] projection rows (W: GT tile columns), c = 0..3, bf16.
// Runtime args: attn, gate, n_units, (R, h) per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t RT = get_named_compile_time_arg_val("rt");
    constexpr uint32_t GT = get_named_compile_time_arg_val("gt");
    constexpr uint32_t G0 = get_named_compile_time_arg_val("g0");
    constexpr uint32_t cb_a = 0, cb_g = 1;
    const uint32_t attn = get_arg_val<uint32_t>(0), gate = get_arg_val<uint32_t>(1);
    const uint32_t n_units = get_arg_val<uint32_t>(2);
    const InterleavedAddrGen<true> aa = {.bank_base_address = attn, .page_size = 2048};
    const InterleavedAddrGen<true> ga = {.bank_base_address = gate, .page_size = 2048};
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t R = get_arg_val<uint32_t>(3 + 2 * u), h = get_arg_val<uint32_t>(4 + 2 * u);
        cb_reserve_back(cb_a, 4);
        cb_reserve_back(cb_g, 4);
        for (uint32_t c = 0; c < 4; ++c) {
            noc_async_read(aa.get_noc_addr((h * RT + R) * 4 + c), get_write_ptr(cb_a) + c * 2048, 2048);
            noc_async_read(ga.get_noc_addr(R * GT + G0 + 4 * h + c), get_write_ptr(cb_g) + c * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, 4);
        cb_push_back(cb_g, 4);
    }
}
