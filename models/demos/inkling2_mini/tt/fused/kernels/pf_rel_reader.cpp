// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill skewed rel bias, reader. Unit = (head h, row phase r): the queries q = 32 qt + r of head h. Per unit:
// the 33 tiles of B_r (the projection skewed for phase r, tiles r * 33 ..), then per block i the rel states of
// queries qt = 32 i + m, m < 32 (ROW_MAJOR rows of 32 bf16, page h * T + q) as one row-major 32 x 32 block
// (missing rows past the chunk left stale: their outputs are not written).
// Runtime args: rs, bsk, n_units, h / r per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T = get_named_compile_time_arg_val("t");
    constexpr uint32_t NB = get_named_compile_time_arg_val("nb");  // tiles of B_r
    constexpr uint32_t QT = T / 32, NA = (QT + 31) / 32;
    constexpr uint32_t cb_rm = 0, cb_b = 1;
    const uint32_t rs = get_arg_val<uint32_t>(0), bsk = get_arg_val<uint32_t>(1), n_units = get_arg_val<uint32_t>(2);
    const InterleavedAddrGen<true> ra = {.bank_base_address = rs, .page_size = 64};
    const InterleavedAddrGen<true> ba = {.bank_base_address = bsk, .page_size = 2048};
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t h = get_arg_val<uint32_t>(3 + 2 * u), r = get_arg_val<uint32_t>(4 + 2 * u);
        cb_reserve_back(cb_b, NB);
        const uint32_t lb = get_write_ptr(cb_b);
        for (uint32_t n = 0; n < NB; ++n) {
            noc_async_read(ba.get_noc_addr(r * NB + n), lb + n * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(cb_b, NB);
        for (uint32_t i = 0; i < NA; ++i) {
            cb_reserve_back(cb_rm, 1);
            const uint32_t l = get_write_ptr(cb_rm);
            for (uint32_t m = 0; m < 32; ++m) {
                const uint32_t qt = 32 * i + m;
                if (qt < QT) {
                    noc_async_read(ra.get_noc_addr(h * T + 32 * qt + r), l + m * 64, 64);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_rm, 1);
        }
    }
}
