// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill message-gate tail, reader. Unit = (column tile j, tile rows R0..R1). Per unit: forget / gain tiles
// (k, j), k < kz (fp32); per row R: the kz z tiles of row R (bf16), h[R, j] and m[R, j] (fp32).
// Runtime args: z, forget, gain, h, m, n_units, (j, R0, R1) per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t WT = get_named_compile_time_arg_val("wt");
    constexpr uint32_t KZ = get_named_compile_time_arg_val("kz");
    constexpr uint32_t M_PAGE = get_named_compile_time_arg_val("m_page");  // 4096 fp32, 2048 bf16
    constexpr uint32_t cb_z = 0, cb_f = 1, cb_g = 2, cb_h = 3, cb_m = 4;
    const InterleavedAddrGen<true> za = {.bank_base_address = get_arg_val<uint32_t>(0), .page_size = 2048};
    const InterleavedAddrGen<true> fa = {.bank_base_address = get_arg_val<uint32_t>(1), .page_size = 4096};
    const InterleavedAddrGen<true> ga = {.bank_base_address = get_arg_val<uint32_t>(2), .page_size = 4096};
    const InterleavedAddrGen<true> ha = {.bank_base_address = get_arg_val<uint32_t>(3), .page_size = 4096};
    const InterleavedAddrGen<true> ma = {.bank_base_address = get_arg_val<uint32_t>(4), .page_size = M_PAGE};
    const uint32_t n_units = get_arg_val<uint32_t>(5);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t j = get_arg_val<uint32_t>(6 + 3 * u), r0 = get_arg_val<uint32_t>(7 + 3 * u);
        const uint32_t r1 = get_arg_val<uint32_t>(8 + 3 * u);
        cb_reserve_back(cb_f, KZ);
        cb_reserve_back(cb_g, KZ);
        for (uint32_t k = 0; k < KZ; ++k) {
            noc_async_read(fa.get_noc_addr(k * WT + j), get_write_ptr(cb_f) + k * 4096, 4096);
            noc_async_read(ga.get_noc_addr(k * WT + j), get_write_ptr(cb_g) + k * 4096, 4096);
        }
        noc_async_read_barrier();
        cb_push_back(cb_f, KZ);
        cb_push_back(cb_g, KZ);
        for (uint32_t R = r0; R < r1; ++R) {
            cb_reserve_back(cb_z, KZ);
            cb_reserve_back(cb_h, 1);
            cb_reserve_back(cb_m, 1);
            for (uint32_t k = 0; k < KZ; ++k) {
                noc_async_read(za.get_noc_addr(R * KZ + k), get_write_ptr(cb_z) + k * 2048, 2048);
            }
            noc_async_read(ha.get_noc_addr(R * WT + j), get_write_ptr(cb_h), 4096);
            noc_async_read(ma.get_noc_addr(R * WT + j), get_write_ptr(cb_m), M_PAGE);
            noc_async_read_barrier();
            cb_push_back(cb_z, KZ);
            cb_push_back(cb_h, 1);
            cb_push_back(cb_m, 1);
        }
    }
}
