// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Page gather, reader: pages src[idx[p]] for p in [p0, p1) into cb 0, NB at a time (pf_pages_writer.cpp writes them
// to dst page p). idx: uint32 ROW_MAJOR DRAM, 256 indices per 1 KB page, read into cb 1.
// Runtime args: src addr, idx addr, p0, p1.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PB = get_named_compile_time_arg_val("pb");  // page bytes
    constexpr uint32_t NB = get_named_compile_time_arg_val("nb");  // pages per batch
    constexpr uint32_t cb_t = 0, cb_i = 1;
    const uint32_t li = get_write_ptr(cb_i);
    const InterleavedAddrGen<true> sa = {.bank_base_address = get_arg_val<uint32_t>(0), .page_size = PB};
    const InterleavedAddrGen<true> ia = {.bank_base_address = get_arg_val<uint32_t>(1), .page_size = 1024};
    const uint32_t p0 = get_arg_val<uint32_t>(2), p1 = get_arg_val<uint32_t>(3);
    const volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(li);
    uint32_t loaded = 0xffffffff;
    for (uint32_t p = p0; p < p1; p += NB) {
        const uint32_t cnt = p1 - p < NB ? p1 - p : NB;
        cb_reserve_back(cb_t, NB);
        const uint32_t lt = get_write_ptr(cb_t);
        for (uint32_t q = p; q < p + cnt; ++q) {
            if ((q >> 8) != loaded) {
                noc_async_read_barrier();  // reads of the old index page's pages issued: their addresses are taken
                loaded = q >> 8;
                noc_async_read(ia.get_noc_addr(loaded), li, 1024);
                noc_async_read_barrier();
            }
            noc_async_read(sa.get_noc_addr(idx[q & 255]), lt + (q - p) * PB, PB);
        }
        noc_async_read_barrier();
        cb_push_back(cb_t, NB);
    }
}
