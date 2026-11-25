// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Page gather, writer: the pages pf_pages_reader.cpp puts in cb 0 to dst pages p0 .. p1 - 1, NB at a time.
// Runtime args: dst addr, p0, p1.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PB = get_named_compile_time_arg_val("pb");
    constexpr uint32_t NB = get_named_compile_time_arg_val("nb");
    constexpr uint32_t cb_t = 0;
    const InterleavedAddrGen<true> da = {.bank_base_address = get_arg_val<uint32_t>(0), .page_size = PB};
    const uint32_t p0 = get_arg_val<uint32_t>(1), p1 = get_arg_val<uint32_t>(2);
    for (uint32_t p = p0; p < p1; p += NB) {
        const uint32_t cnt = p1 - p < NB ? p1 - p : NB;
        cb_wait_front(cb_t, NB);
        const uint32_t lt = get_read_ptr(cb_t);
        for (uint32_t q = 0; q < cnt; ++q) {
            noc_async_write(lt + q * PB, da.get_noc_addr(p + q), PB);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_t, NB);
    }
    noc_async_write_barrier();
}
