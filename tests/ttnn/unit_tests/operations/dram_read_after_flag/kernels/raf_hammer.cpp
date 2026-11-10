// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Background DRAM load for test_dram_read_after_flag_cross_noc.py: streams 8 KB writes (BRISC, NOC0) or
// reads (NCRISC, NOC1) over this core's slice of a separate interleaved DRAM buffer (every bank)
// until each probe receiver on this chip has signalled `stop` once, then takes the signals off.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr bool is_writer = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t buf_cb = get_compile_time_arg_val(2);
    constexpr auto buf_args = TensorAccessorArgs<3>();

    const uint32_t buf_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3));
    const uint32_t stop_count = get_arg_val<uint32_t>(4);
    const uint32_t timeout = get_arg_val<uint32_t>(5);  // safety bound for the whole launch
    const uint32_t page_mult = get_arg_val<uint32_t>(6);  // buffer page = (first_page + p) * page_mult + page_add:
    const uint32_t page_add = get_arg_val<uint32_t>(7);   // (num banks, bank) keeps the load on one bank
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(8));

    const auto buffer = TensorAccessor(buf_args, buf_addr, page_bytes);
    const uint32_t l1 = get_write_ptr(buf_cb);
    uint32_t prev = get_timestamp_32b(), elapsed = 0, p = 0, transfers = 0;
    while ((invalidate_l1_cache(), *stop) < stop_count && elapsed < timeout) {
        for (uint32_t b = 0; b < 4; ++b, p = p + 1 == num_pages ? 0 : p + 1) {
            if constexpr (is_writer) {
                noc_async_write(l1, buffer.get_noc_addr((first_page + p) * page_mult + page_add), page_bytes);
            } else {
                noc_async_read(  // own slot per outstanding read
                    buffer.get_noc_addr((first_page + p) * page_mult + page_add), l1 + b * page_bytes, page_bytes);
            }
        }
        transfers += 4;
        if constexpr (is_writer) {
            noc_async_write_barrier();
        } else {
            noc_async_read_barrier();
        }
        const uint32_t now = get_timestamp_32b();
        elapsed += now - prev;
        prev = now;
    }
    // results (writer words 0..2, reader 4..6): transfers lo / hi, launches that hit the safety timeout
    volatile tt_l1_ptr uint32_t* r = res + (is_writer ? 0 : 4);
    const uint64_t t = (static_cast<uint64_t>(r[1]) << 32 | r[0]) + transfers;
    r[0] = static_cast<uint32_t>(t), r[1] = static_cast<uint32_t>(t >> 32);
    r[2] += *stop >= stop_count ? 0 : 1;
    if (*stop >= stop_count) {
        noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(stop)), 0u - stop_count);
        noc_async_atomic_barrier();
    }
}
