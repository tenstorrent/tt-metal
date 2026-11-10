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

    const auto buffer = TensorAccessor(buf_args, buf_addr, page_bytes);
    const uint32_t l1 = get_write_ptr(buf_cb);
    uint32_t prev = get_timestamp_32b(), elapsed = 0, p = 0;
    while ((invalidate_l1_cache(), *stop) < stop_count && elapsed < timeout) {
        for (uint32_t b = 0; b < 4; ++b, p = p + 1 == num_pages ? 0 : p + 1) {
            if constexpr (is_writer) {
                noc_async_write(l1, buffer.get_noc_addr(first_page + p), page_bytes);
            } else {
                noc_async_read(buffer.get_noc_addr(first_page + p), l1, page_bytes);
            }
        }
        if constexpr (is_writer) {
            noc_async_write_barrier();
        } else {
            noc_async_read_barrier();
        }
        const uint32_t now = get_timestamp_32b();
        elapsed += now - prev;
        prev = now;
    }
    if (*stop >= stop_count) {
        noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(stop)), 0u - stop_count);
        noc_async_atomic_barrier();
    }
}
