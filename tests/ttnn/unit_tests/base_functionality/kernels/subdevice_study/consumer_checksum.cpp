// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Drains cb_data and checksums what arrives: word 0 of every page (full_check = 0, cheap enough to keep the
// stream DRAM-bound) or every word with a position weight (full_check = 1). Writes
// [t_start, t_end, sum, weighted_sum, pages_seen] to page `page` of the stats tensor.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint64_t t_start = get_timestamp();
    const uint32_t stats_addr = get_arg_val<uint32_t>(0);
    const uint32_t page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);
    const uint32_t passes = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_data = get_named_compile_time_arg_val("cb_data");
    constexpr uint32_t cb_stats = get_named_compile_time_arg_val("cb_stats");
    constexpr uint32_t batch = get_named_compile_time_arg_val("batch");
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    constexpr uint32_t full_check = get_named_compile_time_arg_val("full_check");
    constexpr uint32_t words_per_page = page_size / sizeof(uint32_t);
    constexpr auto st_args = TensorAccessorArgs<0>();
    const auto st = TensorAccessor(st_args, stats_addr);

    uint32_t sum = 0;
    uint32_t wsum = 0;
    uint32_t seen = 0;
    for (uint32_t p = 0; p < passes; ++p) {
        for (uint32_t i = 0; i < num_pages; i += batch) {
            cb_wait_front(cb_data, batch);
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_data));
            for (uint32_t j = 0; j < batch; ++j) {
                if constexpr (full_check) {
                    for (uint32_t k = 0; k < words_per_page; ++k) {
                        const uint32_t v = w[j * words_per_page + k];
                        sum += v;
                        wsum += v * (k + 1);
                    }
                } else {
                    sum += w[j * words_per_page];
                }
            }
            seen += batch;
            cb_pop_front(cb_data, batch);
        }
    }
    const uint64_t t_end = get_timestamp();

    cb_reserve_back(cb_stats, 1);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_stats));
    for (uint32_t i = 0; i < 16; ++i) {
        s[i] = 0;
    }
    s[0] = static_cast<uint32_t>(t_start);
    s[1] = static_cast<uint32_t>(t_start >> 32);
    s[2] = static_cast<uint32_t>(t_end);
    s[3] = static_cast<uint32_t>(t_end >> 32);
    s[4] = sum;
    s[5] = wsum;
    s[6] = seen;
    noc_async_write(get_write_ptr(cb_stats), st.get_noc_addr(page), 64);
    noc_async_write_barrier();
}
