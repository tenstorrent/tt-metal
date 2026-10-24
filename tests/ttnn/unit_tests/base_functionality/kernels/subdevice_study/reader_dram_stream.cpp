// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streams pages [start, start + num_pages) of an interleaved DRAM tensor into cb_data, `passes` times,
// `batch` pages per CB push (num_pages must be a multiple of batch).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);
    const uint32_t passes = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_data = get_named_compile_time_arg_val("cb_data");
    constexpr uint32_t batch = get_named_compile_time_arg_val("batch");
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr);

    for (uint32_t p = 0; p < passes; ++p) {
        for (uint32_t i = 0; i < num_pages; i += batch) {
            cb_reserve_back(cb_data, batch);
            uint32_t l1 = get_write_ptr(cb_data);
            for (uint32_t j = 0; j < batch; ++j) {
                noc_async_read(src.get_noc_addr(start + i + j), l1, page_size);
                l1 += page_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_data, batch);
        }
    }
}
