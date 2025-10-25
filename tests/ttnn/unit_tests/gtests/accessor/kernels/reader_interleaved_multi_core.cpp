// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
This reader kernel reads a subset of pages from an interleaved tensor and pushes them to a circular buffer.
Each core handles a specific range of pages [start_page_id, end_page_id).
*/

#include <cstdint>
#include "dataflow_api.h"
#include "accessor/tensor_accessor.h"

void kernel_main() {
    auto args_src = TensorAccessorArgs<0, 0>();
    uint32_t cb_id = get_compile_time_arg_val(args_src.next_compile_time_args_offset());
    uint32_t page_size = get_compile_time_arg_val(args_src.next_compile_time_args_offset() + 1);

    uint32_t input_base_address = get_arg_val<uint32_t>(0);
    uint32_t start_page_id = get_arg_val<uint32_t>(1);
    uint32_t end_page_id = get_arg_val<uint32_t>(2);

    auto tensor_accessor_src = TensorAccessor(args_src, input_base_address, page_size);

    auto process_pages = [&](const auto& page) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read(page.noc_addr(), l1_write_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    };

#ifdef BIG_STEP
    for (uint32_t start_offset = 0; start_offset < BIG_STEP; ++start_offset) {
        auto pages = tensor_accessor_src.pages(start_page_id + start_offset, end_page_id);
        auto page = pages.begin();
        for (; page != pages.end(); page += BIG_STEP) {
            // DPRINT << "write " << page->page_id() << " to " << page->noc_addr() << ENDL();
            process_pages(*page);
        }
    }

#else
    // Iterate over the assigned page range for this core
    auto pages = tensor_accessor_src.pages(start_page_id, end_page_id);
    for (const auto& page : pages) {
        // DPRINT << "write " << page.page_id() << " to " << page.noc_addr() << ENDL();
        process_pages(page);
    }
#endif
}
