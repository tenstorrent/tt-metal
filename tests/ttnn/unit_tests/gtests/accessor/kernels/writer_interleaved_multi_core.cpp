// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
This writer kernel pops pages from a circular buffer and writes them to a subset of pages in an interleaved tensor.
Each core handles a specific range of pages [start_page_id, end_page_id).
*/

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    auto args_dst = TensorAccessorArgs<0, 0>();
    uint32_t cb_id = get_compile_time_arg_val(args_dst.next_compile_time_args_offset());
    uint32_t page_size = get_compile_time_arg_val(args_dst.next_compile_time_args_offset() + 1);

    uint32_t output_base_address = get_arg_val<uint32_t>(0);
    uint32_t start_page_id = get_arg_val<uint32_t>(1);
    uint32_t end_page_id = get_arg_val<uint32_t>(2);

    auto tensor_accessor_dst = TensorAccessor(args_dst, output_base_address);

    auto process_pages = [&](const auto& page) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        noc_async_write(l1_read_addr, page.noc_addr(), page_size);
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    };

#ifdef BIG_STEP
    for (uint32_t start_offset = 0; start_offset < BIG_STEP; ++start_offset) {
        auto pages = tensor_accessor_dst.pages(start_page_id + start_offset, end_page_id);
        auto page = pages.begin();
        for (; page != pages.end(); page += BIG_STEP) {
            DPRINT("write page {} to {}\n", page->page_id(), page->noc_addr());
            process_pages(*page);
        }
    }
#elif defined(ITER_STRIDE)
    // The reader's page order, each page written to the address get_noc_addr gives it (not the iterator's), so a
    // wrong iterator address shows up in the output.
    for (uint32_t start_offset = 0; start_offset < ITER_STRIDE; ++start_offset) {
        for (uint32_t page_id = start_page_id + start_offset; page_id < end_page_id; page_id += ITER_STRIDE) {
            cb_wait_front(cb_id, 1);
            noc_async_write(get_read_ptr(cb_id), tensor_accessor_dst.get_noc_addr(page_id), page_size);
            noc_async_write_barrier();
            cb_pop_front(cb_id, 1);
        }
    }
#else
    // Iterate over the assigned page range for this core
    auto pages = tensor_accessor_dst.pages(start_page_id, end_page_id);
    for (const auto& page : pages) {
        DPRINT("write page {} to {}\n", page.page_id(), page.noc_addr());
        process_pages(page);
    }
#endif
}
