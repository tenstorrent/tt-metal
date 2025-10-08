// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
This reader kernel reads all pages from a tensor and pushes them to a circular buffer.
This kernel is expected to be executed on only one core (RISCV_0).
*/

#include <cstdint>
#include "dataflow_api.h"
#include "accessor/tensor_accessor.h"

void kernel_main() {
    auto args_src = TensorAccessorArgs<0, 0>();
    uint32_t cb_id = get_compile_time_arg_val(args_src.next_compile_time_args_offset());
    uint32_t page_size = get_compile_time_arg_val(args_src.next_compile_time_args_offset() + 1);
    uint32_t tensor_volume = get_compile_time_arg_val(args_src.next_compile_time_args_offset() + 2);
    uint32_t input_base_address = get_common_arg_val<uint32_t>(0);

    auto tensor_accessor_src = TensorAccessor(args_src, input_base_address, page_size);

    // Iterate over all pages in the tensor and read them to CB
    // For interleaved tensors, we need to pass start_page_id and end_page_id to pages()
    // For sharded tensors, pages() uses default end_page_id (tensor_volume from dspec)
    auto all_pages = [&]() {
#if INTERLEAVED_LAYOUT
        return tensor_accessor_src.pages(0, tensor_volume);
#else
        return tensor_accessor_src.pages();
#endif
    }();
    for (const auto& page : all_pages) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id);

        noc_async_read(page.noc_addr(), l1_write_addr, page_size);
        noc_async_read_barrier();

        cb_push_back(cb_id, 1);
    }
}
