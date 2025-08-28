// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
This writer kernel pops pages from a circular buffer and writes them to a tensor.
This kernel is expected to be executed on only one core (RISCV_1).
*/

#include <cstdint>
#include "dataflow_api.h"
#include "accessor/tensor_accessor.h"

void kernel_main() {
    auto args_dst = TensorAccessorArgs<0, 0>();
    uint32_t cb_id = get_compile_time_arg_val(args_dst.next_compile_time_args_offset());
    uint32_t page_size = get_compile_time_arg_val(args_dst.next_compile_time_args_offset() + 1);
    uint32_t output_base_address = get_common_arg_val<uint32_t>(0);

    auto tensor_accessor_dst = TensorAccessor(args_dst, output_base_address, page_size);

    // Iterate over all pages in the tensor and write them from CB
    auto all_pages = tensor_accessor_dst.pages();
    for (const auto& page : all_pages) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);

        noc_async_write(l1_read_addr, page.get_noc_addr(), page_size);
        noc_async_write_barrier();

        cb_pop_front(cb_id, 1);
    }
}
