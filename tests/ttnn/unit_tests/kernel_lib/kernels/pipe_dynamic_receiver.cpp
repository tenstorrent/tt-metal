// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr auto mc = McastArgs</*CT=*/1, /*RT=*/2>();
    constexpr uint32_t SCALARS = mc.next_compile_time_args_offset();
    constexpr uint32_t num_iters = get_compile_time_arg_val(SCALARS + 0);
    constexpr uint32_t total_pages = get_compile_time_arg_val(SCALARS + 1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(SCALARS + 2);
    constexpr auto out_args = TensorAccessorArgs<SCALARS + 3>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_start_page = get_arg_val<uint32_t>(1);
    const auto output = TensorAccessor(out_args, output_addr);

    Noc noc;
    CircularBuffer scratch(cb);
    scratch.reserve_back(total_pages);
    const uint32_t base = scratch.get_write_ptr();
    auto pipe = mc.receiver(noc);

    uint32_t page_offset = 0;
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        const uint32_t pages = iter + 1;
        const uint32_t byte_offset = page_offset * page_bytes;
        pipe.receive(base + byte_offset, pages * page_bytes);
        for (uint32_t page = 0; page < pages; ++page) {
            noc.async_write(
                scratch,
                output,
                page_bytes,
                {.offset_bytes = byte_offset + page * page_bytes},
                {.page_id = output_start_page + page_offset + page});
        }
        noc.async_write_barrier();
        page_offset += pages;
    }
}
