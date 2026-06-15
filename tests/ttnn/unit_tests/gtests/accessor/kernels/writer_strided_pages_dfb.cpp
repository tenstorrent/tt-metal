// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
Strided pages DFB consumer kernel.

Pops DFB entries and writes each to its strided tensor page.
Works for WH/BH (1 thread → all pages) and Quasar (N threads → every N-th page each).
On Quasar, strided_pages() uses get_my_thread_id() / get_num_threads() directly.

Compile-time args: TensorAccessorArgs CTAs starting at index 0.
    Host passes these via the KERNEL_COMPILE_TIME_ARGS compiler define.
Runtime args (positional, accessed via get_arg_val):
    [0] output tensor base address
    [1] total number of pages
DFB binding: local accessor name "my_dfb" (CONSUMER endpoint)
*/

#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    auto args_dst = TensorAccessorArgs<0, 0>();
    uint32_t base_addr = get_arg_val<uint32_t>(0);
    uint32_t total_pages = get_arg_val<uint32_t>(1);

    auto accessor = TensorAccessor(args_dst, base_addr);
    DataflowBuffer buf(dfb::my_dfb);
    Noc noc;
    uint32_t page_size = args_dst.get_aligned_page_size();

    for (const auto& page : accessor.strided_pages(total_pages)) {
        buf.wait_front(1);
        noc.async_write(buf, page, page_size, {}, {});
        noc.async_write_barrier();
        buf.pop_front(1);
    }
    buf.finish();
    buf.write_barrier(noc);
}
