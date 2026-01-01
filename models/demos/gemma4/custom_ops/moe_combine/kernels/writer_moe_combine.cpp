// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer for MoeCombine: output column tile c -> page c of the [1, 1, 32, H] output.
// compile-time args: [cb_out] + TensorAccessorArgs(out); runtime args: [out_addr, start column, column count]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    const uint32_t page = get_local_cb_interface(cb_out).fifo_page_size;
    const auto s_out = TensorAccessor(out_args, out_addr, page);
    Noc noc;
    CircularBuffer cbo(cb_out);
    for (uint32_t c = start; c < start + count; ++c) {
        cbo.wait_front(1);
        noc.async_write(cbo, s_out, page, {.offset_bytes = 0}, {.page_id = c});
        noc.async_write_barrier();
        cbo.pop_front(1);
    }
}
