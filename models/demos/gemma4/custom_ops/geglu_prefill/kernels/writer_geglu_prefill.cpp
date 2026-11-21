// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer for GegluPrefill: the B output tiles of unit `id` are output pages id * B .. id * B + B - 1 (the output
// tile grid is row-major and B divides its width), one write barrier per unit.
// compile-time args: [cb_out, B] + TensorAccessorArgs(out); runtime args: [out_addr, start unit, unit count]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    const uint32_t page = get_local_cb_interface(cb_out).fifo_page_size;
    const auto s_out = TensorAccessor(out_args, out_addr, page);
    Noc noc;
    CircularBuffer cbo(cb_out);
    for (uint32_t id = start; id < start + count; ++id) {
        const uint32_t p = id * B;
        cbo.wait_front(B);
        for (uint32_t i = 0; i < B; ++i) {
            noc.async_write(cbo, s_out, page, {.offset_bytes = i * page}, {.page_id = p + i});
        }
        noc.async_write_barrier();
        cbo.pop_front(B);
    }
}
