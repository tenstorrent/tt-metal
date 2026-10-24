// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer for RopeDecode: per work unit (row tile rt, column c < half_t) the two output tiles (rt, c) and
// (rt, c + half_t), in x's page order.
//
// compile-time args: [cb_out, cols_t, half_t] + TensorAccessorArgs(out)
// runtime args: [out_addr, start unit, unit count]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cols_t = get_compile_time_arg_val(1);
    constexpr uint32_t half_t = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();
    const uint32_t page = get_local_cb_interface(cb_out).fifo_page_size;
    const auto s_out = TensorAccessor(out_args, out_addr, page);
    Noc noc;
    CircularBuffer cbo(cb_out);
    for (uint32_t id = start; id < start + count; ++id) {
        const uint32_t rt = id / half_t;
        const uint32_t c = id - rt * half_t;
        const uint32_t p = rt * cols_t + c;
        cbo.wait_front(2);
        noc.async_write(cbo, s_out, page, {.offset_bytes = 0}, {.page_id = p});
        noc.async_write(cbo, s_out, page, {.offset_bytes = page}, {.page_id = p + half_t});
        noc.async_write_barrier();
        cbo.pop_front(2);
    }
}
