// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    constexpr uint32_t wt = get_compile_time_arg_val(0), block = get_compile_time_arg_val(1);
    const uint32_t out = get_arg_val<uint32_t>(0), state = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2), start = get_arg_val<uint32_t>(3);
    constexpr auto oa = TensorAccessorArgs<2>();
    const auto output = TensorAccessor(oa, out);
#ifndef BACKWARD
    constexpr auto sa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto stats = TensorAccessor(sa, state);
#endif
    const Noc noc;
    DataflowBuffer ob(16), sb(8);
    const uint32_t bytes = get_tile_size(16), stat_bytes = get_tile_size(8);
    for (uint32_t row = start; row < start + count; ++row) {
#ifndef BACKWARD
        sb.wait_front(3);
        for (uint32_t i = 0; i < 3; ++i)
            noc.async_write(sb, stats, stat_bytes, {.offset_bytes = i * stat_bytes}, {.page_id = row * 3 + i});
        noc.async_write_barrier();
        sb.pop_front(3);
#endif
        for (uint32_t col = 0; col < wt; col += block) {
            ob.wait_front(block);
            for (uint32_t i = 0; i < block; ++i)
                noc.async_write(ob, output, bytes, {.offset_bytes = i * bytes}, {.page_id = row * wt + col + i});
            noc.async_write_barrier();
            ob.pop_front(block);
        }
    }
}
