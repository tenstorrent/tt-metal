// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t nt = get_compile_time_arg_val(0);
    constexpr auto aa = TensorAccessorArgs<1>();
    auto dst = TensorAccessor(aa, get_arg_val<uint32_t>(0));
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t bytes = get_tile_size(16);
    Noc noc;
    DataflowBuffer cb(16);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t p = start + i;
        const uint32_t head = p / (nt * 4);
        const uint32_t token_tile = (p % (nt * 4)) / 4;
        const uint32_t channel_tile = p % 4;
        const uint32_t target = head * nt * 4 + channel_tile * nt + token_tile;
        cb.wait_front(1);
        noc.async_write(cb, dst, bytes, {}, {.page_id = target});
        noc.async_write_barrier();
        cb.pop_front(1);
    }
}
