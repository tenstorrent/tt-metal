// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

template <uint32_t CB, typename Accessor>
inline void write_component(Noc& noc, const Accessor& dst, uint32_t start, uint32_t batch) {
    DataflowBuffer cb(CB);
    const uint32_t bytes = get_tile_size(CB);
    cb.wait_front(batch);
    for (uint32_t j = 0; j < batch; ++j) {
        noc.async_write(cb, dst, bytes, {.offset_bytes = j * bytes}, {.page_id = start + j});
    }
    noc.async_write_barrier();
    cb.pop_front(batch);
}

void kernel_main() {
    constexpr uint32_t components = get_compile_time_arg_val(0);
    constexpr uint32_t batch = get_compile_time_arg_val(1);
    constexpr auto args0 = TensorAccessorArgs<2>();
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();
    const uint32_t addresses[3] = {get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2)};
    const uint32_t start = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    const auto dst0 = TensorAccessor(args0, addresses[0]);
    const auto dst1 = TensorAccessor(args1, addresses[1]);
    Noc noc;
    for (uint32_t i = 0; i < count; i += batch) {
        write_component<16>(noc, dst0, start + i, batch);
        write_component<17>(noc, dst1, start + i, batch);
        if constexpr (components == 3) {
            const auto dst2 = TensorAccessor(args2, addresses[2]);
            write_component<18>(noc, dst2, start + i, batch);
        }
    }
}
