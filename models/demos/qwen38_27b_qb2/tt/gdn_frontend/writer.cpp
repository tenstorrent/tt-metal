// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0), 2048);
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1), 2048);
    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(2), 2048);
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t stride = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t tile = first + item * stride;
        cb_wait_front(4, 1);
        if (tile < 16) {
            noc_async_write_tile(tile, q, get_read_ptr(4));
        } else if (tile < 32) {
            noc_async_write_tile(tile - 16, k, get_read_ptr(4));
        } else {
            noc_async_write_tile(tile - 32, v, get_read_ptr(4));
        }
        noc_async_write_barrier();
        cb_pop_front(4, 1);
    }
}
