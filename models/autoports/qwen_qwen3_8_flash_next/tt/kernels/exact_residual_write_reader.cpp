// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto ba = TensorAccessorArgs<0>();
    constexpr auto ia = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    const auto block = TensorAccessor(ba, get_arg_val<uint32_t>(0), 2048);
    const auto inject = TensorAccessor(ia, get_arg_val<uint32_t>(1), 2048);
    const auto residual = TensorAccessor(ra, get_arg_val<uint32_t>(2), 2048);
    const uint32_t start = get_arg_val<uint32_t>(3), count = get_arg_val<uint32_t>(4);
    for (uint32_t tile = start; tile < start + count; ++tile) {
        const uint32_t row = tile / 80;
        cb_reserve_back(0, 1);
        cb_reserve_back(1, 1);
        cb_reserve_back(2, 1);
        noc_async_read(block.get_noc_addr(tile), get_write_ptr(0), 2048);
        noc_async_read(inject.get_noc_addr(row), get_write_ptr(1), 2048);
        noc_async_read(residual.get_noc_addr(tile), get_write_ptr(2), 2048);
        noc_async_read_barrier();
        cb_push_back(0, 1);
        cb_push_back(1, 1);
        cb_push_back(2, 1);
    }
}
