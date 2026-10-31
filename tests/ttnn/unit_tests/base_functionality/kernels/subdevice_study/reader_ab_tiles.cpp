// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads tile 0 (A) into cb_a and tile 1 (B) into cb_b, once.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_a = get_named_compile_time_arg_val("cb_a");
    constexpr uint32_t cb_b = get_named_compile_time_arg_val("cb_b");
    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr);
    const uint32_t tile_bytes = get_tile_size(cb_a);

    cb_reserve_back(cb_a, 1);
    cb_reserve_back(cb_b, 1);
    noc_async_read(src.get_noc_addr(0), get_write_ptr(cb_a), tile_bytes);
    noc_async_read(src.get_noc_addr(1), get_write_ptr(cb_b), tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_a, 1);
    cb_push_back(cb_b, 1);
}
