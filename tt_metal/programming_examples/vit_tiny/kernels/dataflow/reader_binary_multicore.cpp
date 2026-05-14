// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multicore binary reader: reads tiles from offset for both inputs.
// Runtime args: src0_addr, src1_addr, tile_start, tile_count

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    uint32_t tile_count = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;

    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, src0_addr);
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, src1_addr);

    for (uint32_t i = 0; i < tile_count; i++) {
        uint32_t tile_id = tile_start + i;
        cb_reserve_back(cb_in0, 1);
        uint32_t l1_addr0 = get_write_ptr(cb_in0);
        noc_async_read_tile(tile_id, s0, l1_addr0);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);

        cb_reserve_back(cb_in1, 1);
        uint32_t l1_addr1 = get_write_ptr(cb_in1);
        noc_async_read_tile(tile_id, s1, l1_addr1);
        noc_async_read_barrier();
        cb_push_back(cb_in1, 1);
    }
}
