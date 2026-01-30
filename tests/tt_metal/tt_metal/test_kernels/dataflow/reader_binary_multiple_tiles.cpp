// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr uint32_t single_tile_size = 2 * 1024;

    // Create TensorAccessors for the input buffers
    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, src0_addr, single_tile_size);
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, src1_addr, single_tile_size);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(tile_idx, s0, l1_write_addr_in0);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);

        cb_reserve_back(cb_id_in1, 1);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read_tile(tile_idx, s1, l1_write_addr_in1);
        noc_async_read_barrier();
        cb_push_back(cb_id_in1, 1);
    }
}
