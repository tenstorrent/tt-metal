// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// Custom reader that:
// - Loads one tile from B into CB c_1 once
// - Streams A tiles into CB c_0 for num_tiles
// Compile-time args (host provides via TensorAccessorArgs):
//   A TensorAccessorArgs, then B TensorAccessorArgs
// Runtime args:
//   0: A_dram_addr
//   1: B_dram_addr
//   2: num_tiles
//   3: tile_size_bytes (e.g., 32*32*2 for bfloat16)
void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t tile_size_bytes = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;  // A stream
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;  // B single tile

    const uint32_t a_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t b_tile_bytes = get_tile_size(cb_id_in1);

    // Set up TensorAccessors from compile-time args
    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto acc_a = TensorAccessor(a_args, a_addr, a_tile_bytes);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto acc_b = TensorAccessor(b_args, b_addr, b_tile_bytes);

    // Load B's single tile once into c_1 (do not pop later; compute will reuse it)
    cb_reserve_back(cb_id_in1, 1);
    {
        const uint32_t l1_write_addr = get_write_ptr(cb_id_in1);
        noc_async_read_tile(/*tile_idx=*/0, acc_b, l1_write_addr);
        noc_async_read_barrier();
    }
    cb_push_back(cb_id_in1, 1);

    // Stream A tiles into c_0
    cb_reserve_back(cb_id_in0, 1);
    uint32_t last_trid = 1;
    {
        const uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile_dram_sharded_set_trid(last_trid, 0);
        noc_async_read_tile(/*tile_idx=*/0, acc_a, l1_write_addr);
    }

    // Pipeline reads using 2 TRIDS.
    for (uint32_t i = 1; i < num_tiles; ++i) {
        cb_reserve_back(cb_id_in0, 2);
        const uint32_t l1_write_addr = get_write_ptr(cb_id_in0) + tile_size_bytes;
        uint32_t next_trid = 3 - last_trid;  // 1 to 2, 2 to 1
        noc_async_read_tile_dram_sharded_set_trid(next_trid);
        noc_async_read_tile(/*tile_idx=*/i, acc_a, l1_write_addr);
        noc_async_read_barrier_with_trid(last_trid);
        last_trid = next_trid;
        cb_push_back(cb_id_in0, 1);
    }

    noc_async_read_barrier_with_trid(last_trid);
    cb_push_back(cb_id_in0, 1);
}
