// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime arguments - matching test_mul_reduce_scalar.cpp:100-105
    uint32_t src0_addr = get_arg_val<uint32_t>(0);     // src0_dram_buffer->address()
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);  // 0
    uint32_t src1_addr = get_arg_val<uint32_t>(2);     // src1_dram_buffer->address()
    uint32_t src1_bank_id = get_arg_val<uint32_t>(3);  // 0
    uint32_t num_tiles = get_arg_val<uint32_t>(4);     // num_tiles

    // Circular buffer indices
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    // Single tile size in bytes (FP16_b format: 32x32 * 2 bytes)
    constexpr uint32_t single_tile_size = 2 * 1024;

    // Create DRAM interfaces
    const InterleavedAddrGenFast<true> s0 = {
        .bank_base_address = src0_addr, .page_size = single_tile_size, .data_format = DataFormat::Float16_b};

    const InterleavedAddrGenFast<true> s1 = {
        .bank_base_address = src1_addr, .page_size = single_tile_size, .data_format = DataFormat::Float16_b};

    // Read tiles sequentially: reserve, read, barrier, push for each tile
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // Read tile from source 0 (input A)
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(tile_idx, s0, l1_write_addr_in0);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);

        // Read corresponding tile from source 1 (input B)
        cb_reserve_back(cb_id_in1, 1);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read_tile(tile_idx, s1, l1_write_addr_in1);
        noc_async_read_barrier();
        cb_push_back(cb_id_in1, 1);
    }
}
