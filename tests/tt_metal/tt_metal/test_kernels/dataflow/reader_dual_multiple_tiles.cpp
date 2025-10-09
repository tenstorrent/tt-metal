// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // Runtime arguments
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_bank = get_arg_val<uint32_t>(1);
    uint32_t src0_tiles = get_arg_val<uint32_t>(2);
    uint32_t src1_addr = get_arg_val<uint32_t>(3);
    uint32_t src1_bank = get_arg_val<uint32_t>(4);
    uint32_t src1_tiles = get_arg_val<uint32_t>(5);

    // Circular buffer indices
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    // Single tile size in bytes (FP16_b format)
    constexpr uint32_t single_tile_size = 2 * 1024;

    // Create DRAM interface for source 0
    const InterleavedAddrGenFast<true> s0 = {
        .bank_base_address = src0_addr, .page_size = single_tile_size, .data_format = DataFormat::Float16_b};

    // Create DRAM interface for source 1
    const InterleavedAddrGenFast<true> s1 = {
        .bank_base_address = src1_addr, .page_size = single_tile_size, .data_format = DataFormat::Float16_b};

    // Read tiles for both sources
    // We read src0_tiles tiles from source 0 and src1_tiles tiles from source 1
    // These should be the same number (8 tiles each)

    for (uint32_t tile_idx = 0; tile_idx < src0_tiles; ++tile_idx) {
        // Read tile from source 0
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(tile_idx, s0, l1_write_addr_in0);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);

        // Read corresponding tile from source 1
        cb_reserve_back(cb_id_in1, 1);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read_tile(tile_idx, s1, l1_write_addr_in1);
        noc_async_read_barrier();
        cb_push_back(cb_id_in1, 1);
    }
}
