// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
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
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;  // Input tensor 0
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;  // Input tensor 1

    // Single tile size in bytes (FP16_B format)
    constexpr uint32_t single_tile_size = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    // Setup NOC for reading from DRAM
    const InterleavedAddrGenFast<true> src0_addrg = {
        .bank_base_address = src0_addr, .page_size = single_tile_size, .data_format = data_format};

    const InterleavedAddrGenFast<true> src1_addrg = {
        .bank_base_address = src1_addr, .page_size = single_tile_size, .data_format = data_format};

    // Read both input tensors tile by tile
    // Since this is a fused operation, we read corresponding tiles from both tensors
    // at the same time to feed the eltwise binary operation

    for (uint32_t tile_idx = 0; tile_idx < src0_tiles; ++tile_idx) {
        // Read tile from input tensor 0
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(tile_idx, src0_addrg, l1_write_addr_in0);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);

        // Read corresponding tile from input tensor 1
        cb_reserve_back(cb_id_in1, 1);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read_tile(tile_idx, src1_addrg, l1_write_addr_in1);
        noc_async_read_barrier();
        cb_push_back(cb_id_in1, 1);
    }
}
