// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src0_num_tiles = get_arg_val<uint32_t>(2);
    uint32_t src1_addr = get_arg_val<uint32_t>(3);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(4);
    uint32_t src1_num_tiles = get_arg_val<uint32_t>(5);
    uint32_t scaler_value = get_arg_val<uint32_t>(6);  // Scaler for reduce operation

    constexpr uint32_t cb_id_in0 = 0;     // First input for eltwise
    constexpr uint32_t cb_id_in1 = 1;     // Second input for eltwise
    constexpr uint32_t cb_id_scaler = 2;  // Scaler for reduce

    // Generate scaler tile first (one-time setup for reduce operation)
    if (scaler_value != 0) {
        cb_reserve_back(cb_id_scaler, 1);

        // Fill tile with zeros first
        constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
        uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
        uint32_t write_addr = get_write_ptr(cb_id_scaler);

        for (uint32_t i = 0; i < num_zeros_reads; ++i) {
            noc_async_read(zeros_noc_addr, write_addr, MEM_ZEROS_SIZE);
            write_addr += MEM_ZEROS_SIZE;
        }
        noc_async_read_barrier();

        // Set scaler values in first row of each face
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id_scaler));
        uint32_t idx = 0;
        for (uint32_t k = 0; k < 4; ++k) {  // 4 faces per tile
            uint32_t curr_idx = idx;
            for (uint32_t j = 0; j < 8; ++j) {  // 8 values per row (16 bfloat16 values = 8 uint32_t)
                ptr[curr_idx] = scaler_value;
                curr_idx++;
            }
            idx += 128;  // Move to next face (128 uint32_t = 256 bfloat16 values per face)
        }

        cb_push_back(cb_id_scaler, 1);
    }

    // Read input tiles for eltwise binary operation
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);
    constexpr uint32_t ublock_size_tiles = 1;

    // Read tiles from both sources simultaneously
    for (uint32_t i = 0; i < src0_num_tiles; i += ublock_size_tiles) {
        // Get NOC addresses for both sources
        uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_addr);
        uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(src1_bank_id, src1_addr);

        // Reserve space in both circular buffers
        cb_reserve_back(cb_id_in0, ublock_size_tiles);
        cb_reserve_back(cb_id_in1, ublock_size_tiles);

        // Get write pointers
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        // Start async reads for both inputs
        noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
        noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);

        // Wait for both reads to complete
        noc_async_read_barrier();

        // Push tiles to compute
        cb_push_back(cb_id_in0, ublock_size_tiles);
        cb_push_back(cb_id_in1, ublock_size_tiles);

        // Advance to next tiles
        src0_addr += ublock_size_bytes_0;
        src1_addr += ublock_size_bytes_1;
    }
}
