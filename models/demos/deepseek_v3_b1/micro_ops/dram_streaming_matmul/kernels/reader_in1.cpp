// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

/**
 * Simplified in1 reader for DRAM streaming matmul.
 *
 * Reads in1 from DRAM one Kx1 column stick at a time.
 * in1 is pre-shuffled on host so K tiles are contiguous for each N column.
 *
 * Output CB is tensor-backed - just waits for compute to finish.
 */
void kernel_main() {
    // Compile time args
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);
    constexpr uint32_t in1_tensor_addr = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(3);
    constexpr uint32_t per_core_N = get_compile_time_arg_val(4);
    constexpr uint32_t in1_tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t out_num_tiles = get_compile_time_arg_val(6);

    // Runtime args (per-core values)
    const uint32_t dram_bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);

    // in1 stick size: K tiles contiguous for each N column
    constexpr uint32_t in1_stick_bytes = num_tiles_k * in1_tile_bytes;

    // Setup DRAM read for in1
    uint64_t in1_base_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, in1_tensor_addr);

    // Stream in1: one Kx1 stick at a time, per_core_N times
    uint32_t in1_read_offset = 0;

    for (uint32_t n = 0; n < per_core_N; n++) {
        cb_reserve_back(cb_id_in1, num_tiles_k);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        // Read K tiles for this N column (contiguous in shuffled layout)
        uint64_t in1_addr = in1_base_addr + in1_read_offset;
        noc_async_read(in1_addr, l1_write_addr_in1, in1_stick_bytes);
        noc_async_read_barrier();

        cb_push_back(cb_id_in1, num_tiles_k);

        in1_read_offset += in1_stick_bytes;
    }

    // Wait for compute to finish writing all output tiles
    // CB4 is backed by output tensor - data goes directly there
    cb_wait_front(cb_id_out, out_num_tiles);
}
