// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "tensix_types.h"

// #include "debug/dprint_pages.h"

// L1 to DRAM write
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);
    constexpr bool is_dram = get_compile_time_arg_val(4) == 1;

    constexpr uint32_t tile_bytes = get_tile_size(cb_id_out0);
    constexpr DataFormat data_format = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    constexpr uint32_t transaction_size_bytes = num_tiles * tile_size_bytes;
    DeviceTimestampedData("Number of transactions", 1);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    cb_wait_front(cb_id_out0, num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t tile_id = 0; tile_id < num_tiles; tile_id++) {
            noc_async_write_tile(tile_id, s, l1_read_addr);
            // tt::data_movement::common::print_bf16_pages(l1_read_addr, 32 * 32, 1);
            l1_read_addr += tile_size_bytes;
        }
        noc_async_write_barrier();
    }
    cb_pop_front(cb_id_out0, num_tiles);
}
