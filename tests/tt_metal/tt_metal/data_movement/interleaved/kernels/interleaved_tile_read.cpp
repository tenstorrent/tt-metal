// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "tensix_types.h"

// #include "debug/dprint.h"
// #include "debug/dprint_pages.h"

// DRAM to L1 read
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_write_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);
    constexpr bool is_dram = get_compile_time_arg_val(4) == 1;
    constexpr bool sync = get_compile_time_arg_val(5) == 1;

    const InterleavedAddrGenFast<is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float16_b};

    constexpr uint32_t transaction_size_bytes = num_tiles * tile_size_bytes;
    DeviceTimestampedData("Number of transactions", 1);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    if (sync) {
        cb_reserve_back(cb_id_in0, num_tiles);
    }
    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t tile_id = 0; tile_id < num_tiles; tile_id++) {
            noc_async_read_tile(tile_id, s, l1_write_addr);
            l1_write_addr += tile_size_bytes;
        }
        noc_async_read_barrier();
    }
    if (sync) {
        cb_push_back(cb_id_in0, num_tiles);
    }
}
