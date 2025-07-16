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

    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t test_id = get_compile_time_arg_val(4);
    constexpr bool is_dram = get_compile_time_arg_val(5) == 1;
    constexpr bool sync = get_compile_time_arg_val(6) == 1;

    const InterleavedAddrGenFast<is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float16_b};

    constexpr uint32_t transaction_size_bytes = num_of_transactions * num_tiles * tile_size_bytes;
    DeviceTimestampedData("Number of transactions", 1);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    if (sync) {
        cb_reserve_back(cb_id_in0, 1);
    }
    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            for (uint32_t t = 0; t < num_tiles; t++) {
                noc_async_read_tile(t, s, l1_write_addr + t * tile_size_bytes);
            }
        }
        noc_async_read_barrier();
    }
    if (sync) {
        cb_push_back(cb_id_in0, 1);
    }
}
