// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "tensix_types.h"

// #include "debug/dprint.h"
// #include "debug/dprint_pages.h"

template <uint32_t num_of_transactions, uint32_t num_tiles, uint32_t tile_size_bytes, bool is_dram>
FORCE_INLINE void noc_write_helper(const uint32_t l1_read_addr, const InterleavedAddrGenFast<is_dram>& s) {
    for (uint32_t i = 0; i < num_of_transactions; i++) {
        for (uint32_t t = 0; t < num_tiles; t++) {
            noc_async_write_tile(t, s, l1_read_addr + t * tile_size_bytes);
        }
    }
    noc_async_write_barrier();
}

// L1 to DRAM write
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_read_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(3);
    constexpr uint32_t test_id = get_compile_time_arg_val(4);
    constexpr bool is_dram = get_compile_time_arg_val(5) == 1;
    constexpr bool sync = get_compile_time_arg_val(6) == 1;
    constexpr bool default_noc = get_compile_time_arg_val(7) == 1;

    const InterleavedAddrGenFast<is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float16_b};

    constexpr uint32_t transaction_size_bytes = num_of_transactions * num_tiles * tile_size_bytes;
    DeviceTimestampedData("Number of transactions", 1);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    if constexpr (sync) {
        cb_wait_front(cb_id_out0, 1);
    }
    if constexpr (default_noc) {
        {
            DeviceZoneScopedN("RISCV0");
            noc_write_helper<num_of_transactions, num_tiles, tile_size_bytes, is_dram>(l1_read_addr, s);
        }
    } else {
        {
            DeviceZoneScopedN("RISCV1");
            noc_write_helper<num_of_transactions, num_tiles, tile_size_bytes, is_dram>(l1_read_addr, s);
        }
    }
    if constexpr (sync) {
        cb_pop_front(cb_id_out0, 1);
    }
}
