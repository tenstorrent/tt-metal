// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "tensix_types.h"

// #include "debug/dprint.h"
// #include "debug/dprint_pages.h"

template <uint32_t num_of_transactions, uint32_t num_pages, uint32_t page_size_bytes, typename AddrGen>
FORCE_INLINE void noc_write_helper(const uint32_t l1_read_addr, const AddrGen& s) {
    for (uint32_t i = 0; i < num_of_transactions; i++) {
        for (uint32_t p = 0; p < num_pages; p++) {
            noc_async_write_page(p, s, l1_read_addr + p * page_size_bytes);
        }
    }
    noc_async_write_barrier();
}

// L1 to DRAM write
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_read_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(3);
    constexpr uint32_t test_id = get_compile_time_arg_val(4);
    constexpr bool sync = get_compile_time_arg_val(5) == 1;
    constexpr bool default_noc = get_compile_time_arg_val(6) == 1;

    constexpr auto dst_args = TensorAccessorArgs<7>();
    const auto s = TensorAccessor(dst_args, dst_addr, page_size_bytes);

    constexpr uint32_t transaction_size_bytes = page_size_bytes;
    DeviceTimestampedData("Number of transactions", num_of_transactions * num_pages);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    if constexpr (sync) {
        cb_wait_front(cb_id_out0, 1);
    }
    if constexpr (default_noc) {
        {
            DeviceZoneScopedN("RISCV0");
            noc_write_helper<num_of_transactions, num_pages, page_size_bytes>(l1_read_addr, s);
        }
    } else {
        {
            DeviceZoneScopedN("RISCV1");
            noc_write_helper<num_of_transactions, num_pages, page_size_bytes>(l1_read_addr, s);
        }
    }
    if constexpr (sync) {
        cb_pop_front(cb_id_out0, 1);
    }
}
