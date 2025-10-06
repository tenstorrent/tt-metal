// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "tensix_types.h"
#include "accessor/tensor_accessor.h"

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

    // Tensor accessor compile time args appended to kernel's compile time args
    // so the index is offset to start at 6
    auto args = TensorAccessorArgs<6>();
    auto s = TensorAccessor(args, dst_addr, page_size_bytes);

    constexpr uint32_t transaction_size_bytes = page_size_bytes;
    DeviceTimestampedData("Number of transactions", num_of_transactions * num_pages);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            for (uint32_t p = 0; p < num_pages; p++) {
                if constexpr (sync) {
                    cb_wait_front(cb_id_out0, 1);
                }
                uint64_t noc_addr = s.get_noc_addr(p);
                noc_async_write(l1_read_addr + p * page_size_bytes, noc_addr, page_size_bytes);
                if constexpr (sync) {
                    noc_async_write_barrier();
                    cb_pop_front(cb_id_out0, 1);
                }
            }
        }
        if constexpr (!sync) {
            noc_async_write_barrier();
        }
    }
}
