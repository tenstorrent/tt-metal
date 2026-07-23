// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar DM-core cache-write performance kernel.
// Writes `size_bytes` to Tensix L1 either via the uncached port (+4MB alias)
// or via cacheable memory followed by flush_l2_cache_range, timing the region.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"  // pulls in DeviceZoneScopedN / DeviceTimestampedData
#include "dev_mem_map.h"                // MEM_L1_UNCACHED_BASE
#include "experimental/kernel_args.h"   // get_arg(args::name)
#include "risc_common.h"                // flush_l2_cache_range

void kernel_main() {
    std::uint32_t base_addr = get_arg(args::base_addr);
    std::uint32_t size_bytes = get_arg(args::size_bytes);
    std::uint32_t write_path = get_arg(args::write_path);  // 0=uncached, 1=cached+flush
    std::uint32_t test_id = get_arg(args::test_id);

    std::uint32_t dst_addr = (write_path == 0) ? (base_addr + MEM_L1_UNCACHED_BASE) : base_addr;
    volatile std::uint8_t* dst = (volatile std::uint8_t*)(uintptr_t)dst_addr;

    {
        DeviceZoneScopedN("RISCV1");
        for (std::uint32_t i = 0; i < size_bytes; i++) {
            dst[i] = (std::uint8_t)(i & 0xFF);
        }
        if (write_path == 1) {
            flush_l2_cache_range(base_addr, size_bytes);
        }
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", 1);
    DeviceTimestampedData("Transaction size in bytes", size_bytes);
    DeviceTimestampedData("Write path", write_path);
}
