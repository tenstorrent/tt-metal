// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar DM-core cache-write performance kernel.
// Writes `size_bytes` to Tensix L1 and times the region, in one of three modes
// (the "Write path" arg):
//   0 = Uncached, 1-byte stores   -> uncached port (+4MB alias), byte-at-a-time
//   1 = Uncached, 8-byte stores   -> uncached port, 64-bit stores + byte tail
//   2 = Cached+Flush, 8-byte      -> cacheable write (64-bit) then flush_l2_cache_range
// Modes 1 and 2 use the natural 64-bit DM-core store width (8B) for the bulk with
// a byte tail for any sub-8B remainder (sizes 1/2/4). Stores are volatile so the
// compiler cannot coalesce/reorder them.
//
// The write(+flush) region is repeated `num_iterations` times INSIDE the timed
// DeviceZoneScopedN, and the host divides the zone duration by num_iterations
// (stamped as "Number of transactions"). This amortizes fixed per-run overhead
// (zone entry/exit, prologue, cold pipeline) so the reported value is the
// steady-state per-write cost, matching the April two-phase measurement method.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"  // pulls in DeviceZoneScopedN / DeviceTimestampedData
#include "dev_mem_map.h"                // MEM_L1_UNCACHED_BASE
#include "experimental/kernel_args.h"   // get_arg(args::name)
#include "risc_common.h"                // flush_l2_cache_range

void kernel_main() {
    std::uint32_t base_addr = get_arg(args::base_addr);
    std::uint32_t size_bytes = get_arg(args::size_bytes);
    std::uint32_t mode = get_arg(args::write_path);  // 0=uncached 1B, 1=uncached 8B, 2=cached+flush 8B
    std::uint32_t num_iterations = get_arg(args::num_iterations);
    std::uint32_t test_id = get_arg(args::test_id);

    bool cached = (mode == 2);
    bool wide = (mode != 0);  // 8-byte stores for modes 1 and 2
    std::uint32_t dst_addr = cached ? base_addr : (base_addr + MEM_L1_UNCACHED_BASE);

    volatile std::uint64_t* dst64 = (volatile std::uint64_t*)(uintptr_t)dst_addr;
    volatile std::uint8_t* dst8 = (volatile std::uint8_t*)(uintptr_t)dst_addr;
    const std::uint64_t fill_word = 0x5A5A5A5A5A5A5A5AULL;
    const std::uint8_t fill_byte = 0x5A;

    std::uint32_t num_words = size_bytes >> 3;  // full 8-byte stores
    std::uint32_t tail_start = num_words << 3;

    {
        DeviceZoneScopedN("RISCV1");
        for (std::uint32_t iter = 0; iter < num_iterations; iter++) {
            if (wide) {
                for (std::uint32_t i = 0; i < num_words; i++) {
                    dst64[i] = fill_word;
                }
                for (std::uint32_t i = tail_start; i < size_bytes; i++) {
                    dst8[i] = fill_byte;
                }
            } else {
                for (std::uint32_t i = 0; i < size_bytes; i++) {
                    dst8[i] = fill_byte;
                }
            }
            if (cached) {
                flush_l2_cache_range(base_addr, size_bytes);
            }
        }
    }

    DeviceTimestampedData("Test id", test_id);
    // "Number of transactions" = iteration count; host divides zone duration by this
    // to get the amortized per-write cost, and bandwidth = N * size / duration.
    DeviceTimestampedData("Number of transactions", num_iterations);
    DeviceTimestampedData("Transaction size in bytes", size_bytes);
    DeviceTimestampedData("Write path", mode);
}
