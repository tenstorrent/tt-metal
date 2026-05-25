// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Reads the first `marker_bytes` of each (bank, slab) slot from a DRAM buffer
// whose host-side layout is num_shards = num_banks * num_shards_per_bank
// shards. Writes the readback to L1 at `l1_result_addr` in (bank, slab)-major
// order so the host can validate that BufferDistributionSpec's round-robin
// shard placement matches expectations.
void kernel_main() {
    const uint32_t dram_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t l1_result_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_banks = get_compile_time_arg_val(0);
    constexpr uint32_t num_shards_per_bank = get_compile_time_arg_val(1);
    constexpr uint32_t shard_volume_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t marker_bytes = get_compile_time_arg_val(3);

    uint32_t dst = l1_result_addr;
    for (uint32_t bank = 0; bank < num_banks; ++bank) {
        for (uint32_t slab = 0; slab < num_shards_per_bank; ++slab) {
            const uint32_t src_offset = slab * shard_volume_bytes;
            const uint64_t src_noc = get_noc_addr_from_bank_id<true>(bank, dram_base_addr + src_offset);
            noc_async_read(src_noc, dst, marker_bytes);
            dst += marker_bytes;
        }
    }
    noc_async_read_barrier();
}
