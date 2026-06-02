// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Issues num_trids one-packet reads, each tagged with a distinct transaction ID (1..num_trids).
// Polls noc.is_read_trid_flushed() for each slot in a state machine instead of stalling on a
// full barrier, demonstrating that completed slots can be processed while others are in flight.

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    constexpr uint32_t l1_base_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_trids_raw = get_compile_time_arg_val(1);
    constexpr uint32_t num_trids = num_trids_raw < 16 ? num_trids_raw : 15;  // trid 0 reserved
    constexpr uint32_t bytes_per_page = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);
    constexpr uint32_t packed_sub0_coords = get_compile_time_arg_val(4);

    constexpr uint32_t src_x = packed_sub0_coords >> 16;
    constexpr uint32_t src_y = packed_sub0_coords & 0xFFFF;

    Noc noc(noc_index);
    UnicastEndpoint src;
    CoreLocalMem<uint32_t> dst(l1_base_addr);

    // Issue all reads upfront, each with a unique trid (1-indexed to avoid trid 0).
    for (uint32_t i = 1; i <= num_trids; i++) {
        noc.async_read<NocOptions::TXN_ID, NOC_MAX_BURST_SIZE>(
            src,
            dst,
            bytes_per_page,
            {.noc_x = src_x, .noc_y = src_y, .addr = l1_base_addr + (i - 1) * bytes_per_page},
            {.offset_bytes = (i - 1) * bytes_per_page},
            NocOptVals{.trid = i});
    }

    // Poll each slot independently; don't stall on a full barrier.
    uint32_t done_mask = 0;
    constexpr uint32_t expected_mask = (1u << num_trids) - 1;
    while (done_mask != expected_mask) {
        for (uint32_t i = 1; i <= num_trids; i++) {
            uint32_t bit = 1u << (i - 1);
            if (!(done_mask & bit) && noc.is_read_trid_flushed(i)) {
                done_mask |= bit;
            }
        }
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of trids", num_trids);
    DeviceTimestampedData("Bytes per page", bytes_per_page);
}
