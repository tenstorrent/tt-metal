// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Programs source coordinates, page size, and transaction ID into hardware registers once via
// set_async_read_state<ENABLED>, then issues num_pages back-to-back reads via
// async_read_with_state<ENABLED> each inheriting the sticky NOC_PACKET_TAG trid set during
// set_state.  Waits on the per-trid barrier rather than a full read barrier.

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    constexpr uint32_t l1_base_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t bytes_per_page = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);
    constexpr uint32_t packed_sub0_coords = get_compile_time_arg_val(4);

    constexpr uint32_t src_x = packed_sub0_coords >> 16;
    constexpr uint32_t src_y = packed_sub0_coords & 0xFFFF;

    constexpr uint8_t trid = 1;

    Noc noc(noc_index);
    UnicastEndpoint src;
    CoreLocalMem<uint32_t> dst(l1_base_addr);

    noc.set_async_read_state<NocOptions::TXN_ID, NOC_MAX_BURST_SIZE>(
        src,
        bytes_per_page,
        {.noc_x = src_x, .noc_y = src_y, .addr = l1_base_addr},
        NocOptVals{.trid = trid});

    for (uint32_t i = 0; i < num_pages; i++) {
        noc.async_read_with_state<NocOptions::TXN_ID, NOC_MAX_BURST_SIZE>(
            src,
            dst,
            bytes_per_page,
            {.noc_x = src_x, .noc_y = src_y, .addr = l1_base_addr + i * bytes_per_page},
            {.offset_bytes = i * bytes_per_page},
            NocOptVals{.trid = trid});
    }

    noc.async_read_barrier<NocOptions::TXN_ID>({.trid = trid});

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of pages", num_pages);
    DeviceTimestampedData("Bytes per page", bytes_per_page);
}
