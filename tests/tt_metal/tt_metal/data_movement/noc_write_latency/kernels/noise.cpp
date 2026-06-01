// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"

// Background "noise" kernel: continuously blasts posted writes (no ACK wait, so maximum
// injection rate) toward a scratch L1 region on the destination core. Placed on the cores
// along the measured route so the traffic contends for the same NOC links. Silent (no
// DEVICE_PRINT) to avoid slowing the cores or flooding the print stream.
void kernel_main() {
    constexpr uint32_t src_l1_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dst_l1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t dst_noc_x = get_compile_time_arg_val(2);
    constexpr uint32_t dst_noc_y = get_compile_time_arg_val(3);
    constexpr uint32_t num_writes = get_compile_time_arg_val(4);
    constexpr uint32_t transaction_size_bytes = get_compile_time_arg_val(5);

    // Seed a payload in local L1 (uncached alias) so the NOC reads valid data.
    *(volatile tt_l1_ptr uint32_t*)(src_l1_addr + MEM_L1_UNCACHED_BASE) = 0x00C0FFEE;

    uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_l1_addr);

    for (uint32_t i = 0; i < num_writes; i++) {
        noc_async_write<NOC_MAX_BURST_SIZE + 1, true, /*posted=*/true>(
            src_l1_addr, dst_noc_addr, transaction_size_bytes);
    }
    noc_async_posted_writes_flushed();
}
