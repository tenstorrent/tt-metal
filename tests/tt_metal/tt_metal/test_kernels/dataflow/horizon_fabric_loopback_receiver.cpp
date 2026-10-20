// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Horizon fabric loopback — receiver side (runs on chip 1).
//
// 1. Spins on recv_addr until a non-zero value appears (written by the sender on chip 0).
// 2. Reads the value and writes it back to chip 0's echo buffer at NOC (echo_noc_x, echo_noc_y)
//    using the flat address from this chip's own fabric_over_noc_routing_l1_info_t.
// 3. Exits.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/fabric_common.h"  // fabric_over_noc_routing_l1_info_t
#include "dev_mem_map.h"                  // ROUTING_TABLE_BASE

void kernel_main() {
    const uint32_t recv_addr = get_arg(args::recv_addr);        // local L1 addr sender writes to
    const uint32_t echo_l1_addr = get_arg(args::echo_l1_addr);  // L1 addr on chip 0 to write back to
    const uint32_t echo_noc_x = get_arg(args::echo_noc_x);      // NOC0 x of chip 0 target core
    const uint32_t echo_noc_y = get_arg(args::echo_noc_y);      // NOC0 y of chip 0 target core
    const uint32_t src_dev_id = get_arg(args::src_dev_id);      // logical device ID of chip 0

    // Spin until the sender writes a non-zero value to our receive buffer.
    volatile uint32_t* recv = reinterpret_cast<volatile uint32_t*>(recv_addr);
    while (*recv == 0) { /* spin */
    }
    const uint32_t value = *recv;

    // Echo back to chip 0 via the flat address in our routing table.
    const auto* rt =
        reinterpret_cast<const tt_l1_ptr tt::tt_fabric::fabric_over_noc_routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const uint64_t remote_addr = rt->flat_addr[0][src_dev_id][echo_noc_x][echo_noc_y] + echo_l1_addr;

    // Use recv_addr as the source (holds the value we just read).
    noc_async_write(recv_addr, remote_addr, sizeof(uint32_t));
    noc_async_write_barrier();
}
