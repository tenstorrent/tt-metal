// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Horizon fabric loopback — sender side (runs on chip 0).
//
// 1. Writes `value` to chip `dst_dev_id`'s receive buffer at NOC (dst_noc_x, dst_noc_y)
//    using the flat address from fabric_over_noc_routing_l1_info_t (populated by SetFabricConfig).
// 2. Spins on its local echo buffer until the loopback kernel on chip 1 writes back.
// 3. Exits; host verifies the echo buffer holds the original value.
//
// The flat-address lookup is performed inline here rather than through the templated
// fabric_unicast_noc_unicast_write() overload because that overload requires a live
// FabricSender client_interface (unused on the Horizon crossbar path but still
// required for template instantiation).  The lookup logic is identical to the API.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/fabric_common.h"  // fabric_over_noc_routing_l1_info_t
#include "dev_mem_map.h"                  // ROUTING_TABLE_BASE

void kernel_main() {
    const uint32_t value = get_arg(args::value);
    const uint32_t src_addr = get_arg(args::src_addr);        // local L1 addr to read from
    const uint32_t echo_addr = get_arg(args::echo_addr);      // local L1 addr loopback writes to
    const uint32_t dst_l1_addr = get_arg(args::dst_l1_addr);  // L1 addr on destination chip
    const uint32_t dst_noc_x = get_arg(args::dst_noc_x);      // NOC0 x of target core on dst chip
    const uint32_t dst_noc_y = get_arg(args::dst_noc_y);      // NOC0 y of target core on dst chip
    const uint32_t dst_dev_id = get_arg(args::dst_dev_id);    // logical device ID of destination

    // Pre-fill send buffer.
    *reinterpret_cast<volatile uint32_t*>(src_addr) = value;

    // Look up the 64-bit flat address for core (dst_noc_x, dst_noc_y) on device dst_dev_id.
    // flat_addr[mesh][device][noc_x][noc_y] holds the window base; add the L1 byte offset.
    const auto* rt =
        reinterpret_cast<const tt_l1_ptr tt::tt_fabric::fabric_over_noc_routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const uint64_t remote_addr = rt->flat_addr[0][dst_dev_id][dst_noc_x][dst_noc_y] + dst_l1_addr;

    noc_async_write(src_addr, remote_addr, sizeof(uint32_t));
    noc_async_write_barrier();

    // Spin until the loopback kernel on chip 1 echoes the value back.
    // NOTE: relies on NOC writes to chip 0's L1 being visible to this RISC-V core
    // via a volatile load (no explicit cache invalidation).  Correct on the Zebu
    // emulator where L1 is modelled as a flat SRAM with no private RISC-V cache.
    volatile uint32_t* echo = reinterpret_cast<volatile uint32_t*>(echo_addr);
    while (*echo != value) { /* spin */
    }
}
