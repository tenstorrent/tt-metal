// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unicast fused write + atomic_inc sender kernel for auto-packetization tests.
// Sends a payload via fabric_unicast_noc_fused_unicast_with_atomic_inc, which
// auto-packetizes large payloads into chunks. Intermediate chunks are regular
// unicast writes; only the final chunk is fused with atomic_inc.
//
// Does NOT send a separate atomic_inc after -- the fused wrapper handles it.
//
// RT args:
//   0: src_l1_addr         (u32) - L1 address of source data
//   1: total_size           (u32) - total bytes to send (may exceed FABRIC_MAX_PACKET_SIZE)
//   2: dst_base_addr        (u32) - destination buffer base address
//   3: dst_mesh_id          (u32) - destination mesh id (truncated to u16)
//   4: dst_dev_id           (u32) - destination device id (truncated to u8)
//   5: rx_noc_x             (u32) - receiver worker NOC X coordinate
//   6: rx_noc_y             (u32) - receiver worker NOC Y coordinate
//   7: sem_l1_addr          (u32) - receiver semaphore L1 address
//   ... fabric connection args (built by append_fabric_connection_rt_args on host)

#include "tx_kernel_common.h"

void kernel_main() {
    size_t idx = 0;
    TX_KERNEL_PARSE_UNICAST_ARGS(idx)
    TX_KERNEL_SETUP(idx)

    const uint64_t dst_noc_addr = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr, 0);
    const uint64_t sem_noc_addr = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, 0);

    // Fused write + atomic_inc: auto-packetizes, fires atomic_inc on final chunk only
#ifdef FABRIC_2D
    fabric_unicast_noc_fused_unicast_with_atomic_inc(
        &sender,
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        src_l1_addr,
        total_size,
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, sem_noc_addr, /*val=*/1, /*flush=*/true});
#else
    fabric_unicast_noc_fused_unicast_with_atomic_inc(
        &sender,
        packet_header,
        src_l1_addr,
        total_size,
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, sem_noc_addr, /*val=*/1, /*flush=*/true},
        num_hops);
#endif

    // No separate atomic_inc needed -- the fused wrapper handles it.

    TX_KERNEL_TEARDOWN()
}
