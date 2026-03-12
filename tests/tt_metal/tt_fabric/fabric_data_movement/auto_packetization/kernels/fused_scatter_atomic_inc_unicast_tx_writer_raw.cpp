// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unicast fused scatter write + atomic_inc sender kernel for auto-packetization tests.
// Sends a payload via fabric_unicast_noc_fused_scatter_write_atomic_inc, which
// distributes data to TWO destination NOC addresses AND fires an atomic_inc on
// the final chunk. Payloads must fit in a single packet (passthrough).
//
// Does NOT send a separate atomic_inc after -- the fused wrapper handles it.
//
// RT args:
//   0: src_l1_addr         (u32) - L1 address of source data
//   1: total_size           (u32) - total bytes to send (must be <= FABRIC_MAX_PACKET_SIZE)
//   2: dst_base_addr        (u32) - destination buffer base address
//   3: dst_mesh_id          (u32) - destination mesh id (truncated to u16)
//   4: dst_dev_id           (u32) - destination device id (truncated to u8)
//   5: rx_noc_x             (u32) - receiver worker NOC X coordinate
//   6: rx_noc_y             (u32) - receiver worker NOC Y coordinate
//   7: sem_l1_addr          (u32) - receiver semaphore L1 address
//   8: scatter_offset       (u32) - offset for second scatter destination
//   ... fabric connection args (built by append_fabric_connection_rt_args on host)

#include "tx_kernel_common.h"

void kernel_main() {
    size_t idx = 0;
    TX_KERNEL_PARSE_UNICAST_ARGS(idx)
    const uint32_t scatter_offset  = get_arg_val<uint32_t>(idx++);
    TX_KERNEL_SETUP(idx)

    const uint64_t dst_noc_addr0 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr, 0);
    const uint64_t dst_noc_addr1 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr + scatter_offset, 0);
    const uint64_t sem_noc_addr  = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, 0);

    // chunk_size = half of total, scatter splits evenly
    const uint16_t scatter_chunk_size = static_cast<uint16_t>(total_size / 2);

    // Fused scatter + atomic_inc: data to 2 addresses + atomic_inc on completion
#ifdef FABRIC_2D
    fabric_unicast_noc_fused_scatter_write_atomic_inc(
        &sender,
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        src_l1_addr,
        total_size,
        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
            {dst_noc_addr0, dst_noc_addr1},
            sem_noc_addr,
            {scatter_chunk_size},
            /*val=*/1,
            /*flush=*/true});
#else
    fabric_unicast_noc_fused_scatter_write_atomic_inc(
        &sender,
        packet_header,
        src_l1_addr,
        total_size,
        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
            {dst_noc_addr0, dst_noc_addr1},
            sem_noc_addr,
            {scatter_chunk_size},
            /*val=*/1,
            /*flush=*/true},
        num_hops);
#endif

    // No separate atomic_inc needed -- the fused wrapper handles it.

    TX_KERNEL_TEARDOWN()
}
