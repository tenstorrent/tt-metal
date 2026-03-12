// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Raw-size unicast sender kernel for auto-packetization integration tests.
// Sends a payload of `total_size` bytes via the auto-packetizing
// fabric_unicast_noc_unicast_write wrapper, which transparently chunks
// payloads larger than FABRIC_MAX_PACKET_SIZE.
//
// After sending the data payload, sends an atomic_inc to signal completion
// to the receiver kernel.
//
// RT args:
//   0: src_l1_addr         (u32) - L1 address of source data (DRAM buffer base)
//   1: total_size           (u32) - total bytes to send (may exceed FABRIC_MAX_PACKET_SIZE)
//   2: dst_base_addr        (u32) - destination buffer base address (L1 or DRAM offset)
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

    // Compute 64-bit NOC addresses on device (using device-side safe_get_noc_addr)
    const uint64_t dst_noc_addr = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr, /*NOC_INDEX=*/0);
    const uint64_t sem_noc_addr = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);

    // Send payload using auto-packetizing wrapper.
    // This call transparently chunks total_size into FABRIC_MAX_PACKET_SIZE pieces.
#ifdef FABRIC_2D
    fabric_unicast_noc_unicast_write(
        &sender,
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        src_l1_addr,
        total_size,
        tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr});
#else
    fabric_unicast_noc_unicast_write(
        &sender,
        packet_header,
        src_l1_addr,
        total_size,
        tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr},
        num_hops);
#endif

    // Ensure all writes are flushed before sending completion signal
    noc_async_writes_flushed();

    // Signal completion via atomic increment on receiver semaphore
#ifdef FABRIC_2D
    fabric_unicast_noc_unicast_atomic_inc(
        &sender,
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_addr, /*inc=*/1, /*width_bits=*/32));
#else
    fabric_unicast_noc_unicast_atomic_inc(
        &sender,
        packet_header,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_addr, /*inc=*/1, /*width_bits=*/32),
        num_hops);
#endif

    TX_KERNEL_TEARDOWN()
}
