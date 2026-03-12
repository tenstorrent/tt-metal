// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unicast scatter sender kernel for auto-packetization integration tests.
// Sends a payload via fabric_unicast_noc_scatter_write, which distributes
// data to TWO destination NOC addresses (scatter). Payloads must fit in a
// single packet (no chunking for scatter -- passthrough only).
//
// After sending the scatter data, sends a separate atomic_inc for completion.
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
//   8: scatter_offset       (u32) - offset for second scatter destination (addr1 = dst_base_addr + scatter_offset)
//   ... fabric connection args (built by append_fabric_connection_rt_args on host)

#include "tx_kernel_common.h"

void kernel_main() {
    size_t idx = 0;
    TX_KERNEL_PARSE_UNICAST_ARGS(idx)
    const uint32_t scatter_offset  = get_arg_val<uint32_t>(idx++);
    TX_KERNEL_SETUP(idx)

    // Scatter: first half to dst_base_addr, second half to dst_base_addr + scatter_offset
    const uint64_t dst_noc_addr0 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr, 0);
    const uint64_t dst_noc_addr1 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr + scatter_offset, 0);
    const uint64_t sem_noc_addr  = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, 0);

    // chunk_size = half of total, scatter splits evenly between addr0 and addr1
    const uint16_t scatter_chunk_size = static_cast<uint16_t>(total_size / 2);

#ifdef FABRIC_2D
    fabric_unicast_noc_scatter_write(
        &sender,
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        src_l1_addr,
        total_size,
        tt::tt_fabric::NocUnicastScatterCommandHeader{{dst_noc_addr0, dst_noc_addr1}, {scatter_chunk_size}});
#else
    fabric_unicast_noc_scatter_write(
        &sender,
        packet_header,
        src_l1_addr,
        total_size,
        tt::tt_fabric::NocUnicastScatterCommandHeader{{dst_noc_addr0, dst_noc_addr1}, {scatter_chunk_size}},
        num_hops);
#endif

    noc_async_writes_flushed();

    // Separate atomic_inc for completion signaling
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
