// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sparse multicast sender kernel for auto-packetization integration tests.
// Linear-only: uses fabric_sparse_multicast_noc_unicast_write from linear/api.h.
// Does NOT define FABRIC_2D. Single sender, sparse_mask bitmask targeting
// specific devices in the linear chain.
//
// Payloads must fit in a single packet (no chunking for sparse multicast).
// After sending data, sends a separate manual atomic_inc for completion.
//
// RT args:
//   0: src_l1_addr         (u32) - L1 address of source data
//   1: total_size           (u32) - total bytes to send (must be <= FABRIC_MAX_PACKET_SIZE)
//   2: dst_base_addr        (u32) - destination buffer base address
//   3: rx_noc_x             (u32) - receiver worker NOC X coordinate
//   4: rx_noc_y             (u32) - receiver worker NOC Y coordinate
//   5: sem_l1_addr          (u32) - receiver semaphore L1 address
//   6: sparse_mask          (u32) - bitmask of destination hops (bit N = device N hops away)
//   ... fabric connection args (built by append_fabric_connection_rt_args on host)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric;
using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    size_t idx = 0;
    const uint32_t src_l1_addr   = get_arg_val<uint32_t>(idx++);
    const uint32_t total_size    = get_arg_val<uint32_t>(idx++);
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_x      = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y      = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr   = get_arg_val<uint32_t>(idx++);
    const uint16_t sparse_mask   = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));

    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::allocate_header();

    sender.open<true>();

    const uint64_t dst_noc_addr = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr, 0);
    const uint64_t sem_noc_addr = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, 0);

    // Sparse multicast write: targets specific devices in the linear chain
    fabric_sparse_multicast_noc_unicast_write(
        &sender,
        packet_header,
        src_l1_addr,
        total_size,
        tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr},
        sparse_mask);

    noc_async_writes_flushed();

    // Manual atomic_inc for completion signaling.
    // Use sparse multicast for the atomic_inc too so each targeted device gets signaled.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* sem_header = PacketHeaderPool::allocate_header();
    sem_header->to_chip_sparse_multicast(
        tt::tt_fabric::SparseMulticastRoutingCommandHeader<PACKET_HEADER_TYPE>{sparse_mask});
    sem_header->to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader(sem_noc_addr, /*inc=*/1, /*width_bits=*/32));
    sender.wait_for_empty_write_slot();
    sender.send_payload_flush_non_blocking_from_address(
        (uint32_t)sem_header, sizeof(PACKET_HEADER_TYPE));

    sender.close();
}
