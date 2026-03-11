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

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#ifdef FABRIC_2D
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#endif
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric;
#ifdef FABRIC_2D
using namespace tt::tt_fabric::mesh::experimental;
#else
using namespace tt::tt_fabric::linear::experimental;
#endif

void kernel_main() {
    size_t idx = 0;
    const uint32_t src_l1_addr    = get_arg_val<uint32_t>(idx++);
    const uint32_t total_size     = get_arg_val<uint32_t>(idx++);
    const uint32_t dst_base_addr  = get_arg_val<uint32_t>(idx++);

#ifdef FABRIC_2D
    const uint16_t dst_mesh_id    = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint8_t  dst_dev_id     = static_cast<uint8_t>(get_arg_val<uint32_t>(idx++));
#endif

    const uint32_t rx_noc_x       = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y       = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr    = get_arg_val<uint32_t>(idx++);

#ifndef FABRIC_2D
    const uint8_t num_hops        = static_cast<uint8_t>(get_arg_val<uint32_t>(idx++));
#endif

    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::allocate_header();

    sender.open<true>();

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

    sender.close();
}
