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

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric;
using namespace tt::tt_fabric::mesh::experimental;

void kernel_main() {
    size_t idx = 0;
    const uint32_t src_l1_addr     = get_arg_val<uint32_t>(idx++);
    const uint32_t total_size      = get_arg_val<uint32_t>(idx++);
    const uint32_t dst_base_addr   = get_arg_val<uint32_t>(idx++);
    const uint16_t dst_mesh_id     = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint8_t  dst_dev_id      = static_cast<uint8_t>(get_arg_val<uint32_t>(idx++));
    const uint32_t rx_noc_x        = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y        = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr     = get_arg_val<uint32_t>(idx++);
    const uint32_t scatter_offset  = get_arg_val<uint32_t>(idx++);

    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::allocate_header();

    sender.open<true>();

    // Scatter: first half to dst_base_addr, second half to dst_base_addr + scatter_offset
    const uint64_t dst_noc_addr0 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr, 0);
    const uint64_t dst_noc_addr1 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr + scatter_offset, 0);
    const uint64_t sem_noc_addr  = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, 0);

    // chunk_size = half of total, scatter splits evenly between addr0 and addr1
    const uint16_t scatter_chunk_size = static_cast<uint16_t>(total_size / 2);

    fabric_unicast_noc_scatter_write(
        &sender,
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        src_l1_addr,
        total_size,
        tt::tt_fabric::NocUnicastScatterCommandHeader{{dst_noc_addr0, dst_noc_addr1}, {scatter_chunk_size}});

    noc_async_writes_flushed();

    // Separate atomic_inc for completion signaling
    fabric_unicast_noc_unicast_atomic_inc(
        &sender,
        packet_header,
        dst_dev_id,
        dst_mesh_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_addr, /*inc=*/1, /*width_bits=*/32));

    sender.close();
}
